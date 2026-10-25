"""Composition root: the one place where concrete implementations are chosen and wired.

Everything else depends on interfaces (BrainRepository, LLMProvider, Extractor), so tests build
a Container with in-memory/fake parts and production builds it from Settings.
"""

import logging
from dataclasses import dataclass

from app.brain.client import GraphClient
from app.brain.memory_repository import InMemoryBrainRepository
from app.brain.neo4j_repository import Neo4jBrainRepository
from app.brain.repository import BrainRepository
from app.chat.orchestrator import ChatOrchestrator
from app.config import Settings
from app.context.prompt import PromptBuilder
from app.context.query import HybridQueryAnalyzer, LLMQueryAnalyzer, QueryAnalyzer
from app.context.selector import ContextSelector
from app.core.clock import Clock, utc_now
from app.llm.base import LLMProvider
from app.llm.factory import build_llm
from app.memory.extractor import HybridExtractor, LLMExtractor
from app.memory.rules import RuleBasedExtractor
from app.memory.short_term import ShortTermMemory
from app.memory.updater import MemoryUpdater
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: BrainRepository
    llm: LLMProvider
    profiles: ProfileService
    orchestrator: ChatOrchestrator
    graph_client: GraphClient | None = None

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        repo: BrainRepository | None = None,
        llm: LLMProvider | None = None,
        clock: Clock = utc_now,
    ) -> "Container":
        graph_client = None
        if repo is None:
            if settings.brain_backend == "neo4j":
                graph_client = GraphClient(settings)
                repo = Neo4jBrainRepository(graph_client)
            else:
                repo = InMemoryBrainRepository()
        llm = llm or build_llm(settings)

        use_llm_extraction = settings.extraction_mode == "llm" or (
            settings.extraction_mode == "auto" and settings.llm_provider != "fake"
        )
        extractor = HybridExtractor(LLMExtractor(llm) if use_llm_extraction else None, RuleBasedExtractor())
        use_llm_analysis = settings.analysis_mode == "llm" or (
            settings.analysis_mode == "auto" and settings.llm_provider != "fake"
        )
        analyzer = HybridQueryAnalyzer(LLMQueryAnalyzer(llm) if use_llm_analysis else None, QueryAnalyzer())

        profiles = ProfileService(repo, clock)
        orchestrator = ChatOrchestrator(
            repo=repo,
            profiles=profiles,
            short_term=ShortTermMemory(repo, settings.short_term_turns * 2),
            analyzer=analyzer,
            selector=ContextSelector(
                repo,
                max_memories=settings.max_memories_in_context,
                confidence_threshold=settings.memory_confidence_threshold,
                token_budget=settings.context_token_budget,
                history_messages=settings.short_term_turns * 2,
            ),
            prompts=PromptBuilder(),
            llm=llm,
            extractor=extractor,
            updater=MemoryUpdater(
                repo,
                profiles,
                confidence_threshold=settings.memory_confidence_threshold,
                max_per_turn=settings.max_memories_per_turn,
            ),
            clock=clock,
        )
        return cls(settings, repo, llm, profiles, orchestrator, graph_client)

    async def startup(self) -> None:
        if self.graph_client:
            try:
                await self.graph_client.init_schema()
                logger.info("Neo4j schema initialised")
            except Exception:
                # Start anyway: chat degrades to no-memory mode until the graph is reachable.
                logger.exception("Neo4j schema init failed; running in degraded mode")

    async def shutdown(self) -> None:
        if self.graph_client:
            await self.graph_client.close()
