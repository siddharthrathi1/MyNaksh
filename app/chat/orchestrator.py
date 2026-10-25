"""The chat pipeline:

  message → load profile & short-term history → understand query → select context
          → build prompt → LLM → persist turn → respond → update long-term memory

Failure policy: a turn should almost never fail outright.
- Graph down  → answer without memory, skip persistence (degraded=true).
- LLM down    → canned reply (degraded=true); memory extraction still runs via rule fallback.
- Memory update errors are logged and swallowed — they must never break the user's turn.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.brain.repository import BrainRepository
from app.context.prompt import PromptBuilder
from app.context.query import HybridQueryAnalyzer, QueryAnalysis
from app.context.selector import ContextSelector
from app.core.clock import Clock
from app.core.errors import GraphUnavailableError, LLMError, LLMUnavailableError
from app.core.models import ChatMessage, Role, UserProfile
from app.llm.base import LLMProvider
from app.memory.extractor import Extractor
from app.memory.models import MemoryChange
from app.memory.short_term import ShortTermMemory
from app.memory.updater import MemoryUpdater
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)

FALLBACK_REPLY = (
    "I'm having trouble connecting to my guidance engine right now. "
    "Please try again in a moment — I've noted what you shared."
)
EXISTING_MEMORY_LIMIT = 30  # known memories shown to the extractor for key reuse


@dataclass
class MemoryJob:
    """Everything needed to update long-term memory after the reply — can run in the background."""

    user_id: str
    message: str
    source_message_id: str | None
    history: list[ChatMessage]
    now: datetime
    memorable: bool | None = None  # analyzer's verdict; None → extractor's own gate decides


@dataclass
class ChatResult:
    response: str
    user_id: str
    session_id: str
    context_used: list[str]
    analysis: QueryAnalysis
    degraded_reasons: list[str] = field(default_factory=list)
    memory_job: MemoryJob | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)


class ChatOrchestrator:
    def __init__(
        self,
        *,
        repo: BrainRepository,
        profiles: ProfileService,
        short_term: ShortTermMemory,
        analyzer: HybridQueryAnalyzer,
        selector: ContextSelector,
        prompts: PromptBuilder,
        llm: LLMProvider,
        extractor: Extractor,
        updater: MemoryUpdater,
        clock: Clock,
    ):
        self._repo = repo
        self._profiles = profiles
        self._short_term = short_term
        self._analyzer = analyzer
        self._selector = selector
        self._prompts = prompts
        self._llm = llm
        self._extractor = extractor
        self._updater = updater
        self._clock = clock

    async def handle(self, user_id: str, session_id: str, message: str) -> ChatResult:
        started = time.perf_counter()
        now = self._clock()
        degraded: list[str] = []

        # 1. Load user + short-term context. (SessionOwnershipError propagates → 409.)
        graph_ok = True
        try:
            profile = await self._profiles.get_or_create(user_id)
            await self._repo.ensure_session(user_id, session_id, now)
            history = await self._short_term.recent(session_id)
        except GraphUnavailableError:
            logger.warning("graph unavailable; serving turn without memory user=%s", user_id)
            graph_ok, profile, history = False, UserProfile(user_id=user_id), []
            degraded.append("memory_unavailable")

        # 2–3. Understand the query, select only the relevant context.
        analysis = await self._analyzer.analyze(message, history)
        context = await self._selector.select(profile, analysis, history, now, graph_available=graph_ok)
        if context.degraded and graph_ok:
            degraded.append("memory_unavailable")

        # 4–5. Build prompt and generate.
        request = self._prompts.build(context, message, now)
        try:
            reply = (await self._llm.generate(request)).text.strip() or FALLBACK_REPLY
        except (LLMUnavailableError, LLMError):
            logger.error("all LLM providers failed; serving fallback reply user=%s", user_id)
            reply = FALLBACK_REPLY
            degraded.append("llm_unavailable")

        # 6. Persist the turn (user message first: it is the provenance for extracted memories).
        user_msg = ChatMessage(role=Role.USER, content=message, created_at=now)
        # Strictly after the user message so ordering by timestamp is always well-defined.
        reply_at = max(self._clock(), now + timedelta(milliseconds=1))
        assistant_msg = ChatMessage(role=Role.ASSISTANT, content=reply, created_at=reply_at)
        if graph_ok:
            try:
                await self._short_term.append(session_id, [user_msg, assistant_msg])
            except GraphUnavailableError:
                graph_ok = False
                degraded.append("persistence_failed")

        logger.info(
            "chat_turn user=%s session=%s analyzer=%s intent=%s areas=%s memorable=%s context=%s "
            "degraded=%s total_ms=%.0f",
            user_id, session_id, analysis.analyzer, analysis.intent.value,
            [a.value for a in analysis.life_areas], analysis.memorable, context.context_used, degraded,
            (time.perf_counter() - started) * 1000,
        )

        job = MemoryJob(user_id, message, user_msg.id, history, now, analysis.memorable) if graph_ok else None
        return ChatResult(
            response=reply,
            user_id=user_id,
            session_id=session_id,
            context_used=context.context_used,
            analysis=analysis,
            degraded_reasons=sorted(set(degraded)),
            memory_job=job,
        )

    async def update_memory(self, job: MemoryJob) -> list[MemoryChange]:
        """Step 7: extract and persist long-term memories. Never raises."""
        try:
            existing = await self._repo.query_memories(job.user_id, limit=EXISTING_MEMORY_LIMIT)
            result = await self._extractor.extract(job.message, job.now, existing, job.history, job.memorable)
            if result.is_empty():
                return []
            return await self._updater.apply(job.user_id, result, job.source_message_id, job.now)
        except Exception:
            logger.exception("memory update failed user=%s", job.user_id)
            return []
