"""Step 2 of context selection: pick the *relevant* slice of the Shared Brain for this turn.

Never sends the whole graph or the whole conversation:
- profile           → always (compact, ~50 tokens, needed for any personalised answer)
- topical memories  → only those linked to the life areas detected in the query
- preferences       → always (they shape *how* to answer, regardless of topic)
- memory-recall     → broader retrieval, but still ranked and capped
- history           → last N turns of *this* session only, trimmed to a token budget
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from app.brain.repository import BrainRepository
from app.context.query import QueryAnalysis
from app.core.errors import GraphUnavailableError
from app.core.models import ChatMessage, MemoryItem, MemoryKind, UserProfile

logger = logging.getLogger(__name__)

# Ranking weights. Importance dominates; recency decays with a ~90-day time constant so stale
# facts fade from context without being deleted (soft memory decay).
W_IMPORTANCE, W_CONFIDENCE, W_RECENCY = 0.5, 0.3, 0.2
RECENCY_TAU_DAYS = 90.0


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic: good enough for budgeting without a tokenizer dependency."""
    return math.ceil(len(text) / 4)


def score_memory(m: MemoryItem, now: datetime) -> float:
    age_days = max(0.0, (now - m.updated_at).total_seconds() / 86400)
    recency = math.exp(-age_days / RECENCY_TAU_DAYS)
    return W_IMPORTANCE * m.importance + W_CONFIDENCE * m.confidence + W_RECENCY * recency


@dataclass
class SelectedContext:
    profile: UserProfile
    memories: list[MemoryItem] = field(default_factory=list)
    history: list[ChatMessage] = field(default_factory=list)
    context_used: list[str] = field(default_factory=list)
    degraded: bool = False


class ContextSelector:
    def __init__(
        self,
        repo: BrainRepository,
        *,
        max_memories: int,
        confidence_threshold: float,
        token_budget: int,
        history_messages: int,
    ):
        self._repo = repo
        self._max_memories = max_memories
        self._threshold = confidence_threshold
        self._token_budget = token_budget
        self._history_messages = history_messages

    async def select(
        self,
        profile: UserProfile,
        analysis: QueryAnalysis,
        history: list[ChatMessage],
        now: datetime,
        *,
        graph_available: bool = True,
    ) -> SelectedContext:
        memories: list[MemoryItem] = []
        degraded = not graph_available
        if graph_available:
            try:
                memories = await self._retrieve(profile.user_id, analysis)
            except GraphUnavailableError:
                logger.warning("graph unavailable during retrieval; answering without memories")
                degraded = True

        ranked = sorted(
            (m for m in memories if m.confidence >= self._threshold),
            key=lambda m: score_memory(m, now),
            reverse=True,
        )[: self._max_memories]

        # Token budget: memories get priority over older history (they are denser signal).
        budget = self._token_budget
        kept_memories = []
        for m in ranked:
            cost = estimate_tokens(m.text) + 10
            if cost > budget:
                break
            kept_memories.append(m)
            budget -= cost

        kept_history: list[ChatMessage] = []
        for msg in reversed(history[-self._history_messages :]):
            cost = estimate_tokens(msg.content) + 4
            if cost > budget:
                break
            kept_history.insert(0, msg)
            budget -= cost

        return SelectedContext(
            profile=profile,
            memories=kept_memories,
            history=kept_history,
            context_used=self._labels(profile, kept_memories, kept_history),
            degraded=degraded,
        )

    async def _retrieve(self, user_id: str, analysis: QueryAnalysis) -> list[MemoryItem]:
        # Over-fetch, then rank in Python: the DB orders by importance only; final scoring
        # (recency, confidence) is cheap here and keeps Cypher simple.
        fetch = self._max_memories * 3
        topical: list[MemoryItem] = []
        if analysis.is_memory_query:
            topical = await self._repo.query_memories(user_id, areas=analysis.life_areas or None, limit=fetch)
        elif analysis.life_areas:
            topical = await self._repo.query_memories(user_id, areas=analysis.life_areas, limit=fetch)
        # No detected topic → no topical memories. Injecting unrelated facts hurts answers.

        preferences = await self._repo.query_memories(user_id, kinds=[MemoryKind.PREFERENCE], limit=3)
        seen = {m.id for m in topical}
        return topical + [p for p in preferences if p.id not in seen]

    @staticmethod
    def _labels(profile: UserProfile, memories: list[MemoryItem], history: list[ChatMessage]) -> list[str]:
        labels: list[str] = []
        if profile.has_any_details():
            labels.append("user_profile")
        if profile.sun_sign:
            labels.append("astrology_profile")
        for m in memories:
            if m.label not in labels:
                labels.append(m.label)
        if history:
            labels.append("conversation_history")
        return labels
