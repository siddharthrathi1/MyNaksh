"""Applies extraction results to the Shared Brain.

Policy (what gets written and how):
- Gate:        drop candidates below the confidence threshold; cap candidates per turn.
- New key:     create.
- Same key, ADD, same facts:   reinforce (bump confidence & mention count, refresh recency).
- Same key, UPDATE or changed facts (date/area/value): supersede — the old node is kept with
               status=superseded and linked via SUPERSEDES, giving an audit trail.
- REMOVE:      mark retracted (soft delete; still auditable, never retrieved).
- Profile:     single-valued fields are overwritten (latest statement wins).
"""

import logging
from datetime import datetime

from app.brain.repository import BrainRepository
from app.core.models import MemoryItem, MemoryStatus
from app.memory.models import ChangeAction, ExtractedMemory, ExtractionResult, MemoryChange, Operation
from app.profile.service import ProfileService, ProfileUpdate

logger = logging.getLogger(__name__)

REINFORCE_BOOST = 0.05


def _materially_differs(existing: MemoryItem, cand: ExtractedMemory) -> bool:
    """A restatement with different facts is a change, not a reinforcement.
    Pure rewording of `text` is ignored (LLMs paraphrase)."""
    return (
        (cand.target_date is not None and cand.target_date != existing.target_date)
        or (cand.value is not None and cand.value != existing.value)
        or cand.life_area != existing.life_area
    )


def _change(action: ChangeAction, item: MemoryItem) -> MemoryChange:
    return MemoryChange(action=action, kind=item.kind.value, key=item.key, text=item.text, label=item.label)


class MemoryUpdater:
    def __init__(
        self,
        repo: BrainRepository,
        profiles: ProfileService,
        *,
        confidence_threshold: float,
        max_per_turn: int,
    ):
        self._repo = repo
        self._profiles = profiles
        self._threshold = confidence_threshold
        self._max_per_turn = max_per_turn

    async def apply(
        self, user_id: str, result: ExtractionResult, source_message_id: str | None, now: datetime
    ) -> list[MemoryChange]:
        changes: list[MemoryChange] = []

        if result.profile_updates:
            update = ProfileUpdate.lenient(result.profile_updates)
            if not update.is_empty():
                _, changed = await self._profiles.apply(user_id, update)
                changes += [
                    MemoryChange(
                        action=ChangeAction.PROFILE_UPDATED, kind="profile", key=f,
                        text=str(getattr(update, f)), label="user_profile",
                    )
                    for f in changed
                ]

        accepted = [c for c in result.memories if c.confidence >= self._threshold]
        for dropped in result.memories:
            if dropped.confidence < self._threshold:
                logger.info("memory below threshold dropped key=%s conf=%.2f", dropped.key, dropped.confidence)

        for cand in accepted[: self._max_per_turn]:
            change = await self._apply_one(user_id, cand, source_message_id, now)
            if change:
                changes.append(change)

        if changes:
            logger.info(
                "memory_update user=%s extractor=%s changes=%s",
                user_id, result.extractor, [(c.action.value, c.key) for c in changes],
            )
        return changes

    async def _apply_one(
        self, user_id: str, cand: ExtractedMemory, source_message_id: str | None, now: datetime
    ) -> MemoryChange | None:
        existing = await self._repo.find_memory(user_id, cand.key)

        if cand.operation == Operation.REMOVE:
            if not existing:
                return None  # nothing to retract
            await self._repo.set_memory_status(existing.id, MemoryStatus.RETRACTED, now)
            return _change(ChangeAction.RETRACTED, existing)

        if existing is None:
            item = self._new_item(cand, now)
            await self._repo.create_memory(user_id, item, source_message_id)
            return _change(ChangeAction.CREATED, item)

        if cand.operation == Operation.UPDATE or _materially_differs(existing, cand):
            item = self._new_item(cand, now, previous=existing)
            await self._repo.supersede_memory(user_id, existing.id, item, source_message_id)
            return _change(ChangeAction.UPDATED, item)

        reinforced = existing.model_copy(
            update={
                "confidence": min(1.0, max(existing.confidence, cand.confidence) + REINFORCE_BOOST),
                "importance": max(existing.importance, cand.importance),
                "mention_count": existing.mention_count + 1,
                "updated_at": now,
            }
        )
        await self._repo.update_memory(reinforced)
        return _change(ChangeAction.REINFORCED, reinforced)

    @staticmethod
    def _new_item(cand: ExtractedMemory, now: datetime, previous: MemoryItem | None = None) -> MemoryItem:
        return MemoryItem(
            kind=cand.kind,
            key=cand.key,
            text=cand.text,
            life_area=cand.life_area,
            target_date=cand.target_date,
            timeframe_raw=cand.timeframe_raw,
            value=cand.value,
            # An explicit correction is the strongest signal we get.
            confidence=1.0 if cand.operation == Operation.UPDATE else cand.confidence,
            importance=max(cand.importance, previous.importance) if previous else cand.importance,
            mention_count=(previous.mention_count + 1) if previous else 1,
            created_at=now,
            updated_at=now,
        )
