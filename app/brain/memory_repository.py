"""In-process adapter implementing the same contract as the Neo4j repository.

Used by the unit/scenario tests (no infrastructure needed) and for running the API locally
without Docker (BRAIN_BACKEND=memory). Not for production: no persistence, single process.
"""

from datetime import datetime

from app.core.errors import SessionOwnershipError
from app.core.models import (
    ChatMessage,
    LifeArea,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    UserProfile,
)


class InMemoryBrainRepository:
    def __init__(self) -> None:
        self.profiles: dict[str, UserProfile] = {}
        self.sessions: dict[str, str] = {}  # session_id → user_id
        self.messages: dict[str, list[ChatMessage]] = {}
        self.memories: dict[str, MemoryItem] = {}  # memory_id → item
        self.owner: dict[str, str] = {}  # memory_id → user_id
        self.sources: dict[str, str | None] = {}  # memory_id → source message id
        self.supersedes: dict[str, str] = {}  # new memory_id → old memory_id

    async def ping(self) -> bool:
        return True

    # --- profile -------------------------------------------------------------------------

    async def get_profile(self, user_id: str) -> UserProfile | None:
        profile = self.profiles.get(user_id)
        return profile.model_copy() if profile else None

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        existing = self.profiles.get(profile.user_id)
        if existing and existing.created_at:
            profile = profile.model_copy(update={"created_at": existing.created_at})
        self.profiles[profile.user_id] = profile.model_copy()
        return profile

    # --- short-term conversation ---------------------------------------------------------

    async def ensure_session(self, user_id: str, session_id: str, now: datetime) -> None:
        owner = self.sessions.setdefault(session_id, user_id)
        if owner != user_id:
            raise SessionOwnershipError(f"session {session_id} belongs to another user")
        self.messages.setdefault(session_id, [])

    async def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        self.messages.setdefault(session_id, []).extend(messages)

    async def recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        return list(self.messages.get(session_id, [])[-limit:]) if limit > 0 else []

    # --- long-term memory ----------------------------------------------------------------

    def _user_memories(self, user_id: str) -> list[MemoryItem]:
        return [m for mid, m in self.memories.items() if self.owner[mid] == user_id]

    async def query_memories(
        self,
        user_id: str,
        *,
        areas: list[LifeArea] | None = None,
        kinds: list[MemoryKind] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        items = [
            m
            for m in self._user_memories(user_id)
            if m.status == MemoryStatus.ACTIVE
            and (not areas or m.life_area in areas)
            and (not kinds or m.kind in kinds)
        ]
        items.sort(key=lambda m: (m.importance, m.updated_at), reverse=True)
        return [m.model_copy() for m in items[:limit]]

    async def find_memory(self, user_id: str, key: str) -> MemoryItem | None:
        for m in self._user_memories(user_id):
            if m.key == key and m.status == MemoryStatus.ACTIVE:
                return m.model_copy()
        return None

    async def create_memory(
        self, user_id: str, item: MemoryItem, source_message_id: str | None
    ) -> None:
        self.memories[item.id] = item.model_copy()
        self.owner[item.id] = user_id
        self.sources[item.id] = source_message_id

    async def update_memory(self, item: MemoryItem) -> None:
        self.memories[item.id] = item.model_copy()

    async def supersede_memory(
        self, user_id: str, old_id: str, new_item: MemoryItem, source_message_id: str | None
    ) -> None:
        old = self.memories[old_id]
        self.memories[old_id] = old.model_copy(
            update={"status": MemoryStatus.SUPERSEDED, "updated_at": new_item.updated_at}
        )
        await self.create_memory(user_id, new_item, source_message_id)
        self.supersedes[new_item.id] = old_id

    async def set_memory_status(self, memory_id: str, status: MemoryStatus, now: datetime) -> None:
        item = self.memories[memory_id]
        self.memories[memory_id] = item.model_copy(update={"status": status, "updated_at": now})

    async def list_memories(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryItem]:
        items = [
            m
            for m in self._user_memories(user_id)
            if include_inactive or m.status == MemoryStatus.ACTIVE
        ]
        return sorted(items, key=lambda m: m.updated_at, reverse=True)

    async def forget_memories(self, user_id: str) -> int:
        ids = [mid for mid, uid in self.owner.items() if uid == user_id]
        for mid in ids:
            self.memories.pop(mid, None)
            self.owner.pop(mid, None)
            self.sources.pop(mid, None)
        return len(ids)
