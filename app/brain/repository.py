"""Storage port for the Shared Brain.

The rest of the application depends only on this protocol. Two adapters implement it:
- `Neo4jBrainRepository`  — production
- `InMemoryBrainRepository` — tests and running locally without Docker

All read methods return only ACTIVE memories unless stated otherwise.
"""

from datetime import datetime
from typing import Protocol

from app.core.models import ChatMessage, LifeArea, MemoryItem, MemoryKind, MemoryStatus, UserProfile


class BrainRepository(Protocol):
    async def ping(self) -> bool: ...

    # --- profile -------------------------------------------------------------------------
    async def get_profile(self, user_id: str) -> UserProfile | None: ...

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        """Full replace of profile fields (merging is the service layer's job)."""
        ...

    # --- short-term conversation ---------------------------------------------------------
    async def ensure_session(self, user_id: str, session_id: str, now: datetime) -> None:
        """Create the session if new. Raises SessionOwnershipError if owned by another user."""
        ...

    async def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None: ...

    async def recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        """Last `limit` messages, oldest first."""
        ...

    # --- long-term memory ----------------------------------------------------------------
    async def query_memories(
        self,
        user_id: str,
        *,
        areas: list[LifeArea] | None = None,
        kinds: list[MemoryKind] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Active memories filtered by life area / kind, most important & recent first."""
        ...

    async def find_memory(self, user_id: str, key: str) -> MemoryItem | None:
        """The active memory with this key, if any."""
        ...

    async def create_memory(
        self, user_id: str, item: MemoryItem, source_message_id: str | None
    ) -> None: ...

    async def update_memory(self, item: MemoryItem) -> None:
        """Update mutable fields in place (reinforcement)."""
        ...

    async def supersede_memory(
        self, user_id: str, old_id: str, new_item: MemoryItem, source_message_id: str | None
    ) -> None:
        """Atomically mark `old_id` SUPERSEDED and create `new_item` linked via SUPERSEDES."""
        ...

    async def set_memory_status(self, memory_id: str, status: MemoryStatus, now: datetime) -> None: ...

    async def list_memories(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryItem]: ...

    async def forget_memories(self, user_id: str) -> int:
        """Hard-delete all long-term memories for a user (right to be forgotten)."""
        ...
