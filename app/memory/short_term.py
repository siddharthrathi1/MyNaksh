from app.brain.repository import BrainRepository
from app.core.models import ChatMessage


class ShortTermMemory:
    """Per-session conversation window.

    Backed by the graph today (messages double as provenance for EXTRACTED_FROM edges).
    At scale the hot window would move to Redis (LRANGE/LTRIM, TTL per session) with the graph
    kept as the durable log — this class is the seam for that change.
    """

    def __init__(self, repo: BrainRepository, window_messages: int):
        self._repo = repo
        self._window = window_messages

    async def recent(self, session_id: str) -> list[ChatMessage]:
        return await self._repo.recent_messages(session_id, self._window)

    async def append(self, session_id: str, messages: list[ChatMessage]) -> None:
        await self._repo.append_messages(session_id, messages)
