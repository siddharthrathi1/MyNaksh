import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncManagedTransaction, NotificationDisabledCategory
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from app.brain.schema import CONSTRAINTS, INDEXES, LIFE_AREAS, SEED_LIFE_AREAS
from app.config import Settings
from app.core.errors import GraphUnavailableError

logger = logging.getLogger(__name__)

# Errors that mean "the graph is not reachable right now" → callers degrade gracefully.
_UNAVAILABLE = (ServiceUnavailable, SessionExpired, TransientError, OSError)

Statement = tuple[str, dict[str, Any]]


class GraphClient:
    """Thin wrapper over the async Neo4j driver.

    - Uses *managed* transactions (`execute_read` / `execute_write`), which the driver retries
      automatically on transient cluster errors (leader switch, deadlock).
    - Translates connectivity failures into `GraphUnavailableError`.
    - The driver owns a connection pool; one client instance is shared per process.
    """

    def __init__(self, settings: Settings):
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_pool_size,
            connection_acquisition_timeout=settings.neo4j_timeout_s,
            # Memory reads match every relationship type up front; on a young graph some types
            # don't exist yet and the server warns on each query. Expected, so silence it.
            notifications_disabled_categories=[NotificationDisabledCategory.UNRECOGNIZED],
        )

    async def close(self) -> None:
        await self._driver.close()

    async def ping(self) -> bool:
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception:
            logger.warning("Neo4j connectivity check failed", exc_info=True)
            return False

    async def init_schema(self) -> None:
        for stmt in CONSTRAINTS + INDEXES:
            await self.write(stmt)
        await self.write(SEED_LIFE_AREAS, names=LIFE_AREAS)

    async def read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async def work(tx: AsyncManagedTransaction):
            result = await tx.run(query, **params)
            return [record.data() async for record in result]

        return await self._execute(work, write=False)

    async def write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        return await self.write_many([(query, params)])

    async def write_many(self, statements: list[Statement]) -> list[dict[str, Any]]:
        """Run several statements atomically in one transaction; returns the last result."""

        async def work(tx: AsyncManagedTransaction):
            rows: list[dict[str, Any]] = []
            for query, params in statements:
                result = await tx.run(query, **params)
                rows = [record.data() async for record in result]
            return rows

        return await self._execute(work, write=True)

    async def _execute(self, work, *, write: bool):
        try:
            async with self._driver.session() as session:
                if write:
                    return await session.execute_write(work)
                return await session.execute_read(work)
        except _UNAVAILABLE as exc:
            raise GraphUnavailableError(str(exc)) from exc
