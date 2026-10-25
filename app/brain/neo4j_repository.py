"""Neo4j adapter for the Shared Brain.

Labels and relationship types cannot be Cypher parameters, so they are interpolated — but
only from the fixed whitelist in `schema.KIND_GRAPH`, never from user input.
"""

from datetime import datetime
from typing import Any

from app.brain.client import GraphClient
from app.brain.schema import KIND_GRAPH, MEMORY_RELS
from app.core.errors import SessionOwnershipError
from app.core.models import (
    ChatMessage,
    LifeArea,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Role,
    UserProfile,
)


def _native(value: Any) -> Any:
    """neo4j.time.{Date,Time,DateTime} → stdlib equivalents."""
    return value.to_native() if hasattr(value, "to_native") else value


def _memory_props(item: MemoryItem) -> dict[str, Any]:
    # life_area is a relationship (ABOUT), not a property — single source of truth.
    return {
        "id": item.id,
        "kind": item.kind.value,
        "key": item.key,
        "text": item.text,
        "target_date": item.target_date,
        "timeframe_raw": item.timeframe_raw,
        "value": item.value,
        "confidence": item.confidence,
        "importance": item.importance,
        "status": item.status.value,
        "mention_count": item.mention_count,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _to_memory(props: dict[str, Any], area: str | None) -> MemoryItem:
    data = {k: _native(v) for k, v in props.items()}
    return MemoryItem(**data, life_area=LifeArea(area or LifeArea.GENERAL))


_RETURN_MEMORY = "RETURN properties(m) AS m, a.name AS area"


class Neo4jBrainRepository:
    def __init__(self, client: GraphClient):
        self._db = client

    async def ping(self) -> bool:
        return await self._db.ping()

    # --- profile -------------------------------------------------------------------------

    async def get_profile(self, user_id: str) -> UserProfile | None:
        rows = await self._db.read(
            """
            MATCH (u:User {id: $id})
            OPTIONAL MATCH (u)-[:BORN_IN]->(p:Place)
            OPTIONAL MATCH (u)-[:HAS_ZODIAC]->(z:ZodiacSign)
            OPTIONAL MATCH (u)-[:PREFERS]->(l:Language)
            RETURN properties(u) AS u, p.name AS place, z.name AS sign, l.name AS lang
            """,
            id=user_id,
        )
        if not rows:
            return None
        row = rows[0]
        u = {k: _native(v) for k, v in row["u"].items()}
        return UserProfile(
            user_id=u["id"],
            name=u.get("name"),
            dob=u.get("dob"),
            tob=u.get("tob"),
            birth_place=row["place"],
            preferred_language=row["lang"],
            sun_sign=row["sign"],
            created_at=u.get("created_at"),
            updated_at=u.get("updated_at"),
        )

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        params = {
            "id": profile.user_id,
            "name": profile.name,
            "dob": profile.dob,
            "tob": profile.tob,
            "place": profile.birth_place,
            "sign": profile.sun_sign,
            "lang": profile.preferred_language,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        }
        # Scalar attributes live on the node; shareable values (place, sign, language) are
        # their own nodes so they can be traversed / aggregated across users.
        await self._db.write_many(
            [
                (
                    """
                    MERGE (u:User {id: $id})
                    ON CREATE SET u.created_at = $created_at
                    SET u.name = $name, u.dob = $dob, u.tob = $tob, u.updated_at = $updated_at
                    WITH u
                    OPTIONAL MATCH (u)-[r:BORN_IN|HAS_ZODIAC|PREFERS]->()
                    DELETE r
                    """,
                    params,
                ),
                (
                    """
                    MATCH (u:User {id: $id})
                    FOREACH (_ IN CASE WHEN $place IS NULL THEN [] ELSE [1] END |
                        MERGE (p:Place {name: $place}) MERGE (u)-[:BORN_IN]->(p))
                    FOREACH (_ IN CASE WHEN $sign IS NULL THEN [] ELSE [1] END |
                        MERGE (z:ZodiacSign {name: $sign}) ON CREATE SET z.system = 'tropical'
                        MERGE (u)-[:HAS_ZODIAC]->(z))
                    FOREACH (_ IN CASE WHEN $lang IS NULL THEN [] ELSE [1] END |
                        MERGE (l:Language {name: $lang}) MERGE (u)-[:PREFERS]->(l))
                    """,
                    params,
                ),
            ]
        )
        return profile

    # --- short-term conversation ---------------------------------------------------------

    async def ensure_session(self, user_id: str, session_id: str, now: datetime) -> None:
        rows = await self._db.write(
            """
            MATCH (u:User {id: $uid})
            MERGE (s:Session {id: $sid})
            ON CREATE SET s.user_id = $uid, s.created_at = $now
            WITH u, s
            FOREACH (_ IN CASE WHEN s.user_id = $uid THEN [1] ELSE [] END |
                MERGE (u)-[:HAS_SESSION]->(s))
            RETURN s.user_id AS owner
            """,
            uid=user_id,
            sid=session_id,
            now=now,
        )
        if rows and rows[0]["owner"] != user_id:
            raise SessionOwnershipError(f"session {session_id} belongs to another user")

    async def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        await self._db.write(
            """
            MATCH (s:Session {id: $sid})
            UNWIND $msgs AS msg
            CREATE (m:Message) SET m = msg
            CREATE (s)-[:HAS_MESSAGE]->(m)
            """,
            sid=session_id,
            msgs=[
                {"id": m.id, "role": m.role.value, "content": m.content, "created_at": m.created_at}
                for m in messages
            ],
        )

    async def recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        rows = await self._db.read(
            """
            MATCH (:Session {id: $sid})-[:HAS_MESSAGE]->(m:Message)
            RETURN properties(m) AS m
            ORDER BY m.created_at DESC
            LIMIT $limit
            """,
            sid=session_id,
            limit=limit,
        )
        messages = [
            ChatMessage(
                id=r["m"]["id"],
                role=Role(r["m"]["role"]),
                content=r["m"]["content"],
                created_at=_native(r["m"]["created_at"]),
            )
            for r in rows
        ]
        return list(reversed(messages))

    # --- long-term memory ----------------------------------------------------------------

    async def query_memories(
        self,
        user_id: str,
        *,
        areas: list[LifeArea] | None = None,
        kinds: list[MemoryKind] | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        # Graph traversal: User → memory → ABOUT → LifeArea. The LifeArea hub is what makes
        # topical retrieval cheap and explainable without an embedding index.
        rows = await self._db.read(
            f"""
            MATCH (:User {{id: $uid}})-[:{MEMORY_RELS}]->(m:Memory {{status: 'active'}})
                  -[:ABOUT]->(a:LifeArea)
            WHERE ($areas IS NULL OR a.name IN $areas)
              AND ($kinds IS NULL OR m.kind IN $kinds)
            {_RETURN_MEMORY}
            ORDER BY m.importance DESC, m.updated_at DESC
            LIMIT $limit
            """,
            uid=user_id,
            areas=[a.value for a in areas] if areas else None,
            kinds=[k.value for k in kinds] if kinds else None,
            limit=limit,
        )
        return [_to_memory(r["m"], r["area"]) for r in rows]

    async def find_memory(self, user_id: str, key: str) -> MemoryItem | None:
        rows = await self._db.read(
            f"""
            MATCH (:User {{id: $uid}})-[:{MEMORY_RELS}]->(m:Memory {{key: $key, status: 'active'}})
                  -[:ABOUT]->(a:LifeArea)
            {_RETURN_MEMORY}
            LIMIT 1
            """,
            uid=user_id,
            key=key,
        )
        return _to_memory(rows[0]["m"], rows[0]["area"]) if rows else None

    def _create_statement(self, user_id: str, item: MemoryItem, source_message_id: str | None):
        label, rel = KIND_GRAPH[item.kind]
        query = f"""
            MATCH (u:User {{id: $uid}})
            MATCH (a:LifeArea {{name: $area}})
            CREATE (m:Memory:{label})
            SET m = $props
            CREATE (u)-[:{rel}]->(m)
            CREATE (m)-[:ABOUT]->(a)
            WITH m
            OPTIONAL MATCH (src:Message {{id: $src}})
            FOREACH (_ IN CASE WHEN src IS NULL THEN [] ELSE [1] END |
                CREATE (m)-[:EXTRACTED_FROM]->(src))
        """
        params = {
            "uid": user_id,
            "area": item.life_area.value,
            "props": _memory_props(item),
            "src": source_message_id,
        }
        return query, params

    async def create_memory(
        self, user_id: str, item: MemoryItem, source_message_id: str | None
    ) -> None:
        await self._db.write_many([self._create_statement(user_id, item, source_message_id)])

    async def update_memory(self, item: MemoryItem) -> None:
        props = _memory_props(item)
        props.pop("id")
        props.pop("created_at")
        await self._db.write("MATCH (m:Memory {id: $id}) SET m += $props", id=item.id, props=props)

    async def supersede_memory(
        self, user_id: str, old_id: str, new_item: MemoryItem, source_message_id: str | None
    ) -> None:
        create_q, create_p = self._create_statement(user_id, new_item, source_message_id)
        await self._db.write_many(
            [
                (
                    "MATCH (old:Memory {id: $old}) SET old.status = 'superseded', old.updated_at = $now",
                    {"old": old_id, "now": new_item.updated_at},
                ),
                (create_q, create_p),
                (
                    "MATCH (new:Memory {id: $new}), (old:Memory {id: $old}) CREATE (new)-[:SUPERSEDES]->(old)",
                    {"new": new_item.id, "old": old_id},
                ),
            ]
        )

    async def set_memory_status(self, memory_id: str, status: MemoryStatus, now: datetime) -> None:
        await self._db.write(
            "MATCH (m:Memory {id: $id}) SET m.status = $status, m.updated_at = $now",
            id=memory_id,
            status=status.value,
            now=now,
        )

    async def list_memories(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryItem]:
        rows = await self._db.read(
            f"""
            MATCH (:User {{id: $uid}})-[:{MEMORY_RELS}]->(m:Memory)-[:ABOUT]->(a:LifeArea)
            WHERE $all OR m.status = 'active'
            {_RETURN_MEMORY}
            ORDER BY m.updated_at DESC
            """,
            uid=user_id,
            all=include_inactive,
        )
        return [_to_memory(r["m"], r["area"]) for r in rows]

    async def forget_memories(self, user_id: str) -> int:
        rows = await self._db.write(
            f"""
            MATCH (:User {{id: $uid}})-[:{MEMORY_RELS}]->(m:Memory)
            DETACH DELETE m
            RETURN count(m) AS deleted
            """,
            uid=user_id,
        )
        return rows[0]["deleted"] if rows else 0
