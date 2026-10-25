"""Neo4j schema: constraints, indexes, seed data and the memory-kind → graph mapping.

Neo4j has no DDL for relationship types; the allowed relationships are defined here and
enforced by the repository being the only writer. See docs/DESIGN.md §2.
"""

from app.core.models import LifeArea, MemoryKind

# Every long-term memory node carries the generic :Memory label (uniform retrieval, one id
# constraint) plus a kind-specific label for readability in the graph browser.
# The relationship from :User encodes the semantic type, as in the assignment examples.
KIND_GRAPH: dict[MemoryKind, tuple[str, str]] = {
    MemoryKind.GOAL: ("Goal", "HAS_GOAL"),
    MemoryKind.INTEREST: ("Interest", "INTERESTED_IN"),
    MemoryKind.PREFERENCE: ("Preference", "HAS_PREFERENCE"),
    MemoryKind.LIFE_EVENT: ("LifeEvent", "HAS_MEMORY"),
    MemoryKind.CONCERN: ("Concern", "HAS_MEMORY"),
    MemoryKind.FACT: ("Fact", "HAS_MEMORY"),
}

MEMORY_RELS = "|".join(sorted({rel for _, rel in KIND_GRAPH.values()}))

LIFE_AREAS = [area.value for area in LifeArea]

CONSTRAINTS = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT message_id IF NOT EXISTS FOR (n:Message) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT life_area_name IF NOT EXISTS FOR (n:LifeArea) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT place_name IF NOT EXISTS FOR (n:Place) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT zodiac_name IF NOT EXISTS FOR (n:ZodiacSign) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT language_name IF NOT EXISTS FOR (n:Language) REQUIRE n.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX message_created IF NOT EXISTS FOR (n:Message) ON (n.created_at)",
    "CREATE INDEX memory_key IF NOT EXISTS FOR (n:Memory) ON (n.key)",
    "CREATE INDEX memory_status IF NOT EXISTS FOR (n:Memory) ON (n.status)",
]

SEED_LIFE_AREAS = "UNWIND $names AS name MERGE (:LifeArea {name: name})"
