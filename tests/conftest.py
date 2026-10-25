"""Shared fixtures.

Scenario tests run against BOTH storage adapters:
- "memory": always (no infrastructure)
- "neo4j":  when a Neo4j instance is reachable at NEO4J_URI (e.g. `docker compose run --rm tests`),
            otherwise skipped. Same assertions → the adapters honour the same contract.
The LLM is always the deterministic FakeLLM, so no API keys are needed.
"""

import asyncio
import os
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.brain.client import GraphClient
from app.config import Settings
from app.container import Container
from app.llm.base import LLMProvider
from app.llm.fake import FakeLLM
from app.llm.resilient import ResilientLLM
from app.main import create_app

# "Today" for every test: makes "next year" → 2027, matching the assignment example.
START = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


class TickingClock:
    """Deterministic clock that advances 1s per call, so message ordering is well-defined."""

    def __init__(self, start: datetime = START):
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def make_settings(**overrides) -> Settings:
    base = {
        "llm_provider": "fake",
        "brain_backend": "memory",
        "memory_write_mode": "sync",
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    }
    return Settings(_env_file=None, **{**base, **overrides})


_neo4j_ok: bool | None = None


def neo4j_available() -> bool:
    global _neo4j_ok
    if _neo4j_ok is None:

        async def probe() -> bool:
            client = GraphClient(make_settings(neo4j_timeout_s=2.0))
            try:
                return await client.ping()
            finally:
                await client.close()

        _neo4j_ok = asyncio.run(probe())
    return _neo4j_ok


@pytest.fixture(params=["memory", pytest.param("neo4j", marks=pytest.mark.integration)])
def backend(request) -> str:
    if request.param == "neo4j" and not neo4j_available():
        pytest.skip("Neo4j not reachable")
    return request.param


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


ClientFactory = Callable[..., tuple[TestClient, Container]]


@pytest.fixture
def make_client(backend: str) -> Iterator[ClientFactory]:
    """Builds an app wired with test doubles. Usage: client, container = make_client(llm=..., **settings)."""
    clients: list[TestClient] = []

    def factory(llm: LLMProvider | None = None, repo=None, **overrides) -> tuple[TestClient, Container]:
        settings = make_settings(brain_backend=backend, **overrides)
        llm = ResilientLLM([llm or FakeLLM()], max_attempts=2, base_delay_s=0)
        container = Container.build(settings, repo=repo, llm=llm, clock=TickingClock())
        client = TestClient(create_app(container))
        client.__enter__()  # runs lifespan (schema init for Neo4j)
        clients.append(client)
        return client, container

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client, fake_llm) -> TestClient:
    return make_client(llm=fake_llm)[0]


@pytest.fixture
def user_id() -> str:
    # Unique per test → tests are isolated even against a shared Neo4j database.
    return f"test-user-{uuid.uuid4().hex[:8]}"


def chat(client: TestClient, user_id: str, session_id: str, message: str) -> dict:
    """POST /chat. Session ids are namespaced by user so tests never collide in a shared DB."""
    payload = {"user_id": user_id, "session_id": f"{user_id}-{session_id}", "message": message}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()
