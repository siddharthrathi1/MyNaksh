"""Error handling and graceful degradation (in-memory backend only — failures are simulated)."""

import asyncio

import pytest

from app.core.errors import GraphUnavailableError, LLMUnavailableError
from app.llm.base import LLMMessage, LLMRequest
from app.llm.fake import FakeLLM
from app.llm.resilient import ResilientLLM
from app.chat.orchestrator import FALLBACK_REPLY
from tests.conftest import chat


@pytest.fixture
def backend() -> str:  # overrides the parametrised fixture: these tests don't need Neo4j
    return "memory"


class DownRepository:
    """Every storage call fails as if Neo4j were unreachable."""

    async def ping(self) -> bool:
        return False

    def __getattr__(self, name):
        async def unavailable(*args, **kwargs):
            raise GraphUnavailableError("connection refused")

        return unavailable


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "u1", "session_id": "s1", "message": ""},
        {"user_id": "u1", "session_id": "s1", "message": "   "},
        {"user_id": "u1", "session_id": "s1", "message": "x" * 2001},
        {"user_id": "bad id with spaces", "session_id": "s1", "message": "hi"},
        {"user_id": "u1", "message": "hi"},
    ],
)
def test_invalid_chat_input_is_rejected(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_future_date_of_birth_is_rejected(client):
    resp = client.post("/users", json={"user_id": "u1", "dob": "2999-01-01"})
    assert resp.status_code == 422


def test_unknown_user_returns_404(client):
    resp = client.get("/users/nobody")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_llm_outage_serves_fallback_and_still_remembers(make_client, user_id):
    client, _ = make_client(llm=FakeLLM(fail_always=True))
    body = chat(client, user_id, "s1", "I'm planning to switch jobs next year.")

    assert body["response"] == FALLBACK_REPLY
    assert body["degraded"] is True and body["degraded_reasons"] == ["llm_unavailable"]
    # Rule-based extraction does not depend on the LLM, so the fact is not lost.
    assert [c["key"] for c in body["memories_written"]] == ["switch_jobs"]


def test_transient_llm_failure_is_retried(make_client, user_id):
    llm = FakeLLM(fail_times=1)
    client, _ = make_client(llm=llm)
    body = chat(client, user_id, "s1", "Hello")

    assert body["degraded"] is False
    assert len(llm.chat_calls) == 2  # first attempt failed, retry succeeded


def test_fallback_provider_takes_over():
    primary, secondary = FakeLLM(fail_always=True), FakeLLM(lambda r: "from secondary")
    llm = ResilientLLM([primary, secondary], max_attempts=2, base_delay_s=0)
    req = LLMRequest(system="s", messages=[LLMMessage("user", "hi")])

    assert asyncio.run(llm.generate(req)).text == "from secondary"
    assert len(primary.calls) == 2

    all_down = ResilientLLM([FakeLLM(fail_always=True)], max_attempts=1, base_delay_s=0)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(all_down.generate(req))


def test_graph_outage_degrades_chat_but_never_fails_it(make_client, user_id):
    client, _ = make_client(repo=DownRepository())
    body = chat(client, user_id, "s1", "I'm planning to switch jobs next year.")

    assert body["response"]
    assert body["degraded"] is True and "memory_unavailable" in body["degraded_reasons"]
    assert body["memory_update"] == "skipped"
    assert client.get("/health").json()["status"] == "degraded"
    # Endpoints that cannot degrade report a clean 503.
    assert client.get(f"/users/{user_id}").status_code == 503


def test_session_cannot_be_hijacked_by_another_user(client):
    post = lambda uid: client.post("/chat", json={"user_id": uid, "session_id": "shared", "message": "Hello"})
    assert post("alice").status_code == 200
    assert post("bob").status_code == 409


def test_bad_llm_extraction_output_falls_back_to_rules(make_client, user_id):
    def responder(req: LLMRequest) -> str:
        return "Sure! Here are the memories: none, really." if req.json_mode else "ok"

    client, _ = make_client(llm=FakeLLM(responder), extraction_mode="llm")
    body = chat(client, user_id, "s1", "I'm planning to switch jobs next year.")
    assert [c["key"] for c in body["memories_written"]] == ["switch_jobs"]


def test_invalid_items_in_llm_extraction_are_dropped_individually(make_client, user_id):
    def responder(req: LLMRequest) -> str:
        if not req.json_mode:
            return "ok"
        return (
            '{"profile_updates": {"name": "Rahul", "dob": "not-a-date"}, "memories": ['
            '{"kind": "goal", "key": "career_change", "text": "Switch jobs", "life_area": "work", "confidence": 0.9},'
            '{"kind": "spaceship", "key": "x", "text": "invalid kind"},'
            '{"kind": "interest", "key": "chess", "text": "Plays chess", "confidence": 0.3}]}'
        )

    client, _ = make_client(llm=FakeLLM(responder), extraction_mode="llm")
    body = chat(client, user_id, "s1", "My name is Rahul and I'm planning a career change.")

    written = {(c["action"], c["key"]) for c in body["memories_written"]}
    # valid goal kept (unknown area → general); invalid kind dropped; low confidence gated; bad dob dropped
    assert written == {("profile_updated", "name"), ("created", "career_change")}


def test_background_memory_mode(make_client, user_id):
    client, _ = make_client(memory_write_mode="background")
    body = chat(client, user_id, "s1", "I'm planning to switch jobs next year.")

    assert body["memory_update"] == "scheduled" and body["memories_written"] == []
    # TestClient runs background tasks before returning, so the write is visible now.
    assert len(client.get(f"/users/{user_id}/memories").json()) == 1
