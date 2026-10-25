"""End-to-end scenarios from the assignment, through the HTTP API.

Each test runs against the in-memory store and (when reachable) real Neo4j.
"""

import json

from app.llm.base import LLMRequest
from app.llm.fake import FakeLLM
from tests.conftest import chat

RAHUL_INTRO = "My name is Rahul. I was born on 15 August 1995 in Delhi. I'm planning to switch jobs next year."


def _memories(client, user_id, include_inactive=False) -> list[dict]:
    resp = client.get(f"/users/{user_id}/memories", params={"include_inactive": include_inactive})
    assert resp.status_code == 200
    return resp.json()


# 1. New user ------------------------------------------------------------------------------
def test_new_user_is_created_and_asked_for_missing_details(client, fake_llm, user_id):
    body = chat(client, user_id, "s1", "Hi there!")

    assert body["response"]
    assert body["context_used"] == []  # nothing known yet, nothing injected
    assert body["memories_written"] == []  # greeting is not memory-worthy
    system = fake_llm.chat_calls[-1].system
    assert "Missing profile details" in system and "dob" in system
    assert client.get(f"/users/{user_id}").status_code == 200  # user auto-created


# 2. Creating a long-term memory -----------------------------------------------------------
def test_disclosure_creates_profile_and_goal_memory(client, user_id):
    body = chat(client, user_id, "s1", RAHUL_INTRO)

    written = {(c["action"], c["key"]) for c in body["memories_written"]}
    assert ("created", "switch_jobs") in written
    assert {("profile_updated", "name"), ("profile_updated", "dob"), ("profile_updated", "birth_place")} <= written

    profile = client.get(f"/users/{user_id}").json()
    assert profile["name"] == "Rahul"
    assert profile["dob"] == "1995-08-15"
    assert profile["birth_place"] == "Delhi"
    assert profile["sun_sign"] == "Leo"

    [goal] = _memories(client, user_id)
    assert goal["kind"] == "goal" and goal["life_area"] == "career"
    assert goal["target_date"] == "2027"  # "next year" resolved against 2026
    assert goal["timeframe_raw"] == "next year"


# 3. Retrieving a memory -------------------------------------------------------------------
def test_topical_question_uses_relevant_memory(client, fake_llm, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    body = chat(client, user_id, "s1", "What should I focus on for my career?")

    assert {"career_goal", "user_profile", "astrology_profile"} <= set(body["context_used"])
    assert body["memories_written"] == []  # a question is not stored
    system = fake_llm.chat_calls[-1].system
    assert "Switch jobs" in system and "2027" in system
    assert "Sun sign: Leo" in system


# 4. Follow-up question --------------------------------------------------------------------
def test_follow_up_is_resolved_from_short_term_context(client, fake_llm, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    previous = chat(client, user_id, "s1", "What should I focus on for my career?")["response"]
    body = chat(client, user_id, "s1", "Why do you say that?")

    assert body["query_analysis"]["is_follow_up"] is True
    assert body["query_analysis"]["life_areas"] == ["career"]  # topic inherited from history
    assert "conversation_history" in body["context_used"]
    messages = fake_llm.chat_calls[-1].messages
    assert any(m.role == "assistant" and m.content == previous for m in messages)
    assert messages[-1].content == "Why do you say that?"


# 5. New session using previous information ------------------------------------------------
def test_new_session_recalls_long_term_memory_without_old_history(client, fake_llm, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    body = chat(client, user_id, "s2", "What do you remember about my career goals?")

    assert body["query_analysis"]["is_memory_query"] is True
    assert "career_goal" in body["context_used"]
    assert "conversation_history" not in body["context_used"]  # short-term is per session
    call = fake_llm.chat_calls[-1]
    assert len(call.messages) == 1  # only the current message
    assert "Switch jobs" in call.system


# 6. Irrelevant memory ---------------------------------------------------------------------
def test_unrelated_memory_is_not_injected(client, fake_llm, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    body = chat(client, user_id, "s2", "How will my love life be this year?")

    assert body["query_analysis"]["life_areas"] == ["relationship"]
    assert "career_goal" not in body["context_used"]
    assert "Switch jobs" not in fake_llm.chat_calls[-1].system


def test_small_talk_is_not_stored(client, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    before = _memories(client, user_id)
    body = chat(client, user_id, "s1", "Thanks, that's helpful!")

    assert body["memories_written"] == []
    assert _memories(client, user_id) == before


# 7. User correcting existing information --------------------------------------------------
def test_profile_correction_overwrites_birth_place(client, user_id):
    chat(client, user_id, "s1", "I was born in Delhi.")
    body = chat(client, user_id, "s1", "Actually I was born in Mumbai, not Delhi.")

    assert any(c["key"] == "birth_place" and c["text"] == "Mumbai" for c in body["memories_written"])
    assert client.get(f"/users/{user_id}").json()["birth_place"] == "Mumbai"


def test_goal_change_supersedes_old_memory_with_history(make_client, user_id):
    """LLM extraction path: the extractor reuses the existing key with operation=update."""

    def responder(req: LLMRequest) -> str:
        if req.purpose != "extraction":
            return "ok"
        msg = req.messages[-1].content
        if "start my own business" in msg:
            mem = {"operation": "update", "kind": "goal", "key": "career_change",
                   "text": "Start own business instead of switching jobs", "life_area": "career",
                   "target_date": "2027", "timeframe_raw": "next year", "confidence": 0.9, "importance": 0.9}
        else:
            mem = {"operation": "add", "kind": "goal", "key": "career_change", "text": "Switch jobs",
                   "life_area": "career", "target_date": "2027", "timeframe_raw": "next year",
                   "confidence": 0.9, "importance": 0.8}
        return json.dumps({"profile_updates": {}, "memories": [mem]})

    client, _ = make_client(llm=FakeLLM(responder), extraction_mode="llm")
    chat(client, user_id, "s1", "I'm planning to switch jobs next year.")
    body = chat(client, user_id, "s1", "Actually I've decided to start my own business instead.")

    assert [(c["action"], c["key"]) for c in body["memories_written"]] == [("updated", "career_change")]
    active = _memories(client, user_id)
    assert [m["text"] for m in active] == ["Start own business instead of switching jobs"]
    assert active[0]["confidence"] == 1.0  # explicit corrections are trusted most
    statuses = sorted(m["status"] for m in _memories(client, user_id, include_inactive=True))
    assert statuses == ["active", "superseded"]  # old version kept for audit


def test_goal_retraction(client, user_id):
    chat(client, user_id, "s1", "I'm planning to switch jobs next year.")
    body = chat(client, user_id, "s1", "I'm not planning to switch jobs anymore.")

    assert [(c["action"], c["key"]) for c in body["memories_written"]] == [("retracted", "switch_jobs")]
    assert _memories(client, user_id) == []
    assert _memories(client, user_id, include_inactive=True)[0]["status"] == "retracted"


def test_restating_a_fact_reinforces_instead_of_duplicating(client, user_id):
    chat(client, user_id, "s1", "I'm planning to switch jobs next year.")
    body = chat(client, user_id, "s2", "Just a reminder, I'm planning to switch jobs next year.")

    assert [(c["action"], c["key"]) for c in body["memories_written"]] == [("reinforced", "switch_jobs")]
    [goal] = _memories(client, user_id)
    assert goal["mention_count"] == 2


# 8. Missing user information --------------------------------------------------------------
def test_astrology_question_with_missing_birth_details(client, fake_llm, user_id):
    assert client.post("/users", json={"user_id": user_id, "name": "Asha"}).status_code == 201
    body = chat(client, user_id, "s1", "What does my horoscope say about this month?")

    assert body["query_analysis"]["intent"] == "astrology"
    assert "user_profile" in body["context_used"]
    assert "astrology_profile" not in body["context_used"]  # no DOB → no sign
    system = fake_llm.chat_calls[-1].system
    assert "Missing profile details" in system and "dob" in system and "birth_place" in system


def test_profile_api_and_preferred_language(client, fake_llm, user_id):
    resp = client.post(
        "/users",
        json={"user_id": user_id, "name": "rahul", "dob": "1995-08-15", "birth_place": "delhi", "preferred_language": "hi"},
    )
    assert resp.status_code == 201
    profile = resp.json()
    assert (profile["name"], profile["birth_place"], profile["preferred_language"], profile["sun_sign"]) == (
        "Rahul", "Delhi", "Hindi", "Leo"
    )
    assert profile["missing_fields"] == ["tob"]

    chat(client, user_id, "s1", "Any advice for me?")
    assert "Reply in Hindi." in fake_llm.chat_calls[-1].system


def test_forget_memories(client, user_id):
    chat(client, user_id, "s1", RAHUL_INTRO)
    assert client.delete(f"/users/{user_id}/memories").json() == {"deleted": 1}
    assert _memories(client, user_id, include_inactive=True) == []
