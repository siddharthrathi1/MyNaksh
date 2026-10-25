"""Memory extraction: decide what in a user message is worth remembering.

Hybrid strategy:
  1. Gate (free)      — skip questions / small talk entirely (most turns).
  2. LLM extraction   — structured JSON, given the user's existing memory keys so it can
                        reuse keys for corrections instead of creating duplicates.
  3. Rule fallback    — if the LLM is down or returns unusable output.
Every candidate is schema-validated individually; one bad item never discards the rest.
"""

import json
import logging
import re
from datetime import datetime
from typing import Protocol

from pydantic import ValidationError

from app.core.errors import LLMError, LLMUnavailableError
from app.core.models import ChatMessage, MemoryItem, Role
from app.llm.base import LLMMessage, LLMProvider, LLMRequest
from app.memory.gate import answers_question, is_self_disclosure
from app.memory.models import ExtractedMemory, ExtractionResult
from app.memory.rules import RuleBasedExtractor
from app.memory.timeframe import resolve_timeframe

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    async def extract(
        self,
        message: str,
        now: datetime,
        existing: list[MemoryItem],
        history: list[ChatMessage],
        memorable: bool | None = None,
    ) -> ExtractionResult: ...


EXTRACTION_PROMPT = """You extract long-term memories about a user from their latest chat message, for a personal astrology assistant.

Remember ONLY durable, user-specific information that will help personalise FUTURE conversations:
- profile facts: name, date of birth, time of birth, birth place, preferred language
- goals and plans (with timeframe), interests, preferences about how to be answered,
  significant life events, recurring concerns

Do NOT remember:
- questions, greetings, thanks, small talk
- hypotheticals ("what if I moved abroad?") or things about other people
- passing moods ("I'm tired today") or anything the assistant said
- anything you are unsure the user actually stated (do not infer or speculate)

Corrections and changes:
- If the message corrects or changes a known memory, use operation "update" with the SAME key.
- If the user says a known memory no longer holds, use operation "remove" with the SAME key.
- If it restates a known memory, use operation "add" with the SAME key.

Resolve relative dates against today's date ({today}): "next year" → "{next_year}". Keep the original phrase in timeframe_raw.

Return a JSON object exactly in this shape:
{{
  "profile_updates": {{"name": str|null, "dob": "YYYY-MM-DD"|null, "tob": "HH:MM"|null, "birth_place": str|null, "preferred_language": str|null}},
  "memories": [
    {{"operation": "add"|"update"|"remove",
      "kind": "goal"|"interest"|"preference"|"life_event"|"concern"|"fact",
      "key": "short_snake_case_identifier",
      "text": "concise statement of the fact, e.g. 'Switch jobs (career change)'",
      "life_area": "career"|"relationship"|"health"|"finance"|"education"|"family"|"spirituality"|"general",
      "target_date": str|null, "timeframe_raw": str|null, "value": str|null,
      "confidence": 0.0-1.0, "importance": 0.0-1.0}}
  ]
}}
Return {{"profile_updates": {{}}, "memories": []}} if nothing is worth remembering."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_json_object(text: str) -> dict:
    """Tolerant JSON parsing: accepts code fences or surrounding prose."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object in LLM output")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def extract(
        self, message: str, now: datetime, existing: list[MemoryItem], history: list[ChatMessage]
    ) -> ExtractionResult:
        known = "\n".join(f"- key={m.key} kind={m.kind.value} text={m.text!r}" for m in existing) or "(none)"
        # Last assistant turn gives context for terse replies like "no, 1996".
        last_assistant = next((m.content for m in reversed(history) if m.role == Role.ASSISTANT), None)
        user_content = f"Known memories:\n{known}\n\n"
        if last_assistant:
            user_content += f"Assistant's previous message (context only, never extract from it):\n{last_assistant[:500]}\n\n"
        user_content += f"User message:\n{message}"

        response = await self._llm.generate(
            LLMRequest(
                system=EXTRACTION_PROMPT.format(today=now.date().isoformat(), next_year=now.year + 1),
                messages=[LLMMessage(role="user", content=user_content)],
                max_tokens=2048,
                temperature=0.0,
                json_mode=True,
                purpose="extraction",
            )
        )
        data = parse_json_object(response.text)

        memories: list[ExtractedMemory] = []
        for raw in data.get("memories") or []:
            try:
                item = ExtractedMemory.model_validate(raw)
            except ValidationError as exc:
                logger.info("dropping invalid extracted memory %r: %s", raw, exc.errors()[:1])
                continue
            # Backfill resolution if the model kept the phrase but didn't resolve it.
            if item.timeframe_raw and not item.target_date:
                item.target_date = resolve_timeframe(item.timeframe_raw, now)
            memories.append(item)

        profile = {k: v for k, v in (data.get("profile_updates") or {}).items() if v not in (None, "")}
        return ExtractionResult(profile_updates=profile, memories=memories, extractor=self.name)


class HybridExtractor:
    def __init__(self, llm_extractor: LLMExtractor | None, rules: RuleBasedExtractor):
        self._llm = llm_extractor
        self._rules = rules

    async def extract(
        self,
        message: str,
        now: datetime,
        existing: list[MemoryItem],
        history: list[ChatMessage],
        memorable: bool | None = None,
    ) -> ExtractionResult:
        """`memorable` is the LLM query analyzer's verdict; None means it didn't judge (rules mode
        or analysis fell back), so the cheap heuristic gate decides instead."""
        if memorable is None:
            last_assistant = next((m.content for m in reversed(history) if m.role == Role.ASSISTANT), None)
            memorable = is_self_disclosure(message) or answers_question(message, last_assistant)
        if not memorable:
            return ExtractionResult(extractor="gate_skipped")

        if self._llm is not None:
            try:
                return await self._llm.extract(message, now, existing, history)
            except (LLMUnavailableError, LLMError, ValueError) as exc:  # ValueError covers JSON errors
                logger.warning("LLM extraction failed (%s); falling back to rules", exc)

        return self._rules.extract(message, now, existing)
