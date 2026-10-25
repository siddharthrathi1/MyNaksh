"""Step 1 of context selection: understand what the user is asking.

Two implementations behind one async `analyze`:
- LLMQueryAnalyzer   — one small JSON call that classifies the turn AND decides whether it
                       carries anything worth remembering (replaces keyword/regex heuristics).
- QueryAnalyzer      — rule-based; deterministic, zero-latency fallback when the LLM is down,
                       and the default in offline/test mode.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import ValidationError

from app.context.lexicon import detect_areas
from app.core.errors import LLMError, LLMUnavailableError
from app.core.models import ChatMessage, LifeArea, Role
from app.llm.base import LLMMessage, LLMProvider, LLMRequest
from app.memory.gate import is_self_disclosure

logger = logging.getLogger(__name__)


class Intent(StrEnum):
    MEMORY_RECALL = "memory_recall"  # "what do you remember about me?"
    FOLLOW_UP = "follow_up"  # "why do you say that?"
    ASTROLOGY = "astrology"  # "what does my sign say?"
    ADVICE = "advice"  # topical question
    DISCLOSURE = "disclosure"  # user is telling us about themselves
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    intent: Intent
    life_areas: list[LifeArea] = field(default_factory=list)
    is_follow_up: bool = False
    is_memory_query: bool = False
    # Where the life areas came from: "message" or "history" (follow-ups inherit topic).
    areas_source: str = "message"
    # Does the message carry durable info about the user? None = not judged (rules: the
    # extractor applies its own cheap gate instead).
    memorable: bool | None = None
    analyzer: str = "rules"


_MEMORY_QUERY = re.compile(
    r"\b(what do you (know|remember)|do you remember|what have i (told|shared|said)|"
    r"remind me what|did i (tell|mention)|what did i (say|tell)|you remember|"
    r"(what(?:'s| is| are| was| were)|tell me|do you know) my (name|birth|dob|date of birth|time of birth|"
    r"details|profile|job|profession|occupation|goals?|plans?))\b",
    re.I,
)
_FOLLOW_UP_START = re.compile(
    r"^\s*(why|how come|what do you mean|explain|elaborate|tell me more|go on|really|"
    r"and\b|but\b|so\b|can you (expand|explain|elaborate)|what about|more)",
    re.I,
)
_ANAPHORA = re.compile(r"\b(that|this|it|those|these|them|you said)\b", re.I)
_ASTRO = re.compile(
    r"\b(sign|zodiac|horoscope|kundli|kundali|chart|planet|planets|rashi|nakshatra|"
    r"astrology|stars|transit|retrograde|saturn|jupiter|mercury|venus|mars|moon)\b",
    re.I,
)

_FOLLOW_UP_MAX_WORDS = 8
_HISTORY_LOOKBACK = 3  # user turns to look back through for the topic of a follow-up


class QueryAnalyzer:
    def analyze(self, message: str, history: list[ChatMessage]) -> QueryAnalysis:
        areas = detect_areas(message)
        is_memory_query = bool(_MEMORY_QUERY.search(message))
        is_follow_up = self._is_follow_up(message, areas, history)

        areas_source = "message"
        if is_follow_up and not areas:
            areas = self._areas_from_history(history)
            areas_source = "history"

        if is_memory_query:
            intent = Intent.MEMORY_RECALL
        elif is_follow_up:
            intent = Intent.FOLLOW_UP
        elif _ASTRO.search(message):
            intent = Intent.ASTROLOGY
        elif is_self_disclosure(message):
            intent = Intent.DISCLOSURE
        elif areas:
            intent = Intent.ADVICE
        else:
            intent = Intent.GENERAL

        return QueryAnalysis(
            intent=intent,
            life_areas=areas,
            is_follow_up=is_follow_up,
            is_memory_query=is_memory_query,
            areas_source=areas_source,
        )

    @staticmethod
    def _is_follow_up(message: str, areas: list[LifeArea], history: list[ChatMessage]) -> bool:
        if not history or areas:
            return False  # nothing to follow up on, or the message names its own topic
        short = len(message.split()) <= _FOLLOW_UP_MAX_WORDS
        return bool(_FOLLOW_UP_START.search(message)) or (short and bool(_ANAPHORA.search(message)))

    @staticmethod
    def _areas_from_history(history: list[ChatMessage]) -> list[LifeArea]:
        user_turns = [m for m in reversed(history) if m.role == Role.USER][:_HISTORY_LOOKBACK]
        for turn in user_turns:
            if areas := detect_areas(turn.content):
                return areas
        return []


ANALYSIS_PROMPT = """You analyse one chat turn for a personal astrology assistant. Read the user's latest message (and the recent conversation for context) and classify it.

Fields:
- intent: one of
    "memory_recall"  - the user asks what the assistant knows/remembers about them, or asks for their own stored details
                       (e.g. "what are my birth details?", "do you know my job?")
    "follow_up"      - continues the previous exchange without restating the topic ("why?", "and next year?")
    "astrology"      - asks about their chart, sign, planets, transits, dashas, etc.
    "advice"         - asks for guidance on a life topic
    "disclosure"     - mainly tells the assistant something about themselves
    "general"        - greetings, thanks, small talk, anything else
- life_areas: topics the turn is about, most relevant first, from
    ["career","relationship","health","finance","education","family","spirituality"].
    For a follow-up, use the topic of the conversation it continues. Use [] if none.
- is_follow_up: true if the message only makes sense with the previous turns.
- is_memory_query: true if the user asks what the assistant knows or remembers about them.
- memorable: true if the message states ANY durable information about the user that is worth remembering for future
    conversations: name, date/time/place of birth (in any format, even bare like "15/8/95, 10am, Pune"), profession,
    goals and plans, life events, relationships, interests, preferences about how to be answered, recurring concerns,
    or a correction/retraction of something they said earlier. A short answer to the assistant's question
    (e.g. "Pune" after "Where were you born?") counts. Questions, greetings, thanks and passing moods do not.
    When unsure, answer true.

Return only a JSON object:
{"intent": str, "life_areas": [str], "is_follow_up": bool, "is_memory_query": bool, "memorable": bool}"""

_HISTORY_FOR_ANALYSIS = 4  # messages
_HISTORY_CHARS = 300


class LLMQueryAnalyzer:
    """One small, cheap classification call on the hot path (needed before context selection)."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def analyze(self, message: str, history: list[ChatMessage]) -> QueryAnalysis:
        from app.memory.extractor import parse_json_object  # avoid import cycle at module load

        recent = history[-_HISTORY_FOR_ANALYSIS:]
        convo = "\n".join(f"{m.role.value}: {m.content[:_HISTORY_CHARS]}" for m in recent) or "(new conversation)"
        response = await self._llm.generate(
            LLMRequest(
                system=ANALYSIS_PROMPT,
                messages=[LLMMessage(role="user", content=f"Recent conversation:\n{convo}\n\nLatest user message:\n{message}")],
                max_tokens=256,
                temperature=0.0,
                json_mode=True,
                purpose="analysis",
            )
        )
        data = parse_json_object(response.text)

        try:
            intent = Intent(data.get("intent", "general"))
        except ValueError:
            intent = Intent.GENERAL
        areas = []
        for raw in data.get("life_areas") or []:
            try:
                area = LifeArea(str(raw).lower())
            except ValueError:
                continue
            if area != LifeArea.GENERAL and area not in areas:
                areas.append(area)
        is_follow_up = bool(data.get("is_follow_up")) and bool(history)
        is_memory_query = bool(data.get("is_memory_query")) or intent == Intent.MEMORY_RECALL
        return QueryAnalysis(
            intent=intent,
            life_areas=areas,
            is_follow_up=is_follow_up,
            is_memory_query=is_memory_query,
            areas_source="history" if is_follow_up else "message",
            memorable=bool(data.get("memorable", True)),
            analyzer="llm",
        )


class HybridQueryAnalyzer:
    """LLM analysis when available; rules when it is disabled, down, or returns unusable output."""

    def __init__(self, llm_analyzer: LLMQueryAnalyzer | None, rules: QueryAnalyzer):
        self._llm = llm_analyzer
        self._rules = rules

    async def analyze(self, message: str, history: list[ChatMessage]) -> QueryAnalysis:
        if self._llm is not None:
            try:
                return await self._llm.analyze(message, history)
            except (LLMUnavailableError, LLMError, ValueError, ValidationError) as exc:
                logger.warning("LLM query analysis failed (%s); falling back to rules", exc)
        return self._rules.analyze(message, history)
