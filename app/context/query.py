"""Step 1 of context selection: understand what the user is asking.

Rule-based for latency and determinism (no extra LLM round-trip on the hot path). The output
is a small typed struct, so an LLM/classifier implementation can replace it transparently.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.context.lexicon import detect_areas
from app.core.models import ChatMessage, LifeArea, Role
from app.memory.gate import is_self_disclosure


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


_MEMORY_QUERY = re.compile(
    r"\b(what do you (know|remember)|do you remember|what have i (told|shared|said)|"
    r"remind me what|did i (tell|mention)|what did i (say|tell)|you remember)\b",
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
