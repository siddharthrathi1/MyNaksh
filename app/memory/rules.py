"""Rule-based extractor.

Roles:
1. Fallback when the LLM extractor is unavailable or returns garbage.
2. Offline / test mode (deterministic, no API key).
High precision, limited recall by design: it only recognises explicit first-person patterns.
"""

import calendar
import re
from datetime import date, datetime

from app.context.lexicon import primary_area
from app.core.models import LifeArea, MemoryItem, MemoryKind
from app.memory.gate import split_sentences
from app.memory.models import ExtractedMemory, ExtractionResult, Operation, normalise_key
from app.memory.timeframe import TIMEFRAME_PATTERN, find_timeframe, resolve_timeframe

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_LANGS = r"(hindi|english|tamil|telugu|bengali|marathi|gujarati|kannada|malayalam|punjabi)"

# Case-sensitive on the name itself so "My name is Rahul and…" doesn't capture "Rahul and".
_NAME = re.compile(
    r"(?i:\bmy name is)\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)|(?i:\bcall me)\s+([A-Z][A-Za-z'\-]+)"
)
_DOB_DMY = re.compile(r"\bborn on (?:the )?(\d{1,2})(?:st|nd|rd|th)?(?: of)? ([A-Za-z]+),? (\d{4})", re.I)
_DOB_MDY = re.compile(r"\bborn on ([A-Za-z]+) (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})", re.I)
_DOB_ISO = re.compile(r"\bborn on (\d{4}-\d{2}-\d{2})", re.I)
# Place must be Capitalised so "born in 1995" or "born in the morning" don't match.
_PLACE = re.compile(r"(?i:\bborn\b)(?: on [^.]*?)? (?i:in) ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)")
_TOB = re.compile(r"\bat (\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_LANG = re.compile(rf"\b(?:prefer|reply in|respond in|talk to me in|answer in|speak (?:to me )?in|write in)\s+(?:\w+\s+)?{_LANGS}\b", re.I)

_GOAL_STRONG = re.compile(
    r"\b(?:i'?m|i am|we're|we are)\s+(?:planning|preparing|aiming|working|saving)\s+(?:to|for|towards?|on)\s+(.+)", re.I
)
_GOAL_WEAK = re.compile(
    r"\b(?:i'?m|i am)\s+(?:hoping|trying|looking|thinking of|considering)\s+(?:to\s+)?(.+)|"
    r"\bi (?:want|plan|hope|intend|aim|would like|wish) to (.+)|\bmy goal is to (.+)",
    re.I,
)
_RETRACT = re.compile(
    r"\b(?:i'?m|i am)\s+(?:no longer|not)\s+(?:planning|going|preparing|trying|looking)\s+(?:to|for)\s+(.+?)(?:\s+anymore)?$|"
    r"\b(?:i(?:'ve| have)?\s+)?(?:dropped|gave up on|given up on|cancelled|changed my mind about)\s+(.+)",
    re.I,
)
_INTEREST = re.compile(
    r"\bi(?:'m| am)\s+(?:really\s+|very\s+)?(?:interested in|passionate about|into|curious about|fascinated by)\s+(.+)|"
    r"\bi (?:really )?(?:love|enjoy)\s+(.+)",
    re.I,
)
_CONCERN = re.compile(r"\bi(?:'m| am)\s+(?:really\s+|very\s+)?(?:worried|anxious|stressed|concerned|nervous)\s+about\s+(.+)", re.I)
_LIFE_EVENT = re.compile(
    r"\bi (?:just |recently )?(got married|got engaged|had a baby|lost my job|got promoted|graduated|"
    r"moved to [A-Z][a-zA-Z]+|started a new job|got divorced|bought a house)",
    re.I,
)
_LENGTH_PREF = re.compile(r"\bkeep (?:it|your answers|answers|responses|replies) (short|brief|concise|detailed)\b", re.I)

_LEADING_FILLER = re.compile(r"^(?:a|an|the|my|to|for|on)\s+", re.I)
_PRONOUN_OBJECT = re.compile(r"^(?:you|it|this|that|them|him|her)\b", re.I)


def _clean_phrase(text: str) -> str:
    text = TIMEFRAME_PATTERN.sub("", text)
    text = re.split(r"[.,;!?]| because | but | so that ", text)[0]
    text = " ".join(text.split())
    while (stripped := _LEADING_FILLER.sub("", text)) != text:
        text = stripped
    return text.strip()


def _parse_dob(text: str) -> date | None:
    try:
        if m := _DOB_ISO.search(text):
            return date.fromisoformat(m.group(1))
        if m := _DOB_DMY.search(text):
            day, month, year = int(m.group(1)), _MONTHS.get(m.group(2).lower()), int(m.group(3))
            return date(year, month, day) if month else None
        if m := _DOB_MDY.search(text):
            month, day, year = _MONTHS.get(m.group(1).lower()), int(m.group(2)), int(m.group(3))
            return date(year, month, day) if month else None
    except ValueError:
        return None
    return None


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z]{3,}", a.lower())), set(re.findall(r"[a-z]{3,}", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


class RuleBasedExtractor:
    name = "rules"

    def extract(self, message: str, now: datetime, existing: list[MemoryItem]) -> ExtractionResult:
        profile = self._profile(message)
        memories: list[ExtractedMemory] = []

        for sentence in split_sentences(message):
            if sentence.endswith("?"):
                continue  # questions are never facts about the user
            if retraction := self._retraction(sentence, existing):
                memories.append(retraction)
                continue
            for extractor in (self._goal, self._interest, self._concern, self._life_event, self._preference):
                if item := extractor(sentence, now):
                    memories.append(item)
                    break

        return ExtractionResult(profile_updates=profile, memories=memories, extractor=self.name)

    # --- profile -------------------------------------------------------------------------

    @staticmethod
    def _profile(text: str) -> dict:
        updates: dict = {}
        if m := _NAME.search(text):
            updates["name"] = m.group(1) or m.group(2)
        if dob := _parse_dob(text):
            updates["dob"] = dob.isoformat()
        if m := _PLACE.search(text):
            updates["birth_place"] = m.group(1)
        if "born" in text.lower() and (m := _TOB.search(text)):
            hour, minute, meridiem = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3).lower()
            updates["tob"] = f"{hour + (12 if meridiem == 'pm' else 0):02d}:{minute:02d}"
        if m := _LANG.search(text):
            updates["preferred_language"] = m.group(1)
        return updates

    # --- memories ------------------------------------------------------------------------

    @staticmethod
    def _goal(sentence: str, now: datetime) -> ExtractedMemory | None:
        strong = _GOAL_STRONG.search(sentence)
        match = strong or _GOAL_WEAK.search(sentence)
        if not match:
            return None
        raw = next(g for g in match.groups() if g)
        title = _clean_phrase(raw)
        if len(title) < 3:
            return None
        timeframe = find_timeframe(raw)
        return ExtractedMemory(
            kind=MemoryKind.GOAL,
            key=normalise_key(title),
            text=title[0].upper() + title[1:],
            life_area=primary_area(raw),
            timeframe_raw=timeframe,
            target_date=resolve_timeframe(timeframe, now) if timeframe else None,
            confidence=0.85 if strong else 0.7,
            importance=0.8 if timeframe else 0.7,
        )

    @staticmethod
    def _interest(sentence: str, now: datetime) -> ExtractedMemory | None:
        match = _INTEREST.search(sentence)
        if not match:
            return None
        topic = _clean_phrase(next(g for g in match.groups() if g))
        if len(topic) < 3 or _PRONOUN_OBJECT.match(topic):
            return None  # "I love it", "I love you" — not an interest
        return ExtractedMemory(
            kind=MemoryKind.INTEREST,
            key=normalise_key(topic),
            text=f"Interested in {topic}",
            life_area=primary_area(topic),
            confidence=0.75,
            importance=0.5,
        )

    @staticmethod
    def _concern(sentence: str, now: datetime) -> ExtractedMemory | None:
        match = _CONCERN.search(sentence)
        if not match:
            return None
        topic = _clean_phrase(match.group(1))
        if len(topic) < 3:
            return None
        return ExtractedMemory(
            kind=MemoryKind.CONCERN,
            key=f"concern_{normalise_key(topic)}",
            text=f"Worried about {topic}",
            life_area=primary_area(topic),
            confidence=0.7,
            importance=0.6,
        )

    @staticmethod
    def _life_event(sentence: str, now: datetime) -> ExtractedMemory | None:
        match = _LIFE_EVENT.search(sentence)
        if not match:
            return None
        event = match.group(1)
        return ExtractedMemory(
            kind=MemoryKind.LIFE_EVENT,
            key=normalise_key(event),
            text=f"{event[0].upper()}{event[1:]} (around {now.date().isoformat()})",
            life_area=primary_area(event),
            confidence=0.85,
            importance=0.8,
        )

    @staticmethod
    def _preference(sentence: str, now: datetime) -> ExtractedMemory | None:
        match = _LENGTH_PREF.search(sentence)
        if not match:
            return None
        value = "detailed" if match.group(1).lower() == "detailed" else "short"
        return ExtractedMemory(
            kind=MemoryKind.PREFERENCE,
            key="response_length",
            text=f"Prefers {value} answers",
            life_area=LifeArea.GENERAL,
            value=value,
            confidence=0.9,
            importance=0.6,
        )

    @staticmethod
    def _retraction(sentence: str, existing: list[MemoryItem]) -> ExtractedMemory | None:
        match = _RETRACT.search(sentence)
        if not match:
            return None
        phrase = next(g for g in match.groups() if g)
        # Match against what we already know; we can only retract something that exists.
        best = max(existing, key=lambda m: _token_overlap(phrase, m.text), default=None)
        if not best or _token_overlap(phrase, best.text) < 0.25:
            return None
        return ExtractedMemory(
            operation=Operation.REMOVE,
            kind=best.kind,
            key=best.key,
            text=best.text,
            life_area=best.life_area,
            confidence=0.8,
        )
