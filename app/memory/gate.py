"""Cheap pre-filter deciding whether a message could contain anything worth remembering.

Runs before (and saves the cost of) the LLM extraction call. Most chat turns are questions or
small talk; only declarative first-person statements can carry durable facts about the user.
  "What should I focus on for my career?"   → skip (question)
  "Thanks, that's helpful!"                 → skip (no self-reference)
  "I'm planning to switch jobs next year."  → extract
Tuned for recall: a false positive costs one cheap LLM call, a false negative loses a memory.
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|me|my|mine|myself|we|we're|our)\b", re.I)
# Birth details are often given without any pronoun: "DOB 15-08-1995, 10:30 am, Jaipur".
_PROFILE_CUE = re.compile(
    r"\b(born|birth|dob|tob|birthplace|birth place|place of birth|name is)\b|"
    r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.I,
)
_MIN_CHARS = 8


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def is_self_disclosure(text: str) -> bool:
    """True if any non-question sentence talks about the user or carries profile data."""
    return any(
        not sentence.endswith("?")
        and len(sentence) >= _MIN_CHARS
        and (_FIRST_PERSON.search(sentence) or _PROFILE_CUE.search(sentence))
        for sentence in split_sentences(text)
    )


def answers_question(text: str, last_assistant: str | None) -> bool:
    """A non-question reply to the assistant's question ("Where were you born?" -> "Jaipur")."""
    return bool(last_assistant and last_assistant.rstrip().endswith("?") and not text.rstrip().endswith("?"))
