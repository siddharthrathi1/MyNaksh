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
_MIN_CHARS = 8


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def is_self_disclosure(text: str) -> bool:
    """True if any non-question sentence talks about the user."""
    return any(
        not sentence.endswith("?") and len(sentence) >= _MIN_CHARS and _FIRST_PERSON.search(sentence)
        for sentence in split_sentences(text)
    )
