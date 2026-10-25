"""Keyword lexicon mapping text to life areas.

Rule-based on purpose: zero latency, zero cost, deterministic and explainable. It runs on
every turn, so it must be cheap. Known limitation: no synonyms beyond the list and English
only — the upgrade path is an embedding/LLM classifier behind the same `detect_areas` API.
"""

import re

from app.core.models import LifeArea

AREA_KEYWORDS: dict[LifeArea, list[str]] = {
    LifeArea.CAREER: [
        "career", "job", "jobs", "work", "office", "boss", "manager", "promotion", "interview",
        "resume", "hiring", "profession", "business", "startup", "entrepreneur", "entrepreneurship",
        "product management", "switch jobs", "company", "colleague", "appraisal", "layoff",
    ],
    LifeArea.FINANCE: [
        "money", "finance", "finances", "salary", "invest", "investment", "investing", "loan",
        "debt", "savings", "wealth", "property", "stocks", "income", "expenses",
    ],
    LifeArea.RELATIONSHIP: [
        "love", "relationship", "relationships", "partner", "marriage", "married", "wedding",
        "girlfriend", "boyfriend", "wife", "husband", "dating", "breakup", "romance", "soulmate",
        "engaged", "compatibility",
    ],
    LifeArea.HEALTH: [
        "health", "healthy", "sick", "illness", "disease", "stress", "anxiety", "sleep",
        "fitness", "diet", "weight", "exercise", "mental health", "energy",
    ],
    LifeArea.EDUCATION: [
        "exam", "exams", "study", "studies", "college", "university", "degree", "course",
        "mba", "school", "admission", "masters", "phd", "learning",
    ],
    LifeArea.FAMILY: [
        "family", "parents", "mother", "father", "mom", "dad", "kids", "children", "child",
        "son", "daughter", "sibling", "brother", "sister", "baby", "in-laws",
    ],
    LifeArea.SPIRITUALITY: [
        "spiritual", "spirituality", "meditation", "karma", "god", "temple", "puja", "prayer",
        "dharma", "purpose", "mantra",
    ],
}

# One compiled alternation per area; longest phrases first so "product management" wins.
_PATTERNS: dict[LifeArea, re.Pattern] = {
    area: re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    for area, words in AREA_KEYWORDS.items()
}


def detect_areas(text: str) -> list[LifeArea]:
    """Life areas mentioned in `text`, ordered by number of keyword hits (desc)."""
    hits = {area: len(p.findall(text)) for area, p in _PATTERNS.items()}
    return [area for area, n in sorted(hits.items(), key=lambda kv: -kv[1]) if n > 0]


def primary_area(text: str) -> LifeArea:
    areas = detect_areas(text)
    return areas[0] if areas else LifeArea.GENERAL
