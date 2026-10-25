"""Domain models shared across layers. Storage- and transport-agnostic."""

from datetime import date, datetime, time
from enum import StrEnum
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid4().hex


class LifeArea(StrEnum):
    CAREER = "career"
    RELATIONSHIP = "relationship"
    HEALTH = "health"
    FINANCE = "finance"
    EDUCATION = "education"
    FAMILY = "family"
    SPIRITUALITY = "spirituality"
    GENERAL = "general"


class MemoryKind(StrEnum):
    GOAL = "goal"
    INTEREST = "interest"
    PREFERENCE = "preference"
    LIFE_EVENT = "life_event"
    CONCERN = "concern"
    FACT = "fact"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer version (correction / change of plan)
    RETRACTED = "retracted"  # user said it no longer holds


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class UserProfile(BaseModel):
    user_id: str
    name: str | None = None
    dob: date | None = None
    tob: time | None = None
    birth_place: str | None = None
    preferred_language: str | None = None
    sun_sign: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Fields the astrology experience relies on; surfaced to the LLM so it can ask for them.
    REQUIRED_FOR_READING: ClassVar[tuple[str, ...]] = ("name", "dob", "tob", "birth_place")

    def missing_fields(self) -> list[str]:
        return [f for f in self.REQUIRED_FOR_READING if getattr(self, f) is None]

    def has_any_details(self) -> bool:
        return any(getattr(self, f) is not None for f in ("name", "dob", "tob", "birth_place"))


class MemoryItem(BaseModel):
    """One long-term fact in the Shared Brain (a :Memory node)."""

    id: str = Field(default_factory=new_id)
    kind: MemoryKind
    # Stable, normalised identifier used for de-duplication and corrections, e.g. "career_change".
    key: str
    text: str
    life_area: LifeArea = LifeArea.GENERAL
    target_date: str | None = None  # resolved absolute date: "2027", "2026-04", "2026-04-12"
    timeframe_raw: str | None = None  # what the user actually said: "next year"
    value: str | None = None  # for preferences, e.g. key="response_length", value="short"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    mention_count: int = 1
    created_at: datetime
    updated_at: datetime

    @property
    def label(self) -> str:
        """Context label reported in `context_used`, e.g. "career_goal"."""
        return f"{self.life_area.value}_{self.kind.value}"


class ChatMessage(BaseModel):
    id: str = Field(default_factory=new_id)
    role: Role
    content: str
    created_at: datetime
