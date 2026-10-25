"""Contracts between extraction (what *might* be worth remembering) and the updater
(what actually gets written)."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.core.models import LifeArea, MemoryKind


class Operation(StrEnum):
    ADD = "add"  # new fact (or a restatement of a known one → reinforced)
    UPDATE = "update"  # correction / change to a known fact (same key) → supersede
    REMOVE = "remove"  # user says it no longer holds → retract


def normalise_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return key[:60]


class ExtractedMemory(BaseModel):
    operation: Operation = Operation.ADD
    kind: MemoryKind
    key: str
    text: str = Field(min_length=2, max_length=300)
    life_area: LifeArea = LifeArea.GENERAL
    target_date: str | None = None
    timeframe_raw: str | None = None
    value: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("key")
    @classmethod
    def _normalise(cls, v: str) -> str:
        key = normalise_key(v)
        if not key:
            raise ValueError("empty key")
        return key

    @field_validator("life_area", mode="before")
    @classmethod
    def _default_area(cls, v):
        # LLMs occasionally invent areas ("work", "love"); fall back rather than reject.
        return v if v in LifeArea._value2member_map_ else LifeArea.GENERAL


class ExtractionResult(BaseModel):
    profile_updates: dict = Field(default_factory=dict)  # validated leniently by ProfileUpdate
    memories: list[ExtractedMemory] = Field(default_factory=list)
    extractor: str = "none"  # which strategy produced this (observability)

    def is_empty(self) -> bool:
        return not self.profile_updates and not self.memories


class ChangeAction(StrEnum):
    CREATED = "created"
    REINFORCED = "reinforced"
    UPDATED = "updated"
    RETRACTED = "retracted"
    PROFILE_UPDATED = "profile_updated"


class MemoryChange(BaseModel):
    """What the updater did — returned to the client as `memories_written`."""

    action: ChangeAction
    kind: str
    key: str
    text: str
    label: str
