"""HTTP request/response contracts (kept separate from domain models on purpose)."""

from datetime import date, datetime, time
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.core.models import MemoryItem, UserProfile
from app.memory.models import MemoryChange

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:\-]{1,64}$")]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatRequest(BaseModel):
    user_id: Identifier
    session_id: Identifier
    message: MessageText

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"user_id": "user-123", "session_id": "session-456", "message": "What should I focus on in my career?"}
            ]
        }
    }


class QueryAnalysisOut(BaseModel):
    intent: str
    life_areas: list[str]
    is_follow_up: bool
    is_memory_query: bool


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    context_used: list[str]
    memories_written: list[MemoryChange] = Field(
        default_factory=list, description="Empty when memory_update is 'scheduled' (background mode)."
    )
    memory_update: str = Field(description="applied | scheduled | skipped")
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    query_analysis: QueryAnalysisOut


class ProfileFields(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    dob: date | None = None
    tob: time | None = None
    birth_place: str | None = Field(default=None, max_length=100)
    preferred_language: str | None = Field(default=None, max_length=30)


class CreateUserRequest(ProfileFields):
    user_id: Identifier


class ProfileOut(BaseModel):
    user_id: str
    name: str | None
    dob: date | None
    tob: time | None
    birth_place: str | None
    preferred_language: str | None
    sun_sign: str | None
    missing_fields: list[str]
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def of(cls, p: UserProfile) -> "ProfileOut":
        return cls(**p.model_dump(), missing_fields=p.missing_fields())


class MemoryOut(BaseModel):
    id: str
    kind: str
    key: str
    text: str
    life_area: str
    target_date: str | None
    timeframe_raw: str | None
    value: str | None
    confidence: float
    importance: float
    status: str
    mention_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, m: MemoryItem) -> "MemoryOut":
        return cls(**m.model_dump(mode="json"))
