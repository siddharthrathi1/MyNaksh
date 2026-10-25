import logging
from datetime import date, time

from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from app.brain.repository import BrainRepository
from app.core.clock import Clock
from app.core.models import UserProfile
from app.profile.astro import sun_sign

logger = logging.getLogger(__name__)

# Normalise ISO codes to display names so "hi" and "Hindi" are the same graph node.
LANGUAGE_CODES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
}


class ProfileUpdate(BaseModel):
    """Partial profile update — from the API or extracted from conversation."""

    name: str | None = None
    dob: date | None = None
    tob: time | None = None
    birth_place: str | None = None
    preferred_language: str | None = None

    @field_validator("name", "birth_place", "preferred_language")
    @classmethod
    def _clean(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split()).strip()
        if not v:
            return None
        if info.field_name == "preferred_language":
            return LANGUAGE_CODES.get(v.lower(), v.title())
        return v.title()

    @field_validator("dob")
    @classmethod
    def _not_future(cls, v: date | None) -> date | None:
        if v and v > date.today():
            raise ValueError("date of birth cannot be in the future")
        return v

    @classmethod
    def lenient(cls, raw: dict) -> "ProfileUpdate":
        """Validate field-by-field, dropping invalid values instead of failing the whole update.

        Used for LLM-extracted data, where one bad field (e.g. an unparseable date) must not
        discard the good ones.
        """
        clean: dict = {}
        for key, value in raw.items():
            if key not in cls.model_fields or value in (None, ""):
                continue
            try:
                cls.model_validate({key: value})
                clean[key] = value
            except ValidationError:
                logger.info("dropping invalid extracted profile field %s=%r", key, value)
        return cls.model_validate(clean)

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)


class ProfileService:
    def __init__(self, repo: BrainRepository, clock: Clock):
        self._repo = repo
        self._clock = clock

    async def get(self, user_id: str) -> UserProfile | None:
        return await self._repo.get_profile(user_id)

    async def get_or_create(self, user_id: str) -> UserProfile:
        """Unknown users are created on first contact; the bot then asks for details naturally."""
        profile = await self._repo.get_profile(user_id)
        if profile:
            return profile
        now = self._clock()
        return await self._repo.upsert_profile(UserProfile(user_id=user_id, created_at=now, updated_at=now))

    async def apply(self, user_id: str, update: ProfileUpdate) -> tuple[UserProfile, list[str]]:
        """Merge non-null fields into the stored profile. Returns (profile, changed_field_names)."""
        current = await self.get_or_create(user_id)
        changes = {
            field: value
            for field, value in update.model_dump(exclude_none=True).items()
            if getattr(current, field) != value
        }
        if not changes:
            return current, []

        merged = current.model_copy(update={**changes, "updated_at": self._clock()})
        # Derived astrology attributes are recomputed whenever their inputs change.
        merged.sun_sign = sun_sign(merged.dob) if merged.dob else None
        await self._repo.upsert_profile(merged)
        logger.info("profile updated user=%s fields=%s", user_id, sorted(changes))
        return merged, sorted(changes)
