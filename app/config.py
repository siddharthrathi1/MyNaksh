from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, sourced from environment / .env (12-factor)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # --- Shared Brain storage ---
    brain_backend: Literal["neo4j", "memory"] = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mynaksh-dev-pass"
    neo4j_pool_size: int = 50
    neo4j_timeout_s: float = 5.0

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic", "fake"] = "openai"
    llm_model: str | None = None  # None → provider default
    llm_fallback_provider: Literal["openai", "anthropic", "fake"] | None = None
    llm_fallback_model: str | None = None
    llm_timeout_s: float = 30.0
    llm_max_attempts: int = 2  # per provider, before falling through to the fallback
    openai_api_key: str | None = None
    # OpenAI-compatible server (vLLM, Ollama, LM Studio...). When set, an API key is optional.
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_effort: Literal["low", "medium", "high"] = "low"  # chat replies are short
    anthropic_refusal_fallback: bool = True

    # --- Memory & context ---
    # "auto" = LLM extraction with rule fallback, or rules only when the provider is fake.
    extraction_mode: Literal["auto", "llm", "rules"] = "auto"
    # Turn analysis (intent, topics, "anything worth remembering?"): same auto/llm/rules semantics.
    analysis_mode: Literal["auto", "llm", "rules"] = "auto"
    # "sync" returns memories_written in the response (demo/testing);
    # "background" moves extraction off the request path (lower latency, eventual consistency).
    memory_write_mode: Literal["sync", "background"] = "sync"
    memory_confidence_threshold: float = 0.6
    max_memories_per_turn: int = 5
    short_term_turns: int = 6  # user+assistant pairs kept as short-term context
    max_memories_in_context: int = 8
    context_token_budget: int = 1500  # rough budget for profile + memories + history

    @field_validator(
        "llm_model", "llm_fallback_provider", "llm_fallback_model", "openai_api_key", "openai_base_url", "anthropic_api_key",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v):
        # `KEY=` in .env means "unset", not "empty string".
        return None if isinstance(v, str) and not v.strip() else v


@lru_cache
def get_settings() -> Settings:
    return Settings()
