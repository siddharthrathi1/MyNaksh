"""Application-level exceptions.

Infrastructure adapters translate vendor exceptions (neo4j, openai, anthropic) into these,
so the orchestration layer can apply fallbacks without knowing which vendor is behind them.
"""


class AppError(Exception):
    """Base class for all expected, handled application errors."""


class GraphUnavailableError(AppError):
    """The Shared Brain store cannot be reached. Callers should degrade, not crash."""


class LLMError(AppError):
    """A single LLM call failed. `retryable` tells the resilience layer whether to retry."""

    def __init__(self, message: str, *, retryable: bool, provider: str = "unknown"):
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider


class LLMUnavailableError(AppError):
    """Every configured provider failed (after retries)."""


class SessionOwnershipError(AppError):
    """A session id that belongs to another user was reused."""


class NotFoundError(AppError):
    """Requested entity does not exist."""
