"""Provider-agnostic LLM interface.

The application only ever sees `LLMProvider.generate(LLMRequest) -> LLMResponse`.
Swapping OpenAI ↔ Anthropic ↔ a local model is a config change, not a code change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: list[LLMMessage]
    max_tokens: int = 2048
    temperature: float = 0.4  # honoured by providers that support it
    json_mode: bool = False  # ask for a single JSON object back
    purpose: str = "chat"  # "chat" | "extraction" — for logging, metrics and routing


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    meta: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Implementations must raise `LLMError` (with `retryable` set) on failure."""

    name: str = "base"

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
