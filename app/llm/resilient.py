import logging

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.core.errors import LLMError, LLMUnavailableError
from app.llm.base import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.retryable


class ResilientLLM(LLMProvider):
    """Retry → fallback chain in front of concrete providers.

    For each provider in order: retry retryable errors (429/5xx/timeouts) with jittered
    exponential backoff; on exhaustion or a non-retryable error, move to the next provider.
    If all fail, raise `LLMUnavailableError` so the caller can serve a canned reply.
    """

    name = "resilient"

    def __init__(self, providers: list[LLMProvider], *, max_attempts: int = 2, base_delay_s: float = 0.5):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        for provider in self.providers:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._max_attempts),
                    wait=wait_exponential_jitter(initial=self._base_delay_s, max=4),
                    retry=retry_if_exception(_is_retryable),
                    reraise=True,
                ):
                    with attempt:
                        response = await provider.generate(request)
                        logger.info(
                            "llm_call purpose=%s provider=%s model=%s latency_ms=%.0f in_tok=%s out_tok=%s",
                            request.purpose, response.provider, response.model,
                            response.latency_ms, response.input_tokens, response.output_tokens,
                        )
                        return response
            except LLMError as exc:
                logger.warning("llm provider %s failed (%s); trying next", provider.name, exc)
        raise LLMUnavailableError(f"all LLM providers failed: {self.provider_names}")
