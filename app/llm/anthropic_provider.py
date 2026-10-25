import time

import anthropic
from anthropic import AsyncAnthropic

from app.core.errors import LLMError
from app.llm.base import LLMProvider, LLMRequest, LLMResponse

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout_s: float = 30.0,
        effort: str = "low",
        refusal_fallback: bool = True,
    ):
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model or DEFAULT_MODEL
        self._effort = effort
        self._refusal_fallback = refusal_fallback

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Current Claude models think adaptively and reject sampling params (temperature),
        # so depth/cost is controlled through `effort` instead.
        extra_body: dict = {"output_config": {"effort": self._effort}}
        extra_headers: dict = {}
        if self._refusal_fallback:
            # Server-side fallback: if a safety classifier declines, the API reroutes the
            # request to a fallback model inside the same call.
            extra_headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
            extra_body["fallbacks"] = "default"

        system = request.system
        if request.json_mode:
            system += "\n\nRespond with a single JSON object only — no prose, no code fences."

        started = time.perf_counter()
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                extra_body=extra_body,
                extra_headers=extra_headers or None,
            )
        except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise LLMError(str(exc), retryable=True, provider=self.name) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(str(exc), retryable=exc.status_code >= 500, provider=self.name) from exc

        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request", retryable=False, provider=self.name)

        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
