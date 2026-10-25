import time

import openai
from openai import AsyncOpenAI

from app.core.errors import LLMError
from app.llm.base import LLMProvider, LLMRequest, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout_s: float = 30.0,
        base_url: str | None = None,
    ):
        # SDK retries disabled: ResilientLLM owns retry/fallback policy in one place.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0
        )
        self.model = model or DEFAULT_MODEL

    async def generate(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": "system", "content": request.system}]
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        extra = {"response_format": {"type": "json_object"}} if request.json_mode else {}

        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                **extra,
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise LLMError(str(exc), retryable=True, provider=self.name) from exc
        except openai.APIStatusError as exc:
            raise LLMError(str(exc), retryable=exc.status_code >= 500, provider=self.name) from exc

        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
