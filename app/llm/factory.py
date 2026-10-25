from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.fake import FakeLLM
from app.llm.resilient import ResilientLLM


def _build_provider(name: str, model: str | None, settings: Settings) -> LLMProvider:
    # Vendor SDKs are imported lazily so an unused provider never needs to be installed/configured.
    if name == "openai":
        if not settings.openai_api_key and not settings.openai_base_url:
            raise ValueError("LLM provider 'openai' requires OPENAI_API_KEY or OPENAI_BASE_URL")
        from app.llm.openai_provider import OpenAIProvider

        # Local OpenAI-compatible servers usually ignore the key, but the SDK requires one.
        return OpenAIProvider(
            settings.openai_api_key or "EMPTY",
            model,
            settings.llm_timeout_s,
            base_url=settings.openai_base_url,
        )
    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM provider 'anthropic' requires ANTHROPIC_API_KEY")
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings.anthropic_api_key,
            model,
            settings.llm_timeout_s,
            effort=settings.anthropic_effort,
            refusal_fallback=settings.anthropic_refusal_fallback,
        )
    if name == "fake":
        return FakeLLM()
    raise ValueError(f"unknown LLM provider: {name}")


def build_llm(settings: Settings) -> ResilientLLM:
    providers = [_build_provider(settings.llm_provider, settings.llm_model, settings)]
    if settings.llm_fallback_provider:
        providers.append(
            _build_provider(settings.llm_fallback_provider, settings.llm_fallback_model, settings)
        )
    return ResilientLLM(providers, max_attempts=settings.llm_max_attempts)
