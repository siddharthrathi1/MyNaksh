"""Deterministic LLM for tests and key-less local runs.

It records every request (so tests can assert exactly what context reached the model) and
produces a reply that echoes the facts it was given — enough to demo the memory loop
end-to-end without an API key.
"""

import re
import time
from collections.abc import Callable

from app.core.errors import LLMError
from app.llm.base import LLMProvider, LLMRequest, LLMResponse

Responder = Callable[[LLMRequest], str]

_FACTS = re.compile(r"<known_facts>\n(.*?)\n</known_facts>", re.S)


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, responder: Responder | None = None, *, fail_times: int = 0, fail_always: bool = False):
        self.responder = responder
        self.fail_times = fail_times
        self.fail_always = fail_always
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.fail_always or self.fail_times > 0:
            self.fail_times = max(0, self.fail_times - 1)
            raise LLMError("simulated provider outage", retryable=True, provider=self.name)

        started = time.perf_counter()
        text = self.responder(request) if self.responder else self._default(request)
        return LLMResponse(
            text=text, provider=self.name, model="fake", latency_ms=(time.perf_counter() - started) * 1000
        )

    @property
    def chat_calls(self) -> list[LLMRequest]:
        return [c for c in self.calls if c.purpose == "chat"]

    @staticmethod
    def _default(request: LLMRequest) -> str:
        if request.json_mode:
            return '{"profile_updates": {}, "memories": []}'
        question = request.messages[-1].content if request.messages else ""
        match = _FACTS.search(request.system)
        facts = match.group(1).replace("\n", "; ") if match else "nothing stored yet"
        return f"[offline reply] You asked: \"{question}\". What I know about you: {facts}."
