"""Thin LLM clients: Anthropic, OpenAI (and any OpenAI-compatible server such as
Ollama, Groq, Together, vLLM via ``SPECX_LLM_BASE_URL``), plus a callable wrapper for tests.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol

from .config import Settings


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`pip install anthropic` to use the Anthropic provider.") from exc
        self.name = f"anthropic:{model}"
        self.model, self.max_tokens = model, max_tokens
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    def __init__(self, model: str, max_tokens: int, base_url: str = ""):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`pip install openai` to use the OpenAI / OpenAI-compatible provider.") from exc
        self.name = f"openai:{model}" + (f"@{base_url}" if base_url else "")
        self.model, self.max_tokens = model, max_tokens
        # local servers (Ollama) ignore the key but the SDK insists on one
        self.client = OpenAI(base_url=base_url or None, api_key=os.getenv("OPENAI_API_KEY") or "not-needed")

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        try:
            resp = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception:                      # server without JSON mode
            resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class CallableLLM:
    """Wrap ``fn(system, user) -> str``; handy for tests and notebooks."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "callable"):
        self.fn, self.name = fn, name

    def complete(self, system: str, user: str) -> str:
        return self.fn(system, user)


def get_llm(settings: Settings) -> Optional[LLM]:
    provider = settings.resolved_llm_provider()
    model = settings.resolved_llm_model()
    if provider == "anthropic":
        return AnthropicLLM(model, settings.llm_max_tokens)
    if provider == "openai":
        return OpenAILLM(model, settings.llm_max_tokens, settings.llm_base_url)
    return None
