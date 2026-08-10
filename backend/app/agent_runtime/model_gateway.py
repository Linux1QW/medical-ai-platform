"""Model gateway protocol and production implementation.

The protocol allows tests to inject deterministic fakes while production
reuses the existing Qwen/OpenAI-compatible provider.
"""
from __future__ import annotations

import json
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.services.qwen_client import call_qwen_chat

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class CoachModelGateway(Protocol):
    """Structured LLM completion gateway for the Coach graph."""

    async def complete_structured(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1024,
        timeout: float = 6.0,
    ) -> T:
        """Call the LLM and parse the response into a Pydantic schema."""
        ...


class QwenModelGateway:
    """Production gateway backed by the Qwen chat API."""

    def __init__(self, *, model: str | None = None, temperature: float = 0.2) -> None:
        self._model = model
        self._temperature = temperature

    async def complete_structured(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1024,
        timeout: float = 6.0,
    ) -> T:
        """Call Qwen and parse JSON response into the requested schema.

        If parsing fails, raises ValueError with the raw text for the caller
        to handle (e.g. fallback to a safe default).
        """
        raw = await call_qwen_chat(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=max_tokens,
        )
        return _parse_json(raw, schema)


def _parse_json(raw: str, schema: type[T]) -> T:
    """Best-effort JSON extraction from LLM output."""
    text = raw.strip()
    # Try to find JSON block inside markdown fences
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model output as JSON: {exc}") from exc
    return schema.model_validate(data)
