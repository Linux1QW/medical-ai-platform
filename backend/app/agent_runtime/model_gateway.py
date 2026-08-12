"""Model gateway protocol and production implementation.

The protocol allows tests to inject deterministic fakes while production
reuses the existing Qwen/OpenAI-compatible provider.

Timeout and error handling:
- complete_structured enforces a real timeout via asyncio.wait_for.
- Model or JSON parse failures raise stable error codes (CoachModelError).
- Raw model text and exception details only go to controlled logs.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.services.qwen_client import call_qwen_chat

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CoachModelError(Exception):
    """Stable error for model gateway failures.

    Attributes:
        error_code: Machine-readable error code for client-facing messages.
    """

    def __init__(self, error_code: str, detail: str = "") -> None:
        self.error_code = error_code
        super().__init__(f"{error_code}: {detail}" if detail else error_code)


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

        Raises:
            CoachModelError: With stable error codes for timeout, API failure,
                or JSON parse failure.
        """
        try:
            raw = await asyncio.wait_for(
                call_qwen_chat(
                    messages,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Model gateway timeout after %.1fs", timeout)
            raise CoachModelError("COACH_MODEL_TIMEOUT", f"timeout={timeout}s") from None
        except Exception as exc:
            logger.warning("Model gateway API error: %s", str(exc)[:200])
            raise CoachModelError("COACH_MODEL_ERROR", str(exc)[:200]) from exc

        return _parse_json(raw, schema)


def _parse_json(raw: str, schema: type[T]) -> T:
    """Best-effort JSON extraction from LLM output.

    Raises:
        CoachModelError: If JSON parsing or schema validation fails.
    """
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
        logger.warning("Model output JSON parse failed: %s", str(exc)[:200])
        raise CoachModelError("COACH_JSON_PARSE", str(exc)[:200]) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.warning("Model output schema validation failed: %s", str(exc)[:200])
        raise CoachModelError("COACH_SCHEMA_ERROR", str(exc)[:200]) from exc
