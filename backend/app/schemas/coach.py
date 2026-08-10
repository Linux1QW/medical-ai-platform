"""Coach API request/response schemas."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class CoachStreamRequest(BaseModel):
    """Request to start a coach suggestion stream."""
    consultation_id: int
    doctor_id: int
    latest_message: str = Field(min_length=1, max_length=5000)
    idempotency_key: str = Field(min_length=8, max_length=64)


class CoachStateResponse(BaseModel):
    """Current coach state for a consultation."""
    consultation_id: int
    status: Literal["disabled", "idle", "thinking", "suggestion", "degraded", "error"]
    current_stage: str | None = None
    turn_no: int = 0


class SuggestionFeedbackRequest(BaseModel):
    """Feedback on a coach suggestion."""
    feedback: Literal["accepted", "rejected", "ignored"]
    reason: str | None = Field(default=None, max_length=500)


class SuggestionFeedbackResponse(BaseModel):
    """Response after recording feedback."""
    suggestion_id: UUID
    feedback: str
    recorded: bool


class SSEEvent(BaseModel):
    """A Server-Sent Event payload."""
    event: str  # "thinking", "suggestion", "error", "done"
    data: dict | str
    id: str | None = None
