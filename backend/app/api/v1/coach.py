"""Coach API endpoints."""
from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from app.schemas.coach import (
    CoachStateResponse,
    CoachStreamRequest,
    SuggestionFeedbackRequest,
    SuggestionFeedbackResponse,
)
from app.services.coach_service import CoachService

router = APIRouter(prefix="/coach", tags=["coach"])

# Module-level service instance (in production, inject via DI)
_coach_service = CoachService()


def get_coach_service() -> CoachService:
    return _coach_service


@router.post("/consultations/{consultation_id}/suggestions/stream")
async def stream_coach_suggestion(
    consultation_id: int,
    request: CoachStreamRequest,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream coach suggestions via SSE.

    Supports idempotency via Idempotency-Key header.
    Supports replay via Last-Event-ID header.
    """
    service = get_coach_service()

    async def event_generator():
        async for event in service.stream_suggestion(
            consultation_id=consultation_id,
            doctor_id=request.doctor_id,
            latest_message=request.latest_message,
            idempotency_key=request.idempotency_key,
        ):
            event_name = event.get("event", "message")
            data = event.get("data", {})
            event_id = event.get("id")

            yield f"event: {event_name}\n"
            if event_id:
                yield f"id: {event_id}\n"
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/consultations/{consultation_id}/state", response_model=CoachStateResponse)
async def get_coach_state(consultation_id: int) -> CoachStateResponse:
    """Get current coach state for a consultation."""
    service = get_coach_service()
    state = service.get_state(consultation_id)
    return CoachStateResponse(**state)


@router.post("/suggestions/{suggestion_id}/feedback", response_model=SuggestionFeedbackResponse)
async def submit_suggestion_feedback(
    suggestion_id: UUID,
    request: SuggestionFeedbackRequest,
) -> SuggestionFeedbackResponse:
    """Submit feedback on a coach suggestion."""
    service = get_coach_service()
    result = service.record_feedback(
        suggestion_id=suggestion_id,
        feedback=request.feedback,
        reason=request.reason,
    )
    return SuggestionFeedbackResponse(**result)
