"""Coach API endpoints.

All endpoints require authentication via current_user.
- Stream / state / feedback: require coach:use permission.
- Admin trace: requires coach:trace:view permission.
- Consultation ownership enforced (no IDOR).
- No body doctor_id or consultation_id — from URL and auth token.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import require_permission
from app.db.session import get_db
from app.models.consultation import Consultation
from app.models.user import User
from app.schemas.coach import (
    CoachStateResponse,
    CoachStreamRequest,
    SuggestionFeedbackRequest,
    SuggestionFeedbackResponse,
)
from app.services.coach_service import CoachDisabledError, CoachService

router = APIRouter(prefix="/coach", tags=["coach"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _get_service(db: AsyncSession) -> CoachService:
    return CoachService(db)


async def _verify_consultation_ownership(
    consultation_id: int,
    doctor_id: int,
    db: AsyncSession,
) -> Consultation:
    """Verify the authenticated user owns the consultation (IDOR guard).

    Admins bypass ownership check.
    """
    stmt = select(Consultation).where(Consultation.id == consultation_id)
    result = await db.execute(stmt)
    consultation = result.scalar_one_or_none()

    if consultation is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "COACH_CONSULTATION_NOT_FOUND", "message": "Consultation not found"},
        )

    if consultation.doctor_id != doctor_id:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "COACH_IDOR_DENIED", "message": "Cannot access another doctor's consultation"},
        )

    return consultation


# ------------------------------------------------------------------
# Stream endpoint
# ------------------------------------------------------------------


@router.post("/consultations/{consultation_id}/suggestions/stream")
async def stream_coach_suggestion(
    consultation_id: int,
    request: CoachStreamRequest,
    current_user: User = require_permission("coach:use"),
    db: AsyncSession = Depends(get_db),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream coach suggestions via SSE.

    - Idempotency via idempotency_key in body.
    - Replay via Last-Event-ID header.
    - Heartbeat comments every 15 seconds.
    """
    service = await _get_service(db)

    # Disabled → 409
    if not service.enabled:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "FEATURE_DISABLED",
                "message": "Coach feature is disabled",
            },
        )

    # IDOR guard
    await _verify_consultation_ownership(consultation_id, current_user.id, db)

    async def event_generator():
        try:
            async for event in service.stream_suggestion(
                consultation_id=consultation_id,
                doctor_id=current_user.id,
                latest_message=request.latest_message,
                idempotency_key=request.idempotency_key,
                last_event_id=last_event_id,
            ):
                event_name = event.get("event", "message")
                data = event.get("data", {})
                event_id = event.get("id")

                yield f"event: {event_name}\n"
                if event_id:
                    yield f"id: {event_id}\n"
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except CoachDisabledError:
            # Should not reach here due to pre-check, but guard anyway
            yield f"event: error\ndata: {json.dumps({'message': 'Coach feature is disabled'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# State endpoint
# ------------------------------------------------------------------


@router.get("/consultations/{consultation_id}/state", response_model=CoachStateResponse)
async def get_coach_state(
    consultation_id: int,
    current_user: User = require_permission("coach:use"),
    db: AsyncSession = Depends(get_db),
) -> CoachStateResponse:
    """Get current coach state for a consultation."""
    service = await _get_service(db)

    # IDOR guard
    await _verify_consultation_ownership(consultation_id, current_user.id, db)

    state = await service.get_state(consultation_id)
    return CoachStateResponse(**state)


# ------------------------------------------------------------------
# Feedback endpoint
# ------------------------------------------------------------------


@router.post("/suggestions/{suggestion_id}/feedback", response_model=SuggestionFeedbackResponse)
async def submit_suggestion_feedback(
    suggestion_id: UUID,
    request: SuggestionFeedbackRequest,
    current_user: User = require_permission("coach:use"),
    db: AsyncSession = Depends(get_db),
) -> SuggestionFeedbackResponse:
    """Submit feedback on a coach suggestion."""
    service = await _get_service(db)
    result = await service.record_feedback(
        suggestion_id=suggestion_id,
        feedback=request.feedback,
        doctor_id=current_user.id,
        reason=request.reason,
    )
    return SuggestionFeedbackResponse(**result)


# ------------------------------------------------------------------
# Admin trace endpoint
# ------------------------------------------------------------------


@router.get("/admin/consultations/{consultation_id}/trace")
async def get_coach_trace(
    consultation_id: int,
    current_user: User = require_permission("coach:trace:view"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Admin-only: get full coach trace for a consultation."""
    service = await _get_service(db)
    return await service.get_trace(consultation_id)
