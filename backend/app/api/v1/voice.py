"""Voice API endpoints (Beta).

All endpoints require authentication.
Token generation uses LiveKit SDK.
Non-owner access returns 403.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import require_consultation_access
from app.core.config import settings
from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.voice import (
    CreateVoiceSessionRequest,
    VoiceSessionResponse,
    VoiceStateResponse,
)
from app.voice.session import VoiceSessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

_manager = VoiceSessionManager()


def get_manager() -> VoiceSessionManager:
    return _manager


def _require_livekit_config() -> tuple[str, str]:
    """Ensure LiveKit configuration is present; raise 503 if not."""
    api_key = settings.LIVEKIT_API_KEY
    api_secret = settings.LIVEKIT_API_SECRET
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "VOICE_NOT_CONFIGURED",
                "message": "语音功能未配置 LiveKit 凭据",
            },
        )
    return api_key.get_secret_value(), api_secret.get_secret_value()


@router.post("/sessions", response_model=VoiceSessionResponse)
async def create_voice_session(
    request: CreateVoiceSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceSessionResponse:
    """Create a new voice session.

    - Requires authentication
    - Checks consultation ownership (non-owner → 403)
    - Returns LiveKit room token (TTL ≤ 10 min)
    """
    # Verify ownership / access
    consultation = await require_consultation_access(
        db, request.consultation_id, current_user
    )

    api_key, api_secret = _require_livekit_config()

    session = _manager.create_session(
        consultation_id=request.consultation_id,
        doctor_id=consultation.doctor_id,
    )

    identity = f"doctor-{current_user.id}"
    token = _manager.generate_room_token(
        session.room_name,
        identity,
        api_key=api_key,
        api_secret=api_secret,
    )

    return VoiceSessionResponse(
        session_id=str(session.session_id),
        room_name=session.room_name,
        consultation_id=session.consultation_id,
        status=session.status,
        room_token=token,
        expires_at=session.expires_at,
    )


@router.get("/sessions/{room_name}/state", response_model=VoiceStateResponse)
async def get_voice_state(
    room_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceStateResponse:
    """Get voice session state.

    - Requires authentication
    - Non-owner → 403
    """
    session = _manager.get_session(room_name)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": f"Session {room_name} not found"},
        )

    # Check ownership
    await require_consultation_access(db, session.consultation_id, current_user)

    return VoiceStateResponse(
        room_name=session.room_name,
        status=session.status,
        participant_count=session.participant_count,
        is_expired=session.is_expired,
    )


@router.post("/sessions/{room_name}/end")
async def end_voice_session(
    room_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """End a voice session.

    - Requires authentication
    - Non-owner → 403
    """
    session = _manager.get_session(room_name)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": f"Session {room_name} not found"},
        )

    # Check ownership
    await require_consultation_access(db, session.consultation_id, current_user)

    _manager.end_session(room_name)
    return {"room_name": room_name, "status": "ended"}
