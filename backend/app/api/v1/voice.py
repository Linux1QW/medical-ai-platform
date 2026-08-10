"""Voice API endpoints (Beta)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.voice import (
    CreateVoiceSessionRequest,
    VoiceSessionResponse,
    VoiceStateResponse,
)
from app.voice.session import VoiceSessionManager

router = APIRouter(prefix="/voice", tags=["voice"])

_manager = VoiceSessionManager()


def get_manager() -> VoiceSessionManager:
    return _manager


@router.post("/sessions", response_model=VoiceSessionResponse)
async def create_voice_session(
    request: CreateVoiceSessionRequest,
) -> VoiceSessionResponse:
    """Create a new voice session."""
    session = _manager.create_session(
        consultation_id=request.consultation_id,
        doctor_id=request.doctor_id,
    )
    token = _manager.generate_room_token(session.room_name)
    return VoiceSessionResponse(
        session_id=str(session.session_id),
        room_name=session.room_name,
        consultation_id=session.consultation_id,
        status=session.status,
        room_token=token,
        expires_at=session.expires_at,
    )


@router.get("/sessions/{room_name}/state", response_model=VoiceStateResponse)
async def get_voice_state(room_name: str) -> VoiceStateResponse:
    """Get voice session state."""
    session = _manager.get_session(room_name)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session {room_name} not found"
        )
    return VoiceStateResponse(
        room_name=session.room_name,
        status=session.status,
        participant_count=session.participant_count,
        is_expired=session.is_expired,
    )


@router.post("/sessions/{room_name}/end")
async def end_voice_session(room_name: str) -> dict:
    """End a voice session."""
    session = _manager.end_session(room_name)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session {room_name} not found"
        )
    return {"room_name": room_name, "status": "ended"}
