"""Voice API schemas."""
from __future__ import annotations

from pydantic import BaseModel


class CreateVoiceSessionRequest(BaseModel):
    consultation_id: int


class VoiceSessionResponse(BaseModel):
    session_id: str
    room_name: str
    consultation_id: int
    status: str
    room_token: str
    expires_at: float


class VoiceStateResponse(BaseModel):
    room_name: str
    status: str
    participant_count: int
    is_expired: bool
