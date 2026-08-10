"""Voice session management for LiveKit integration."""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID, uuid4

VOICE_ROOM_TTL_SECONDS = 600  # 10 minutes
VOICE_MAX_PARTICIPANTS = 4


@dataclass
class VoiceSession:
    """A voice consultation session."""
    session_id: UUID
    room_name: str
    consultation_id: int
    doctor_id: int
    status: Literal["created", "active", "ended", "expired"] = "created"
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    participant_count: int = 0
    transcript_segments: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + VOICE_ROOM_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def refresh_token(self) -> None:
        """Refresh the session TTL."""
        self.expires_at = time.time() + VOICE_ROOM_TTL_SECONDS


class VoiceSessionManager:
    """Manages voice sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}  # room_name → session

    def create_session(
        self,
        *,
        consultation_id: int,
        doctor_id: int,
    ) -> VoiceSession:
        """Create a new voice session."""
        session_id = uuid4()
        room_name = f"voice-{consultation_id}-{session_id.hex[:8]}"

        session = VoiceSession(
            session_id=session_id,
            room_name=room_name,
            consultation_id=consultation_id,
            doctor_id=doctor_id,
        )
        self._sessions[room_name] = session
        return session

    def get_session(self, room_name: str) -> VoiceSession | None:
        """Get a voice session by room name."""
        session = self._sessions.get(room_name)
        if session and session.is_expired:
            session.status = "expired"
        return session

    def end_session(self, room_name: str) -> VoiceSession | None:
        """End a voice session."""
        session = self._sessions.get(room_name)
        if session:
            session.status = "ended"
        return session

    def generate_room_token(
        self,
        room_name: str,
        *,
        api_key: str = "dev-api-key",
        api_secret: str = "dev-api-secret",
    ) -> str:
        """Generate a room token (HMAC-based for demo).

        In production, this would use the LiveKit API to generate a proper JWT.
        For the beta, we use HMAC-SHA256 as a placeholder.
        """
        payload = f"{room_name}:{int(time.time())}:{api_key}"
        token = hmac.new(
            api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"lk_token_{token[:32]}"

    def list_active_sessions(self) -> list[VoiceSession]:
        """List all active (non-expired, non-ended) sessions."""
        return [
            s
            for s in self._sessions.values()
            if s.status in ("created", "active") and not s.is_expired
        ]
