"""Voice session management for LiveKit integration.

Uses official LiveKit SDK JWT builder for access token generation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

VOICE_ROOM_TTL_SECONDS = 600  # 10 minutes
VOICE_MAX_PARTICIPANTS = 4


def _build_livekit_token(
    room_name: str,
    identity: str,
    *,
    api_key: str,
    api_secret: str,
    ttl_seconds: int = VOICE_ROOM_TTL_SECONDS,
) -> str:
    """Generate an official LiveKit access token using the SDK JWT builder.

    Args:
        room_name: LiveKit room name.
        identity: Participant identity (e.g. "doctor-<id>").
        api_key: LiveKit API key (from settings, never exposed in response).
        api_secret: LiveKit API secret (from settings, never exposed in response/logs).
        ttl_seconds: Token TTL, capped at 600 seconds.

    Returns:
        Signed JWT string.
    """
    from livekit import api as lk_api

    # Cap TTL at 10 minutes
    ttl = min(ttl_seconds, VOICE_ROOM_TTL_SECONDS)

    token = (
        lk_api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            lk_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(ttl)
    )
    jwt_str = token.to_jwt()
    return str(jwt_str)


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
    """Manages voice sessions and token generation."""

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
        identity: str,
        *,
        api_key: str,
        api_secret: str,
        ttl_seconds: int = VOICE_ROOM_TTL_SECONDS,
    ) -> str:
        """Generate a LiveKit room access token.

        Uses the official LiveKit SDK JWT builder.
        API secret is never logged or included in responses.
        """
        return _build_livekit_token(
            room_name=room_name,
            identity=identity,
            api_key=api_key,
            api_secret=api_secret,
            ttl_seconds=ttl_seconds,
        )

    def list_active_sessions(self) -> list[VoiceSession]:
        """List all active (non-expired, non-ended) sessions."""
        return [
            s
            for s in self._sessions.values()
            if s.status in ("created", "active") and not s.is_expired
        ]
