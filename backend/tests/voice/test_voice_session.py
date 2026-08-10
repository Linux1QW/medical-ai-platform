"""Tests for voice session management and voice agent."""
from __future__ import annotations

import asyncio
import time

import pytest

from app.voice.agent import VoiceAgent, VoiceAgentConfig, VoiceTranscript
from app.voice.session import VOICE_ROOM_TTL_SECONDS, VoiceSession, VoiceSessionManager


# ---------------------------------------------------------------------------
# VoiceSession / VoiceSessionManager tests
# ---------------------------------------------------------------------------


class TestVoiceSession:
    """Tests for VoiceSession dataclass."""

    def test_create_voice_session(self) -> None:
        """Create session, verify room_name format and status."""
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=42, doctor_id=7)

        assert session.room_name.startswith("voice-42-")
        assert len(session.room_name) > len("voice-42-")
        assert session.status == "created"
        assert session.consultation_id == 42
        assert session.doctor_id == 7
        assert session.participant_count == 0

    def test_voice_session_ttl_10min(self) -> None:
        """expires_at - created_at == 600."""
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        ttl = session.expires_at - session.created_at
        assert ttl == pytest.approx(VOICE_ROOM_TTL_SECONDS, abs=1.0)

    def test_voice_session_expired(self) -> None:
        """Manually expire and check is_expired."""
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        # Force expiry by backdating expires_at
        session.expires_at = time.time() - 1
        assert session.is_expired is True

        # get_session should mark it expired
        fetched = manager.get_session(session.room_name)
        assert fetched is not None
        assert fetched.status == "expired"

    def test_voice_session_refresh(self) -> None:
        """Refresh extends TTL."""
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        old_expires = session.expires_at
        time.sleep(0.05)
        session.refresh_token()

        assert session.expires_at > old_expires
        assert session.is_expired is False

    def test_end_session(self) -> None:
        """End session → status='ended'."""
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        ended = manager.end_session(session.room_name)
        assert ended is not None
        assert ended.status == "ended"

    def test_end_nonexistent_session(self) -> None:
        """End a session that doesn't exist returns None."""
        manager = VoiceSessionManager()
        result = manager.end_session("nonexistent-room")
        assert result is None

    def test_generate_room_token(self) -> None:
        """Token starts with 'lk_token_'."""
        manager = VoiceSessionManager()
        token = manager.generate_room_token("test-room")

        assert token.startswith("lk_token_")
        assert len(token) > len("lk_token_")

    def test_list_active_sessions(self) -> None:
        """Only active sessions returned."""
        manager = VoiceSessionManager()

        s1 = manager.create_session(consultation_id=1, doctor_id=1)
        s2 = manager.create_session(consultation_id=2, doctor_id=2)
        s3 = manager.create_session(consultation_id=3, doctor_id=3)

        # End s2, expire s3
        manager.end_session(s2.room_name)
        s3.expires_at = time.time() - 1

        active = manager.list_active_sessions()
        active_rooms = {s.room_name for s in active}

        assert s1.room_name in active_rooms
        assert s2.room_name not in active_rooms
        assert s3.room_name not in active_rooms


# ---------------------------------------------------------------------------
# VoiceAgent tests
# ---------------------------------------------------------------------------


class TestVoiceAgent:
    """Tests for VoiceAgent."""

    def test_voice_agent_start_stop(self) -> None:
        """VoiceAgent start/stop lifecycle."""
        config = VoiceAgentConfig(
            room_name="test-room", consultation_id=1, doctor_id=1
        )
        agent = VoiceAgent(config)

        assert agent.is_running is False

        asyncio.get_event_loop().run_until_complete(agent.start())
        assert agent.is_running is True

        asyncio.get_event_loop().run_until_complete(agent.stop())
        assert agent.is_running is False

    def test_voice_agent_no_raw_audio(self) -> None:
        """save_raw_audio defaults to False."""
        config = VoiceAgentConfig(
            room_name="test-room", consultation_id=1, doctor_id=1
        )
        assert config.save_raw_audio is False

    def test_voice_transcript_processing(self) -> None:
        """Process transcript, verify stored."""
        config = VoiceAgentConfig(
            room_name="test-room", consultation_id=1, doctor_id=1
        )
        agent = VoiceAgent(config)

        transcript = VoiceTranscript(
            text="你好，请问哪里不舒服？",
            speaker="doctor",
            start_time=0.0,
            end_time=3.5,
            is_final=True,
        )

        result = asyncio.get_event_loop().run_until_complete(
            agent.process_transcript(transcript)
        )

        assert result["processed"] is True
        assert result["speaker"] == "doctor"
        assert len(agent.transcripts) == 1
        assert agent.transcripts[0].text == "你好，请问哪里不舒服？"

        # Summary
        summary = agent.get_transcript_summary()
        assert "[doctor]" in summary
        assert "你好" in summary
