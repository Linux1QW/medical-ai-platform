"""Tests for voice session management, voice agent, session store, and worker."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.agent import VoiceAgent, VoiceAgentConfig, VoiceTranscript
from app.voice.session import VOICE_ROOM_TTL_SECONDS, VoiceSession, VoiceSessionManager
from app.voice.session_store import (
    VoiceSessionStoreMemory,
    create_voice_store,
)
from app.voice.worker import VoiceWorker

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
        """End session -> status='ended'."""
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


class TestVoiceSessionTokenGeneration:
    """Tests for LiveKit token generation."""

    def _setup_mock(self):
        """Create mock objects for livekit module."""
        mock_token_instance = MagicMock()
        mock_token_instance.with_identity.return_value = mock_token_instance
        mock_token_instance.with_grants.return_value = mock_token_instance
        mock_token_instance.with_ttl.return_value = mock_token_instance
        mock_token_instance.to_jwt.return_value = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJ0ZXN0In0.signature"

        mock_lk_api = MagicMock()
        mock_lk_api.AccessToken.return_value = mock_token_instance

        mock_livekit = MagicMock()
        mock_livekit.api = mock_lk_api

        return mock_livekit, mock_lk_api, mock_token_instance

    def test_generate_room_token_returns_jwt(self) -> None:
        """generate_room_token returns a JWT-like string (3 dot-separated parts)."""
        manager = VoiceSessionManager()
        mock_livekit, mock_lk_api, mock_token_instance = self._setup_mock()

        with patch.dict("sys.modules", {"livekit": mock_livekit, "livekit.api": mock_lk_api}):
            token = manager.generate_room_token(
                "test-room",
                "doctor-1",
                api_key="test-key",
                api_secret="test-secret",
            )

        assert token == "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJ0ZXN0In0.signature"
        # JWT has 3 parts separated by dots
        assert len(token.split(".")) == 3

    def test_generate_room_token_identity(self) -> None:
        """Token generation includes correct identity."""
        manager = VoiceSessionManager()
        mock_livekit, mock_lk_api, mock_token_instance = self._setup_mock()

        with patch.dict("sys.modules", {"livekit": mock_livekit, "livekit.api": mock_lk_api}):
            manager.generate_room_token(
                "test-room",
                "doctor-42",
                api_key="test-key",
                api_secret="test-secret",
            )

        # Verify identity was set
        mock_token_instance.with_identity.assert_called_once_with("doctor-42")

    def test_generate_room_token_ttl_capped(self) -> None:
        """Token TTL is capped at VOICE_ROOM_TTL_SECONDS (600)."""
        manager = VoiceSessionManager()
        mock_livekit, mock_lk_api, mock_token_instance = self._setup_mock()

        with patch.dict("sys.modules", {"livekit": mock_livekit, "livekit.api": mock_lk_api}):
            # Request 3600s TTL, should be capped to 600
            manager.generate_room_token(
                "test-room",
                "doctor-1",
                api_key="test-key",
                api_secret="test-secret",
                ttl_seconds=3600,
            )

        # Verify TTL was capped
        mock_token_instance.with_ttl.assert_called_once_with(VOICE_ROOM_TTL_SECONDS)


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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(agent.start())
            assert agent.is_running is True

            loop.run_until_complete(agent.stop())
            assert agent.is_running is False
        finally:
            loop.close()

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
            room_sid="room-1",
            participant_sid="participant-1",
            turn_id="turn-1",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                agent.process_transcript(transcript)
            )
        finally:
            loop.close()

        assert result["processed"] is True
        assert result["persisted"] is True
        assert result["speaker"] == "doctor"
        assert len(agent.transcripts) == 1
        assert agent.transcripts[0].text == "你好，请问哪里不舒服？"

        # Summary
        summary = agent.get_transcript_summary()
        assert "[doctor]" in summary
        assert "你好" in summary

    def test_duplicate_final_transcript_creates_one_message(self) -> None:
        """Duplicate final transcript (same room_sid, participant_sid, turn_id) creates only one message."""
        config = VoiceAgentConfig(
            room_name="test-room", consultation_id=1, doctor_id=1
        )
        agent = VoiceAgent(config)

        transcript = VoiceTranscript(
            text="你好",
            speaker="doctor",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
            room_sid="room-1",
            participant_sid="participant-1",
            turn_id="turn-1",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result1 = loop.run_until_complete(agent.process_transcript(transcript))
            # Same dedup key -> should be skipped
            result2 = loop.run_until_complete(agent.process_transcript(transcript))
        finally:
            loop.close()

        assert result1["processed"] is True
        assert result1["duplicated"] is False
        assert result2["processed"] is False
        assert result2["duplicated"] is True

        # Only one transcript stored (dedup worked)
        assert len(agent.transcripts) == 1

    def test_partial_transcript_memory_only(self) -> None:
        """Partial transcripts are stored in memory but not persisted."""
        config = VoiceAgentConfig(
            room_name="test-room", consultation_id=1, doctor_id=1
        )
        agent = VoiceAgent(config)

        partial = VoiceTranscript(
            text="你",
            speaker="doctor",
            start_time=0.0,
            end_time=0.5,
            is_final=False,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent.process_transcript(partial))
        finally:
            loop.close()

        assert result["processed"] is True
        assert result["persisted"] is False
        # Partial transcripts are in memory
        assert len(agent.transcripts) == 1
        # But not in summary (only final)
        summary = agent.get_transcript_summary()
        assert summary == ""


# ---------------------------------------------------------------------------
# VoiceSessionStoreMemory tests
# ---------------------------------------------------------------------------


class TestVoiceSessionStoreMemory:
    """Tests for the in-memory session store."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_create_and_get(self) -> None:
        store = VoiceSessionStoreMemory()
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        self._run(store.create(session))
        fetched = self._run(store.get(session.room_name))
        assert fetched is not None
        assert fetched.room_name == session.room_name

    def test_get_nonexistent(self) -> None:
        store = VoiceSessionStoreMemory()
        result = self._run(store.get("nonexistent"))
        assert result is None

    def test_end_session(self) -> None:
        store = VoiceSessionStoreMemory()
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)

        self._run(store.create(session))
        self._run(store.end(session.room_name))
        fetched = self._run(store.get(session.room_name))
        assert fetched is not None
        assert fetched.status == "ended"

    def test_list_active(self) -> None:
        store = VoiceSessionStoreMemory()
        manager = VoiceSessionManager()
        s1 = manager.create_session(consultation_id=1, doctor_id=1)
        s2 = manager.create_session(consultation_id=2, doctor_id=2)

        self._run(store.create(s1))
        self._run(store.create(s2))
        self._run(store.end(s2.room_name))

        active = self._run(store.list_active())
        assert len(active) == 1
        assert active[0].room_name == s1.room_name

    def test_expired_session_marked(self) -> None:
        store = VoiceSessionStoreMemory()
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=1)
        session.expires_at = time.time() - 1

        self._run(store.create(session))
        fetched = self._run(store.get(session.room_name))
        assert fetched is not None
        assert fetched.status == "expired"


class TestCreateVoiceStore:
    """Tests for the store factory."""

    def test_default_returns_memory(self) -> None:
        store = create_voice_store()
        assert isinstance(store, VoiceSessionStoreMemory)

    def test_with_redis_url_returns_redis(self) -> None:
        from app.voice.session_store import VoiceSessionStoreRedis
        store = create_voice_store("redis://localhost:6379/9")
        assert isinstance(store, VoiceSessionStoreRedis)


# ---------------------------------------------------------------------------
# VoiceWorker tests
# ---------------------------------------------------------------------------


class TestVoiceWorker:
    """Tests for VoiceWorker."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_worker_start_stop(self) -> None:
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        assert worker.is_running is False
        self._run(worker.start())
        assert worker.is_running is True
        self._run(worker.stop())
        assert worker.is_running is False

    def test_worker_rejects_when_not_running(self) -> None:
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        result = self._run(worker.handle_final_transcript(
            text="hello",
            speaker="doctor",
            room_sid="r1",
            participant_sid="p1",
            turn_id="t1",
        ))
        assert result["accepted"] is False
        assert result["reason"] == "worker_not_running"

    def test_worker_dedup_in_memory(self) -> None:
        """Duplicate final transcript only accepted once (in-memory dedup)."""
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        self._run(worker.start())

        r1 = self._run(worker.handle_final_transcript(
            text="hello",
            speaker="doctor",
            room_sid="r1",
            participant_sid="p1",
            turn_id="t1",
        ))
        r2 = self._run(worker.handle_final_transcript(
            text="hello",
            speaker="doctor",
            room_sid="r1",
            participant_sid="p1",
            turn_id="t1",
        ))
        assert r1["accepted"] is True
        assert r2["accepted"] is False
        assert r2["reason"] == "duplicate"

        self._run(worker.stop())

    def test_worker_calls_message_handler(self) -> None:
        """Worker calls message_handler for doctor transcripts."""
        handler = AsyncMock(return_value="我头疼")
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
            message_handler=handler,
        )
        self._run(worker.start())

        result = self._run(worker.handle_final_transcript(
            text="你好",
            speaker="doctor",
            room_sid="r1",
            participant_sid="p1",
            turn_id="t1",
        ))
        assert result["accepted"] is True
        assert result["patient_response"] == "我头疼"
        handler.assert_called_once_with(1, "你好")

        self._run(worker.stop())

    def test_worker_partial_transcript_buffer(self) -> None:
        """Partial transcripts are buffered in memory."""
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        self._run(worker.start())

        worker.buffer_partial_transcript(text="你", speaker="doctor")
        worker.buffer_partial_transcript(text="你好", speaker="doctor")

        assert len(worker._partial_cache) == 2
        assert worker._partial_cache[0]["text"] == "你"
        assert worker._partial_cache[1]["text"] == "你好"

        self._run(worker.stop())

    def test_worker_partial_cache_bounded(self) -> None:
        """Partial cache evicts oldest when full."""
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        # Override max for testing
        worker._PARTIAL_CACHE_MAX = 5

        for i in range(10):
            worker.buffer_partial_transcript(text=f"msg-{i}", speaker="doctor")

        assert len(worker._partial_cache) <= 5

    def test_worker_cleanup_on_stop(self) -> None:
        """Stop clears partial cache and dedup state."""
        worker = VoiceWorker(
            room_name="test-room",
            consultation_id=1,
            doctor_id=1,
        )
        self._run(worker.start())

        worker.buffer_partial_transcript(text="hello", speaker="doctor")
        worker._seen_turns.add("some-key")

        self._run(worker.stop())

        assert worker.is_running is False
        assert worker._task is None
