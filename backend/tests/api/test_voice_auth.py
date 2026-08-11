# -*- coding: utf-8 -*-
"""Voice API authentication and authorization tests.

Coverage:
- VOICE_ENABLED=false → 503
- Unauthenticated → 401
- Non-owner → 403
- Owner → 200 (when VOICE_ENABLED=true and LiveKit configured)
- Config missing → 503
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.main import app

# ── Helpers ──────────────────────────────────────────────────────────────────

DOCTOR_A = SimpleNamespace(id=10, username="doc_a", role="doctor", permissions=None)
DOCTOR_B = SimpleNamespace(id=20, username="doc_b", role="doctor", permissions=None)
ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin", permissions=None)


def _override_get_db():
    yield None


@pytest.fixture
def client():
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()
    app.dependency_overrides.clear()


def _as_user(user):
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = lambda: user


# ── VOICE_ENABLED=false → 503 ────────────────────────────────────────────────


class TestVoiceDisabled:
    """When VOICE_ENABLED is false, all endpoints return 503."""

    def test_create_session_voice_disabled(self, client: TestClient) -> None:
        """POST /voice/sessions with VOICE_ENABLED=false → 503."""
        _as_user(DOCTOR_A)
        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = False
            resp = client.post(
                "/api/v1/voice/sessions",
                json={"consultation_id": 1},
                headers={"Authorization": "Bearer dummy"},
            )
            assert resp.status_code == 503
            body = resp.json()
            assert body["error_code"] == "VOICE_DISABLED"

    def test_get_state_voice_disabled(self, client: TestClient) -> None:
        """GET /voice/sessions/{room_name}/state with VOICE_ENABLED=false → 503."""
        _as_user(DOCTOR_A)
        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = False
            resp = client.get(
                "/api/v1/voice/sessions/test-room/state",
                headers={"Authorization": "Bearer dummy"},
            )
            assert resp.status_code == 503

    def test_end_session_voice_disabled(self, client: TestClient) -> None:
        """POST /voice/sessions/{room_name}/end with VOICE_ENABLED=false → 503."""
        _as_user(DOCTOR_A)
        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = False
            resp = client.post(
                "/api/v1/voice/sessions/test-room/end",
                headers={"Authorization": "Bearer dummy"},
            )
            assert resp.status_code == 503


# ── Unauthenticated → 401 ────────────────────────────────────────────────────


class TestVoiceUnauthenticated:
    """Unauthenticated requests should return 401."""

    def test_create_session_unauthenticated(self, client: TestClient) -> None:
        """POST /voice/sessions without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/voice/sessions",
            json={"consultation_id": 1},
        )
        assert resp.status_code == 401

    def test_get_state_unauthenticated(self, client: TestClient) -> None:
        """GET /voice/sessions/{room_name}/state without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.get("/api/v1/voice/sessions/test-room/state")
        assert resp.status_code == 401

    def test_end_session_unauthenticated(self, client: TestClient) -> None:
        """POST /voice/sessions/{room_name}/end without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post("/api/v1/voice/sessions/test-room/end")
        assert resp.status_code == 401


# ── Non-owner → 403 ──────────────────────────────────────────────────────────


class TestVoiceNonOwner:
    """Non-owner access should return 403."""

    def test_create_session_non_owner(self, client: TestClient) -> None:
        """POST /voice/sessions with non-owner → 403."""
        _as_user(DOCTOR_B)

        from fastapi import HTTPException

        async def raise_403(*args, **kwargs):
            raise HTTPException(status_code=403, detail="FORBIDDEN")

        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = True
            with patch("app.api.v1.voice.require_consultation_access", side_effect=raise_403):
                resp = client.post(
                    "/api/v1/voice/sessions",
                    json={"consultation_id": 1},
                    headers={"Authorization": "Bearer dummy"},
                )
                assert resp.status_code == 403

    def test_get_state_non_owner(self, client: TestClient) -> None:
        """GET /voice/sessions/{room_name}/state with non-owner → 403."""
        _as_user(DOCTOR_B)

        # Create a session in the store
        from app.api.v1.voice import get_store
        import asyncio

        store = get_store()
        from app.voice.session import VoiceSessionManager
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=10)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(store.create(session))
        finally:
            loop.close()

        from fastapi import HTTPException

        async def raise_403(*args, **kwargs):
            raise HTTPException(status_code=403, detail="FORBIDDEN")

        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = True
            with patch("app.api.v1.voice.require_consultation_access", side_effect=raise_403):
                resp = client.get(
                    f"/api/v1/voice/sessions/{session.room_name}/state",
                    headers={"Authorization": "Bearer dummy"},
                )
                assert resp.status_code == 403

    def test_end_session_non_owner(self, client: TestClient) -> None:
        """POST /voice/sessions/{room_name}/end with non-owner → 403."""
        _as_user(DOCTOR_B)

        from app.api.v1.voice import get_store
        import asyncio

        store = get_store()
        from app.voice.session import VoiceSessionManager
        manager = VoiceSessionManager()
        session = manager.create_session(consultation_id=1, doctor_id=10)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(store.create(session))
        finally:
            loop.close()

        from fastapi import HTTPException

        async def raise_403(*args, **kwargs):
            raise HTTPException(status_code=403, detail="FORBIDDEN")

        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = True
            with patch("app.api.v1.voice.require_consultation_access", side_effect=raise_403):
                resp = client.post(
                    f"/api/v1/voice/sessions/{session.room_name}/end",
                    headers={"Authorization": "Bearer dummy"},
                )
                assert resp.status_code == 403


# ── Config missing → 503 ────────────────────────────────────────────────────


class TestVoiceConfigMissing:
    """When LiveKit config is missing, create returns 503."""

    def test_create_session_config_missing(self, client: TestClient) -> None:
        """POST /voice/sessions with no LiveKit config → 503."""
        _as_user(DOCTOR_A)

        from unittest.mock import MagicMock

        mock_consultation = SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.voice.settings") as mock_settings:
            mock_settings.VOICE_ENABLED = True
            mock_settings.LIVEKIT_API_KEY = None
            mock_settings.LIVEKIT_API_SECRET = None

            async def mock_access(*args, **kwargs):
                return mock_consultation

            with patch("app.api.v1.voice.require_consultation_access", side_effect=mock_access):
                resp = client.post(
                    "/api/v1/voice/sessions",
                    json={"consultation_id": 1},
                    headers={"Authorization": "Bearer dummy"},
                )
                assert resp.status_code == 503
                body = resp.json()
                assert body["error_code"] == "VOICE_NOT_CONFIGURED"
