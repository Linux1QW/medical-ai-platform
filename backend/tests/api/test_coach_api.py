"""Tests for Coach API endpoints.

Coverage:
- Unauthenticated → 401
- IDOR: doctor cannot access another doctor's consultation → 403
- Disabled Coach returns 409
- Idempotent suggestion creation
- Permission checks (coach:use, coach:trace:view)
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.main import app

# ── Helpers ──────────────────────────────────────────────────────────────────

DOCTOR_A = SimpleNamespace(id=10, username="doc_a", role="doctor", permissions=None)
DOCTOR_B = SimpleNamespace(id=20, username="doc_b", role="doctor", permissions=None)
ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin", permissions=None)
NO_COACH_USER = SimpleNamespace(id=30, username="no_coach", role="doctor", permissions=["evaluation:view"])


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


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events = []
    current_event: dict = {}
    current_data: str = ""

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[len("event: "):]
        elif line.startswith("id: "):
            current_event["id"] = line[len("id: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event.get("event"):
            if current_data:
                try:
                    current_event["data"] = json.loads(current_data)
                except json.JSONDecodeError:
                    current_event["data"] = current_data
            else:
                current_event["data"] = {}
            events.append(current_event)
            current_event = {}
            current_data = ""

    return events


# ── Unauthenticated → 401 ────────────────────────────────────────────────────


class TestCoachUnauthenticated:
    """Unauthenticated requests should return 401."""

    def test_coach_stream_requires_auth(self, client: TestClient) -> None:
        """POST stream without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={"latest_message": "你好", "idempotency_key": "test-key-0000001"},
        )
        assert resp.status_code == 401

    def test_coach_state_requires_auth(self, client: TestClient) -> None:
        """GET state without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.get("/api/v1/coach/consultations/1/state")
        assert resp.status_code == 401

    def test_coach_feedback_requires_auth(self, client: TestClient) -> None:
        """POST feedback without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            f"/api/v1/coach/suggestions/{uuid4()}/feedback",
            json={"feedback": "accepted"},
        )
        assert resp.status_code == 401

    def test_coach_trace_requires_auth(self, client: TestClient) -> None:
        """GET trace without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.get("/api/v1/coach/admin/consultations/1/trace")
        assert resp.status_code == 401


# ── Permission denied → 403 ─────────────────────────────────────────────────


class TestCoachPermissionDenied:
    """Users without coach:use should get 403."""

    def test_no_coach_permission_stream(self, client: TestClient) -> None:
        """Doctor without coach:use → 403."""
        _as_user(NO_COACH_USER)
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={"latest_message": "你好", "idempotency_key": "test-key-0000002"},
        )
        assert resp.status_code == 403

    def test_no_coach_trace_permission(self, client: TestClient) -> None:
        """Doctor without coach:trace:view → 403 on trace endpoint."""
        _as_user(DOCTOR_A)
        resp = client.get("/api/v1/coach/admin/consultations/1/trace")
        assert resp.status_code == 403


# ── IDOR: doctor cannot access another's consultation ────────────────────────


class TestCoachIDOR:
    """Doctor cannot access another doctor's consultation."""

    def test_doctor_cannot_access_other_consultation(self, client: TestClient) -> None:
        """Doctor A tries to stream for Doctor B's consultation → 403."""
        _as_user(DOCTOR_A)

        mock_service = MagicMock()
        mock_service.enabled = True

        from fastapi import HTTPException

        async def fake_verify(consultation_id, doctor_id, db):
            if doctor_id != 20:  # DOCTOR_A has id=10
                raise HTTPException(status_code=403, detail={"error_code": "COACH_IDOR_DENIED", "message": "IDOR denied"})
            return SimpleNamespace(id=100, doctor_id=20)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):
            resp = client.post(
                "/api/v1/coach/consultations/100/suggestions/stream",
                json={"latest_message": "你好", "idempotency_key": "test-key-0000003"},
            )
            assert resp.status_code == 403

    def test_doctor_cannot_view_other_state(self, client: TestClient) -> None:
        """Doctor A tries to get state for Doctor B's consultation → 403."""
        _as_user(DOCTOR_A)

        mock_service = MagicMock()
        mock_service.enabled = True

        from fastapi import HTTPException

        async def fake_verify(consultation_id, doctor_id, db):
            if doctor_id != 20:
                raise HTTPException(status_code=403, detail={"error_code": "COACH_IDOR_DENIED", "message": "IDOR denied"})
            return SimpleNamespace(id=100, doctor_id=20)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):
            resp = client.get("/api/v1/coach/consultations/100/state")
            assert resp.status_code == 403


# ── Disabled Coach → 409 ────────────────────────────────────────────────────


class TestCoachDisabled:
    """When COACH_ENABLED=false, stream returns 409."""

    def test_disabled_coach_returns_409(self, client: TestClient) -> None:
        """Stream endpoint returns 409 when Coach is disabled."""
        _as_user(DOCTOR_A)

        mock_service = MagicMock()
        mock_service.enabled = False

        async def fake_verify(consultation_id, doctor_id, db):
            return SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):
            resp = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "你好", "idempotency_key": "test-key-0000004"},
            )
            assert resp.status_code == 409
            data = resp.json()
            assert data["error_code"] == "FEATURE_DISABLED"


# ── Idempotent suggestion ───────────────────────────────────────────────────


class TestCoachIdempotent:
    """Same idempotency_key → same decision."""

    def test_idempotent_suggestion_creation(self, client: TestClient) -> None:
        """Two requests with same key produce one decision."""
        _as_user(DOCTOR_A)

        mock_service = MagicMock()
        mock_service.enabled = True

        async def fake_stream(**kwargs):
            yield {"event": "thinking", "data": {"turn_no": 1}, "id": str(uuid4())}
            yield {
                "event": "suggestion",
                "data": {"suggestion_id": str(uuid4()), "content": "test"},
                "id": str(uuid4()),
            }
            yield {"event": "done", "data": {}, "id": None}

        mock_service.stream_suggestion = fake_stream

        async def fake_verify(consultation_id, doctor_id, db):
            return SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):
            payload = {
                "latest_message": "你好",
                "idempotency_key": "idem-key-00000000001",
            }
            resp = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json=payload,
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = _parse_sse_events(resp.text)
            event_names = [e["event"] for e in events]
            assert "thinking" in event_names
            assert "suggestion" in event_names
            assert "done" in event_names


# ── Schema validation ───────────────────────────────────────────────────────


class TestCoachSchemaValidation:
    """Request body validation."""

    def test_idempotency_key_too_short(self, client: TestClient) -> None:
        """idempotency_key < 16 chars → 422."""
        _as_user(DOCTOR_A)
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={"latest_message": "你好", "idempotency_key": "short"},
        )
        assert resp.status_code == 422

    def test_idempotency_key_invalid_chars(self, client: TestClient) -> None:
        """idempotency_key with invalid chars → 422."""
        _as_user(DOCTOR_A)
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={"latest_message": "你好", "idempotency_key": "key with spaces!!!"},
        )
        assert resp.status_code == 422

    def test_extra_fields_forbidden(self, client: TestClient) -> None:
        """Extra body fields → 422."""
        _as_user(DOCTOR_A)
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={
                "latest_message": "你好",
                "idempotency_key": "valid-key-00000001",
                "doctor_id": 999,  # forbidden
            },
        )
        assert resp.status_code == 422

    def test_empty_message_forbidden(self, client: TestClient) -> None:
        """Empty latest_message → 422."""
        _as_user(DOCTOR_A)
        resp = client.post(
            "/api/v1/coach/consultations/1/suggestions/stream",
            json={"latest_message": "", "idempotency_key": "valid-key-00000002"},
        )
        assert resp.status_code == 422
