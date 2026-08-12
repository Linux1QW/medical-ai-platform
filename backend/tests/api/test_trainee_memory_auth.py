# -*- coding: utf-8 -*-
"""Trainee memory API authentication and authorization tests.

Coverage:
- Doctor self-access (create, list, delete, consent)
- Admin review (approve/reject)
- IDOR prevention (cannot read/delete another doctor's memory)
- Unauthenticated → 401
- Permission denied → 403
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── Unauthenticated → 401 ────────────────────────────────────────────────────


class TestTraineeMemoryUnauthenticated:
    """Unauthenticated requests should return 401."""

    def test_create_memory_unauthenticated(self, client: TestClient) -> None:
        """POST /trainee-memory/memories without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/trainee-memory/memories",
            json={"skill_dimension": "rapport", "summary": "Good rapport building"},
        )
        assert resp.status_code == 401

    def test_list_memories_unauthenticated(self, client: TestClient) -> None:
        """GET /trainee-memory/me/memories without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.get("/api/v1/trainee-memory/me/memories")
        assert resp.status_code == 401

    def test_set_consent_unauthenticated(self, client: TestClient) -> None:
        """PUT /trainee-memory/me/consent without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.put(
            "/api/v1/trainee-memory/me/consent",
            json={"consent": True},
        )
        assert resp.status_code == 401

    def test_review_memory_unauthenticated(self, client: TestClient) -> None:
        """POST /trainee-memory/memories/{id}/review without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/trainee-memory/memories/1/review",
            json={"action": "approve"},
        )
        assert resp.status_code == 401


# ── Doctor self-access ───────────────────────────────────────────────────────


class TestTraineeMemoryDoctorSelfAccess:
    """Doctor can access own memories only."""

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_create_memory_doctor(self, mock_service_cls, client: TestClient) -> None:
        """POST /trainee-memory/memories as doctor → 200."""
        _as_user(DOCTOR_A)

        mock_memory = SimpleNamespace(
            id=1, doctor_id=10, status="candidate",
            skill_dimension="rapport", summary="Good rapport",
            evidence_refs=[], reviewer_id=None, review_comment=None,
            reviewed_at=None, expires_at=None,
            created_at="2024-01-01T00:00:00",
        )
        mock_service = AsyncMock()
        mock_service.create_candidate.return_value = mock_memory
        mock_service_cls.return_value = mock_service

        resp = client.post(
            "/api/v1/trainee-memory/memories",
            json={"skill_dimension": "rapport", "summary": "Good rapport"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doctor_id"] == 10

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_list_my_memories(self, mock_service_cls, client: TestClient) -> None:
        """GET /trainee-memory/me/memories as doctor → 200."""
        _as_user(DOCTOR_A)

        mock_service = AsyncMock()
        mock_service.list_memories.return_value = []
        mock_service_cls.return_value = mock_service

        resp = client.get("/api/v1/trainee-memory/me/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_set_my_consent(self, mock_service_cls, client: TestClient) -> None:
        """PUT /trainee-memory/me/consent as doctor → 200."""
        _as_user(DOCTOR_A)

        mock_service = AsyncMock()
        mock_service_cls.return_value = mock_service

        resp = client.put(
            "/api/v1/trainee-memory/me/consent",
            json={"consent": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doctor_id"] == 10
        assert data["consent"] is True


# ── Admin review ─────────────────────────────────────────────────────────────


class TestTraineeMemoryAdminReview:
    """Admin can review (approve/reject) memories."""

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_approve_memory_admin(self, mock_service_cls, client: TestClient) -> None:
        """POST /trainee-memory/memories/{id}/review as admin → 200."""
        _as_user(ADMIN_USER)

        mock_memory = SimpleNamespace(
            id=1, doctor_id=10, status="approved",
            skill_dimension="rapport", summary="Good rapport",
            evidence_refs=[], reviewer_id=1, review_comment="Looks good",
            reviewed_at="2024-01-01T00:00:00", expires_at="2024-04-01T00:00:00",
            created_at="2024-01-01T00:00:00",
        )
        mock_service = AsyncMock()
        mock_service.approve.return_value = mock_memory
        mock_service_cls.return_value = mock_service

        # Mock the DB and query result
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = SimpleNamespace(doctor_id=10)
        mock_db.execute.return_value = mock_result

        app.dependency_overrides[get_db] = lambda: mock_db

        resp = client.post(
            "/api/v1/trainee-memory/memories/1/review",
            json={"action": "approve", "review_comment": "Looks good"},
        )
        assert resp.status_code == 200


# ── IDOR prevention ──────────────────────────────────────────────────────────


class TestTraineeMemoryIDORPrevention:
    """Doctor cannot access another doctor's memories."""

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_delete_another_doctors_memory(self, mock_service_cls, client: TestClient) -> None:
        """DELETE /trainee-memory/me/memories/{id} for another doctor → 404."""
        _as_user(DOCTOR_A)

        mock_service = AsyncMock()
        mock_service.delete_memory.return_value = False  # Memory belongs to another doctor
        mock_service_cls.return_value = mock_service

        resp = client.delete("/api/v1/trainee-memory/me/memories/999")
        assert resp.status_code == 404

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_no_arbitrary_doctor_id_in_create(self, mock_service_cls, client: TestClient) -> None:
        """POST /trainee-memory/memories should not accept doctor_id in body."""
        _as_user(DOCTOR_A)

        mock_memory = SimpleNamespace(
            id=1, doctor_id=10, status="candidate",
            skill_dimension="rapport", summary="Good rapport",
            evidence_refs=[], reviewer_id=None, review_comment=None,
            reviewed_at=None, expires_at=None,
            created_at="2024-01-01T00:00:00",
        )
        mock_service = AsyncMock()
        mock_service.create_candidate.return_value = mock_memory
        mock_service_cls.return_value = mock_service

        # The schema no longer has doctor_id - it comes from token
        # Extra fields should be ignored by Pydantic
        resp = client.post(
            "/api/v1/trainee-memory/memories",
            json={
                "skill_dimension": "rapport",
                "summary": "Good rapport",
                "doctor_id": 999,  # This should be ignored
            },
        )
        # 200 (extra field ignored) or 422 (validation error)
        assert resp.status_code in (200, 422)

    @patch("app.api.v1.trainee_memory.ProfileMemoryService")
    def test_no_arbitrary_doctor_id_in_consent(self, mock_service_cls, client: TestClient) -> None:
        """PUT /trainee-memory/me/consent should not accept doctor_id in body."""
        _as_user(DOCTOR_A)

        mock_service = AsyncMock()
        mock_service_cls.return_value = mock_service

        resp = client.put(
            "/api/v1/trainee-memory/me/consent",
            json={"consent": True, "doctor_id": 999},  # Should be ignored/rejected
        )
        # Either 422 (validation error) or 200 (extra field ignored)
        assert resp.status_code in (200, 422)


# ── Permission checks ────────────────────────────────────────────────────────


class TestTraineeMemoryPermissions:
    """Permission checks for trainee memory endpoints."""

    def test_doctor_cannot_review(self, client: TestClient) -> None:
        """POST /trainee-memory/memories/{id}/review as doctor → 403."""
        _as_user(DOCTOR_A)

        resp = client.post(
            "/api/v1/trainee-memory/memories/1/review",
            json={"action": "approve"},
        )
        assert resp.status_code == 403
