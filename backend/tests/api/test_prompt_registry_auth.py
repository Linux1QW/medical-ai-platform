# -*- coding: utf-8 -*-
"""Prompt registry API authentication and authorization tests.

Coverage:
- Unauthenticated → 401
- Permission checks (prompt:manage, experiment:manage)
- Immutable bundles (active bundles cannot be edited)
- Doctor cannot manage prompts
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


class TestPromptRegistryUnauthenticated:
    """Unauthenticated requests should return 401."""

    def test_register_bundle_unauthenticated(self, client: TestClient) -> None:
        """POST /prompt-registry/bundles without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/prompt-registry/bundles",
            json={
                "name": "test-bundle",
                "version": "1.0.0",
                "system_prompt": "You are a helpful assistant",
                "source_commit": "abc1234",
            },
        )
        assert resp.status_code == 401

    def test_get_bundle_unauthenticated(self, client: TestClient) -> None:
        """GET /prompt-registry/bundles/{name}/{version} without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.get("/api/v1/prompt-registry/bundles/test/1.0.0")
        assert resp.status_code == 401

    def test_create_experiment_unauthenticated(self, client: TestClient) -> None:
        """POST /prompt-registry/experiments without auth → 401."""
        app.dependency_overrides.clear()
        resp = client.post(
            "/api/v1/prompt-registry/experiments",
            json={
                "name": "test-exp",
                "baseline_bundle_name": "baseline",
                "treatment_bundle_name": "treatment",
            },
        )
        assert resp.status_code == 401


# ── Permission checks ────────────────────────────────────────────────────────


class TestPromptRegistryPermissions:
    """Permission checks for prompt registry endpoints."""

    def test_doctor_cannot_register_bundle(self, client: TestClient) -> None:
        """POST /prompt-registry/bundles as doctor → 403."""
        _as_user(DOCTOR_A)

        resp = client.post(
            "/api/v1/prompt-registry/bundles",
            json={
                "name": "test-bundle",
                "version": "1.0.0",
                "system_prompt": "You are a helpful assistant",
                "source_commit": "abc1234",
            },
        )
        assert resp.status_code == 403

    def test_doctor_cannot_create_experiment(self, client: TestClient) -> None:
        """POST /prompt-registry/experiments as doctor → 403."""
        _as_user(DOCTOR_A)

        resp = client.post(
            "/api/v1/prompt-registry/experiments",
            json={
                "name": "test-exp",
                "baseline_bundle_name": "baseline",
                "treatment_bundle_name": "treatment",
            },
        )
        assert resp.status_code == 403

    def test_doctor_cannot_list_bundles(self, client: TestClient) -> None:
        """GET /prompt-registry/bundles as doctor → 403."""
        _as_user(DOCTOR_A)

        resp = client.get("/api/v1/prompt-registry/bundles")
        assert resp.status_code == 403


# ── Admin access ─────────────────────────────────────────────────────────────


class TestPromptRegistryAdminAccess:
    """Admin can manage prompts and experiments."""

    @patch("app.api.v1.prompt_registry.PromptRegistry")
    def test_admin_register_bundle(self, mock_registry_cls, client: TestClient) -> None:
        """POST /prompt-registry/bundles as admin → 201."""
        _as_user(ADMIN_USER)

        mock_bundle = SimpleNamespace(
            id=1, name="test-bundle", version="1.0.0",
            status="draft", content_hash="abc123",
            source_commit="abc1234", author="admin",
        )
        mock_registry = AsyncMock()
        mock_registry.register_bundle.return_value = mock_bundle
        mock_registry_cls.return_value = mock_registry

        resp = client.post(
            "/api/v1/prompt-registry/bundles",
            json={
                "name": "test-bundle",
                "version": "1.0.0",
                "system_prompt": "You are a helpful assistant",
                "source_commit": "abc1234",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-bundle"
        assert data["author"] == "admin"

    @patch("app.api.v1.prompt_registry.PromptRegistry")
    def test_admin_list_bundles(self, mock_registry_cls, client: TestClient) -> None:
        """GET /prompt-registry/bundles as admin → 200."""
        _as_user(ADMIN_USER)

        mock_registry = AsyncMock()
        mock_registry.list_bundles.return_value = []
        mock_registry_cls.return_value = mock_registry

        resp = client.get("/api/v1/prompt-registry/bundles")
        assert resp.status_code == 200
        assert resp.json() == []


# ── Immutable bundles ────────────────────────────────────────────────────────


class TestPromptRegistryImmutableBundles:
    """Active bundles cannot be edited."""

    @patch("app.api.v1.prompt_registry.PromptRegistry")
    def test_register_duplicate_bundle(self, mock_registry_cls, client: TestClient) -> None:
        """POST /prompt-registry/bundles with duplicate name/version → 409."""
        _as_user(ADMIN_USER)

        mock_registry = AsyncMock()
        mock_registry.register_bundle.side_effect = ValueError("Bundle already exists")
        mock_registry_cls.return_value = mock_registry

        resp = client.post(
            "/api/v1/prompt-registry/bundles",
            json={
                "name": "test-bundle",
                "version": "1.0.0",
                "system_prompt": "You are a helpful assistant",
                "source_commit": "abc1234",
            },
        )
        assert resp.status_code == 409
