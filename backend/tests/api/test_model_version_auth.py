"""模型版本端点鉴权测试 — 确认 config_json 不匿名可读"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_admin, get_current_user
from app.db.session import get_db
from app.main import app

ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin", permissions=None)
DOCTOR_USER = SimpleNamespace(id=2, username="doctor", role="doctor", permissions=None)


def _override_get_db():
    yield None


@pytest.fixture
def client():
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()
    app.dependency_overrides.clear()


def _as_admin():
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_admin] = lambda: ADMIN_USER
    app.dependency_overrides[get_current_user] = lambda: ADMIN_USER


def _as_doctor():
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = lambda: DOCTOR_USER


class TestModelVersionListAuth:
    """GET /model-versions/"""

    def test_unauthenticated_returns_401(self, client):
        resp = client.get("/api/v1/model-versions/")
        assert resp.status_code == 401

    def test_doctor_returns_403(self, client):
        _as_doctor()
        resp = client.get("/api/v1/model-versions/")
        assert resp.status_code == 403

    def test_admin_returns_200(self, client):
        _as_admin()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = client.get("/api/v1/model-versions/")
        assert resp.status_code == 200


class TestModelVersionActiveAuth:
    """GET /model-versions/{name}/active"""

    def test_unauthenticated_returns_401(self, client):
        resp = client.get("/api/v1/model-versions/qwen/active")
        assert resp.status_code == 401

    def test_doctor_returns_403(self, client):
        _as_doctor()
        resp = client.get("/api/v1/model-versions/qwen/active")
        assert resp.status_code == 403

    def test_admin_returns_200_or_404(self, client):
        """admin 能访问端点（200 或 404 取决于数据，但不是 401/403）"""
        _as_admin()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = client.get("/api/v1/model-versions/qwen/active")
        assert resp.status_code in (200, 404)
