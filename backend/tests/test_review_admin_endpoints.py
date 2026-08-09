# -*- coding: utf-8 -*-
"""P0 安全修复回归测试 — 复核/管理端点鉴权、/metrics 保护、/health 连通性检查

Task 8: 更新为适配新的原子复核 API（evaluation_id 为 int，新 schema）。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.api.v1.review as review_module
from app.core.config import settings
from app.core.deps import get_current_admin, get_current_user
from app.db.session import get_db
from app.main import app

ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin")
DOCTOR_USER = SimpleNamespace(id=2, username="doctor", role="doctor")


def _override_get_db():
    yield None


@pytest.fixture
def client():
    # 不进入上下文管理器，避免触发 lifespan（LangGraph checkpointer 等真实依赖）
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


# ── 复核端点鉴权 ─────────────────────────────────────────────────────────────


class TestReviewEndpointAuth:
    def test_pending_requires_auth(self, client):
        resp = client.get("/api/v1/reviews/pending")
        assert resp.status_code == 401

    def test_pending_forbidden_for_doctor(self, client):
        _as_doctor()
        resp = client.get("/api/v1/reviews/pending")
        assert resp.status_code == 403
        assert resp.json()["error_code"] == "AUTH_FORBIDDEN"

    def test_pending_ok_for_admin(self, client):
        _as_admin()
        with (
            patch.object(
                review_module, "list_pending_evaluations", new=AsyncMock(return_value=[])
            ),
            patch.object(
                review_module, "count_pending_evaluations", new=AsyncMock(return_value=0)
            ),
        ):
            resp = client.get("/api/v1/reviews/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pending_reviews"] == []
        assert body["total"] == 0

    def test_submit_requires_auth(self, client):
        resp = client.post(
            "/api/v1/reviews/1/submit", json={"feedback": "ok enough"}
        )
        assert resp.status_code == 401

    def test_status_requires_auth(self, client):
        resp = client.get("/api/v1/reviews/1/status")
        assert resp.status_code == 401

    def test_submit_not_found_returns_404(self, client):
        """evaluation 不存在时返回 404"""
        _as_admin()
        with patch.object(
            review_module, "svc_submit_review", new=AsyncMock(return_value=None)
        ):
            resp = client.post(
                "/api/v1/reviews/999/submit", json={"feedback": "测试不存在"}
            )
        assert resp.status_code == 404

    def test_submit_conflict_returns_409(self, client):
        """状态冲突时返回 409"""
        _as_admin()
        from app.services.review_service import ReviewConflictError

        with patch.object(
            review_module,
            "svc_submit_review",
            new=AsyncMock(side_effect=ReviewConflictError("Already reviewed")),
        ):
            resp = client.post(
                "/api/v1/reviews/1/submit", json={"feedback": "重复提交测试"}
            )
        assert resp.status_code == 409

    def test_submit_feedback_too_short_rejected(self, client):
        """feedback 太短被拒绝"""
        _as_admin()
        resp = client.post(
            "/api/v1/reviews/1/submit", json={"feedback": "x"}
        )
        assert resp.status_code == 422

    def test_submit_feedback_too_long_rejected(self, client):
        _as_admin()
        resp = client.post(
            "/api/v1/reviews/1/submit", json={"feedback": "x" * 5001}
        )
        assert resp.status_code == 422

    def test_submit_no_reviewer_id_in_body(self, client):
        """请求体不包含 reviewer_id 字段"""
        _as_admin()
        from uuid import uuid4
        from datetime import datetime

        mock_result = {
            "review_id": uuid4(),
            "evaluation_id": 1,
            "run_id": uuid4(),
            "status": "reviewed",
            "reviewed_at": datetime.now(),
        }
        with patch.object(
            review_module, "svc_submit_review", new=AsyncMock(return_value=mock_result)
        ):
            resp = client.post(
                "/api/v1/reviews/1/submit",
                json={"feedback": "确认无误"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "reviewed"


# ── 管理缓存端点鉴权 ─────────────────────────────────────────────────────────


class TestAdminCacheEndpointAuth:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/v1/admin/cache/retrieval/clear"),
            ("get", "/api/v1/admin/cache/retrieval/stats"),
            ("get", "/api/v1/admin/cache-stats"),
        ],
    )
    def test_requires_auth(self, client, method, path):
        resp = getattr(client, method)(path)
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/v1/admin/cache/retrieval/clear"),
            ("get", "/api/v1/admin/cache/retrieval/stats"),
            ("get", "/api/v1/admin/cache-stats"),
        ],
    )
    def test_forbidden_for_doctor(self, client, method, path):
        _as_doctor()
        resp = getattr(client, method)(path)
        assert resp.status_code == 403


# ── /metrics 端点保护 ────────────────────────────────────────────────────────


class TestMetricsProtection:
    def test_metrics_open_when_no_token_in_dev(self, client, monkeypatch):
        monkeypatch.setattr(settings, "METRICS_TOKEN", "")
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_forbidden_without_token_in_production(self, client, monkeypatch):
        monkeypatch.setattr(settings, "METRICS_TOKEN", "")
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        resp = client.get("/metrics")
        assert resp.status_code == 403

    def test_metrics_requires_bearer_when_token_set(self, client, monkeypatch):
        monkeypatch.setattr(settings, "METRICS_TOKEN", "sekret")
        assert client.get("/metrics").status_code == 403
        assert (
            client.get(
                "/metrics", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 403
        )
        resp = client.get("/metrics", headers={"Authorization": "Bearer sekret"})
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ── /health 依赖连通性检查 ───────────────────────────────────────────────────


class _AsyncCM:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *args):
        return False


class _FakeConn:
    async def execute(self, *_args, **_kwargs):
        return None


class _FakeEngineOK:
    def connect(self):
        return _AsyncCM(_FakeConn())


class _FakeEngineDown:
    def connect(self):
        raise RuntimeError("mysql down")


def _health_patches(engine, redis_client):
    """统一 patch /health 的外部依赖，避免真实连接"""
    return (
        patch("app.main.engine", engine),
        patch("app.main._get_cache_redis", new=AsyncMock(return_value=redis_client)),
        patch("app.main.get_checkpointer", new=MagicMock(return_value=None)),
    )


class TestHealthCheck:
    def test_health_ok_when_dependencies_up(self, client):
        redis_client = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)
        p1, p2, p3 = _health_patches(_FakeEngineOK(), redis_client)
        with p1, p2, p3:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"] == {"mysql": "ok", "redis": "ok"}

    def test_health_degraded_when_mysql_down(self, client):
        redis_client = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)
        p1, p2, p3 = _health_patches(_FakeEngineDown(), redis_client)
        with p1, p2, p3:
            resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["mysql"] == "unavailable"

    def test_health_degraded_when_redis_down(self, client):
        p1, p2, p3 = _health_patches(_FakeEngineOK(), None)
        with p1, p2, p3:
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["checks"]["redis"] == "unavailable"
