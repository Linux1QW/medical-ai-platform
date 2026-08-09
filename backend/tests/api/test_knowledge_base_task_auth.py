# -*- coding: utf-8 -*-
"""知识库管理端点鉴权测试 — knowledge:manage 权限矩阵 + Celery FAILURE 安全输出"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── 端点矩阵 ─────────────────────────────────────────────────────────────────

KB_ENDPOINTS = [
    ("post", "/api/v1/knowledge-base/rebuild"),
    ("post", "/api/v1/knowledge-base/cache/clear"),
    ("get", "/api/v1/knowledge-base/rebuild/status"),
]


class TestKnowledgeBaseAuthMatrix:
    """未认证 → 401, doctor → 403, admin → 非 401/403"""

    @pytest.mark.parametrize("method,path", KB_ENDPOINTS)
    def test_unauthenticated_returns_401(self, client, method, path):
        resp = getattr(client, method)(path)
        assert resp.status_code == 401, f"{method.upper()} {path} should be 401"

    @pytest.mark.parametrize("method,path", KB_ENDPOINTS)
    def test_doctor_returns_403(self, client, method, path):
        _as_doctor()
        resp = getattr(client, method)(path)
        assert resp.status_code == 403, f"{method.upper()} {path} should be 403"

    def test_admin_rebuild_returns_202(self, client):
        _as_admin()
        mock_result = MagicMock()
        mock_result.id = "task-123"
        mock_rebuild = MagicMock()
        mock_rebuild.delay.return_value = mock_result
        with patch(
            "app.tasks.rag_index_task.rebuild_rag_index",
            mock_rebuild,
        ):
            resp = client.post("/api/v1/knowledge-base/rebuild")
        assert resp.status_code == 202

    def test_admin_cache_clear_returns_200(self, client):
        _as_admin()
        with patch(
            "app.api.v1.knowledge_base.clear_embed_cache",
        ):
            resp = client.post("/api/v1/knowledge-base/cache/clear")
        assert resp.status_code == 200


class TestRebuildStatusTaskIdAccess:
    """随机/他人 task ID 对非管理员不可查询 → 404"""

    def test_doctor_cannot_query_task_id(self, client):
        _as_doctor()
        resp = client.get(
            "/api/v1/knowledge-base/rebuild/status",
            params={"task_id": "some-task-id"},
        )
        assert resp.status_code == 403

    def test_admin_can_query_task_id(self, client):
        _as_admin()
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.status = "PENDING"
        mock_result.info = {}
        with patch(
            "app.api.v1.knowledge_base.AsyncResult",
            return_value=mock_result,
        ):
            resp = client.get(
                "/api/v1/knowledge-base/rebuild/status",
                params={"task_id": "some-task-id"},
            )
        assert resp.status_code == 200


class TestCeleryFailureSanitization:
    """Celery FAILURE 只返回标准 error_code，不返回 traceback/文件路径/prompt"""

    def test_failure_does_not_leak_traceback(self, client):
        _as_admin()
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.status = "FAILURE"
        mock_result.info = {}
        # 模拟 Celery 的异常结果，包含敏感信息
        mock_result.result = Exception(
            "Traceback: File /secret/path/prompt.txt\n"
            "API_KEY=sk-secret\n"
            "prompt: 患者信息..."
        )
        with patch(
            "app.api.v1.knowledge_base.AsyncResult",
            return_value=mock_result,
        ):
            resp = client.get(
                "/api/v1/knowledge-base/rebuild/status",
                params={"task_id": "failed-task"},
            )
        assert resp.status_code == 200
        body = resp.text
        # 不得泄露原始异常字符串
        assert "/secret/path" not in body
        assert "sk-secret" not in body
        # 应返回标准 error_code
        assert resp.json().get("error_code") == "TASK_FAILED"
