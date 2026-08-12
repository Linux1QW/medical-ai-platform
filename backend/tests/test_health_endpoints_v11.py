# -*- coding: utf-8 -*-
"""健康端点测试 — /health/live 和 /health/ready

TDD Phase 1: 这些测试在实现之前应该失败。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建测试客户端，不触发真实 lifespan"""
    from app.main import app
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()


# ── /health/live ──────────────────────────────────────────────────────────────


class TestHealthLive:
    """liveness: 只证明进程存活，永远不查询外部依赖"""

    def test_live_returns_200_with_status_and_version(self, client):
        """live 端点返回 200 + 当前应用版本"""
        resp = client.get("/health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.2.0"

    def test_live_does_not_contain_sensitive_info(self, client):
        """live 响应不含 token_usage、prompt、API key"""
        resp = client.get("/health/live")
        body = resp.json()
        assert "token_usage" not in body
        assert "prompt" not in body
        assert "api_key" not in body
        assert "secret" not in body

    def test_live_does_not_query_db_or_redis(self, client):
        """live 端点不查询 MySQL 或 Redis"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main._get_cache_redis") as mock_redis:
            resp = client.get("/health/live")
            assert resp.status_code == 200
            # engine.connect 不应被调用
            mock_engine.connect.assert_not_called()
            # Redis 不应被查询
            mock_redis.assert_not_called()


# ── /health/ready ─────────────────────────────────────────────────────────────


class TestHealthReady:
    """readiness: 检查 MySQL、状态 Redis、LangGraph checkpointer 和 progress bus"""

    def test_ready_returns_200_when_all_healthy(self, client):
        """所有依赖正常时返回 200"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb:
            # MySQL OK
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            # Checkpointer OK
            mock_cp.return_value = MagicMock()
            # Progress bus OK
            mock_pb.return_value = True

            resp = client.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"

    def test_ready_returns_503_when_mysql_down(self, client):
        """MySQL 不可用时返回 503"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb:
            # MySQL FAIL
            mock_engine.connect.side_effect = Exception("connection refused")
            mock_cp.return_value = MagicMock()
            mock_pb.return_value = True

            resp = client.get("/health/ready")
            assert resp.status_code == 503

    def test_ready_returns_503_when_checkpointer_down(self, client):
        """checkpointer 不可用时返回 503"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb:
            # MySQL OK
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            # Checkpointer FAIL
            mock_cp.return_value = None
            mock_pb.return_value = True

            resp = client.get("/health/ready")
            assert resp.status_code == 503

    def test_ready_returns_503_when_progress_bus_down(self, client):
        """progress bus 不可用时返回 503"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb:
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_cp.return_value = MagicMock()
            mock_pb.return_value = False

            resp = client.get("/health/ready")
            assert resp.status_code == 503

    def test_ready_returns_200_degraded_when_cache_redis_down(self, client):
        """redis-cache 故障时返回 200 + degraded=['cache']"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb, \
             patch("app.main._get_cache_redis") as mock_cache:
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_cp.return_value = MagicMock()
            mock_pb.return_value = True
            # Cache Redis FAIL
            mock_cache.return_value = None

            resp = client.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert "cache" in body.get("degraded", [])

    def test_ready_does_not_contain_sensitive_info(self, client):
        """ready 响应不含 token_usage、prompt、API key"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._check_progress_bus") as mock_pb, \
             patch("app.main._get_cache_redis") as mock_cache:
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_cp.return_value = MagicMock()
            mock_pb.return_value = True
            mock_cache.return_value = None

            resp = client.get("/health/ready")
            body = resp.json()
            assert "token_usage" not in body
            assert "prompt" not in body
            assert "api_key" not in body


# ── /health 兼容 ──────────────────────────────────────────────────────────────


class TestHealthCompat:
    """旧 /health 保留兼容但只返回 minimal ready 结果"""

    def test_health_does_not_expose_token_usage(self, client):
        """/health 响应不含 token_usage"""
        with patch("app.main.engine") as mock_engine, \
             patch("app.main.get_checkpointer") as mock_cp, \
             patch("app.main._get_cache_redis") as mock_cache:
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            mock_cp.return_value = MagicMock()
            mock_cache.return_value = None

            resp = client.get("/health")
            body = resp.json()
            assert "token_usage" not in body
            assert "llm_cache" not in body
            assert "retrieval_cache" not in body
