# -*- coding: utf-8 -*-
"""监控端点测试 — 工具运行时快照、run 级 trace、失败归因与成本聚合（鉴权 + 数据组装）"""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_admin, get_current_user
from app.db.session import get_db
from app.main import app

ADMIN_USER = SimpleNamespace(id=1, username="admin", role="admin")
DOCTOR_USER = SimpleNamespace(id=2, username="doctor", role="doctor")


class _FakeRunResult:
    def __init__(self, run):
        self._run = run

    def scalar_one_or_none(self):
        return self._run


class _FakeNodesResult:
    def __init__(self, nodes):
        self._nodes = nodes

    def scalars(self):
        return SimpleNamespace(all=lambda: self._nodes)


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _FakeRowsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    """按调用顺序返回预设结果（第一次 run 查询，第二次 nodes 查询）"""

    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


@pytest.fixture
def client():
    # 不进入上下文管理器，避免触发 lifespan（LangGraph checkpointer 等真实依赖）
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()
    app.dependency_overrides.clear()


def _as_admin(db=None):
    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_admin] = lambda: ADMIN_USER
    app.dependency_overrides[get_current_user] = lambda: ADMIN_USER


def _as_doctor():
    def _override_get_db():
        yield None

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = lambda: DOCTOR_USER


# ── 鉴权 ─────────────────────────────────────────────────────────────────────


class TestMonitoringAuth:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/admin/monitoring/tool-runtime",
            "/api/v1/admin/monitoring/runs/run-1/trace",
            "/api/v1/admin/monitoring/failures/summary",
            "/api/v1/admin/monitoring/usage/summary",
        ],
    )
    def test_requires_auth(self, client, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/admin/monitoring/tool-runtime",
            "/api/v1/admin/monitoring/runs/run-1/trace",
            "/api/v1/admin/monitoring/failures/summary",
            "/api/v1/admin/monitoring/usage/summary",
        ],
    )
    def test_forbidden_for_doctor(self, client, path):
        _as_doctor()
        assert client.get(path).status_code == 403


# ── 工具运行时快照 ───────────────────────────────────────────────────────────


class TestToolRuntimeSnapshot:
    def test_returns_snapshot_for_admin(self, client):
        _as_admin()
        resp = client.get("/api/v1/admin/monitoring/tool-runtime")
        assert resp.status_code == 200
        body = resp.json()
        # 快照结构来自 get_tool_runtime_snapshot
        assert "hardened_executor" in body
        assert "circuit_breakers" in body


# ── run 级 trace 查询 ────────────────────────────────────────────────────────


def _sample_run():
    return SimpleNamespace(
        id="run-1",
        consultation_id=42,
        evaluation_id=7,
        status="completed",
        graph_version="evaluation-graph-v1",
        selected_agents=["knowledge", "inquiry"],
        error_type=None,
        error_message=None,
        started_at=datetime(2026, 5, 26, 10, 0, 0),
        finished_at=datetime(2026, 5, 26, 10, 1, 0),
    )


def _sample_node():
    return SimpleNamespace(
        id=1,
        node_name="agent:knowledge",
        attempt=1,
        status="success",
        duration_ms=1200,
        result_summary={"score": 88, "trace": {"tool_trace": []}},
        error_type=None,
        started_at=datetime(2026, 5, 26, 10, 0, 1),
        finished_at=datetime(2026, 5, 26, 10, 0, 2),
    )


class TestRunTrace:
    def test_returns_trace_tree(self, client):
        db = _FakeDB([
            _FakeRunResult(_sample_run()),
            _FakeNodesResult([_sample_node()]),
        ])
        _as_admin(db)
        resp = client.get("/api/v1/admin/monitoring/runs/run-1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run"]["run_id"] == "run-1"
        assert body["run"]["consultation_id"] == 42
        assert body["node_count"] == 1
        node = body["nodes"][0]
        assert node["node_name"] == "agent:knowledge"
        assert node["duration_ms"] == 1200
        assert node["result_summary"]["score"] == 88
        assert node["started_at"] == "2026-05-26T10:00:01"

    def test_missing_run_returns_404(self, client):
        db = _FakeDB([_FakeRunResult(None)])
        _as_admin(db)
        resp = client.get("/api/v1/admin/monitoring/runs/nope/trace")
        assert resp.status_code == 404


# ── 失败归因聚合 ─────────────────────────────────────────────────────────────


def _failed_run(run_id="run-f1", error_type="timeout"):
    return SimpleNamespace(
        id=run_id,
        consultation_id=42,
        status="failed",
        error_type=error_type,
        error_message="x" * 300,
        attempt=2,
        started_at=datetime(2026, 5, 26, 10, 0, 0),
        finished_at=datetime(2026, 5, 26, 10, 1, 0),
    )


class TestFailuresSummary:
    def test_aggregates_by_reason(self, client):
        # 固定 3 次 execute 顺序：总数 → error_type 分组 → 最近失败样本
        db = _FakeDB([
            _FakeScalarResult(20),
            _FakeRowsResult([("timeout", 3), ("connection_error", 1)]),
            _FakeNodesResult([_failed_run()]),
        ])
        _as_admin(db)
        resp = client.get("/api/v1/admin/monitoring/failures/summary?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["days"] == 7
        assert body["total_runs"] == 20
        assert body["failed_runs"] == 4
        assert body["failure_rate"] == 0.2
        assert body["by_reason"] == {"timeout": 3, "connection_error": 1}
        failure = body["recent_failures"][0]
        assert failure["run_id"] == "run-f1"
        assert failure["error_type"] == "timeout"
        assert failure["attempt"] == 2
        # error_message 截断到 200 字符
        assert len(failure["error_message"]) == 200
        assert failure["started_at"] == "2026-05-26T10:00:00"

    def test_empty_and_days_clamped(self, client):
        db = _FakeDB([
            _FakeScalarResult(0),
            _FakeRowsResult([]),
            _FakeNodesResult([]),
        ])
        _as_admin(db)
        resp = client.get("/api/v1/admin/monitoring/failures/summary?days=500")
        assert resp.status_code == 200
        body = resp.json()
        assert body["days"] == 90  # 上限钳制
        assert body["failed_runs"] == 0
        assert body["failure_rate"] == 0.0
        assert body["recent_failures"] == []


# ── Token 成本聚合 ───────────────────────────────────────────────────────────


class TestUsageSummary:
    def test_returns_runs_and_daily(self, client, monkeypatch):
        from app.services.token_tracker import token_tracker

        runs_summary = {
            "runs_count": 2, "prompt_tokens": 800, "completion_tokens": 200,
            "total_tokens": 1000, "estimated_cost": 0.02,
            "by_agent": {"knowledge": 600}, "top_runs": [],
        }

        async def fake_runs_summary(top_n=10):
            return runs_summary

        async def fake_daily(date_str=None):
            return {"date": date_str, "total_tokens": 100}

        monkeypatch.setattr(token_tracker, "get_runs_usage_summary", fake_runs_summary)
        monkeypatch.setattr(token_tracker, "get_daily_usage", fake_daily)
        _as_admin()
        resp = client.get("/api/v1/admin/monitoring/usage/summary?days=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["days"] == 3
        assert body["runs"] == runs_summary
        assert len(body["daily"]) == 3


class _FakeRedis:
    """最小 SCAN + HGETALL 假 Redis，用于 run 用量聚合单测"""

    def __init__(self, store):
        self._store = store

    def scan_iter(self, match=None, count=None):
        async def _gen():
            for key in self._store:
                yield key

        return _gen()

    async def hgetall(self, key):
        return self._store.get(key, {})


class TestRunsUsageAggregation:
    def test_aggregates_runs_and_agents(self, monkeypatch):
        import app.services.token_tracker as tt

        store = {
            "token_usage:run:run-a": {
                "prompt_tokens": "600", "completion_tokens": "400", "total_tokens": "1000",
                "agent:knowledge:total_tokens": "700", "agent:inquiry:total_tokens": "300",
            },
            "token_usage:run:run-b": {
                "prompt_tokens": "300", "completion_tokens": "200", "total_tokens": "500",
                "agent:knowledge:total_tokens": "500",
            },
        }

        async def fake_get_redis():
            return _FakeRedis(store)

        monkeypatch.setattr(tt, "_get_redis", fake_get_redis)
        summary = asyncio.run(tt.token_tracker.get_runs_usage_summary())
        assert summary["runs_count"] == 2
        assert summary["total_tokens"] == 1500
        assert summary["by_agent"] == {"knowledge": 1200, "inquiry": 300}
        # top_runs 按 total_tokens 降序
        assert summary["top_runs"][0] == {"run_id": "run-a", "total_tokens": 1000}
        # estimated_cost = 1500/1000 * COST_PER_1K_TOKENS(0.02)
        assert summary["estimated_cost"] == 0.03

    def test_redis_unavailable_returns_zeroes(self, monkeypatch):
        import app.services.token_tracker as tt

        async def fake_get_redis():
            return None

        monkeypatch.setattr(tt, "_get_redis", fake_get_redis)
        summary = asyncio.run(tt.token_tracker.get_runs_usage_summary())
        assert summary["runs_count"] == 0
        assert summary["total_tokens"] == 0
        assert summary["estimated_cost"] == 0.0
        assert summary["top_runs"] == []
