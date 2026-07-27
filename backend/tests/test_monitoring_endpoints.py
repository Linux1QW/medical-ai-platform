# -*- coding: utf-8 -*-
"""监控端点测试 — 工具运行时快照与 run 级 trace 查询（鉴权 + 数据组装）"""

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
        ],
    )
    def test_requires_auth(self, client, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/admin/monitoring/tool-runtime",
            "/api/v1/admin/monitoring/runs/run-1/trace",
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
