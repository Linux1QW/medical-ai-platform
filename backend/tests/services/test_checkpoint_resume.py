# -*- coding: utf-8 -*-
"""断点续跑测试 — checkpoint 恢复输入解析 / worker checkpointer 重建 / TTL 接通"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services.evaluation_service import _resolve_graph_input


class _StubGraph:
    """可配置 aget_state 行为的图桩"""

    def __init__(self, snapshot=None, error: Exception | None = None):
        self._snapshot = snapshot
        self._error = error

    async def aget_state(self, config):
        if self._error is not None:
            raise self._error
        return self._snapshot


_INITIAL_STATE = {"run_id": "r1", "consultation_id": 1}


# ── _resolve_graph_input：checkpoint 恢复输入解析 ────────────────────────────


@pytest.mark.asyncio
async def test_resume_with_pending_checkpoint_returns_none():
    """存在未完成 checkpoint（values 非空且有待执行节点）→ None 输入从断点恢复"""
    snapshot = SimpleNamespace(
        values={"run_id": "r1", "agent_results": []},
        next=("aggregate_results",),
    )
    graph = _StubGraph(snapshot=snapshot)

    result = await _resolve_graph_input(graph, {}, _INITIAL_STATE)

    assert result is None


@pytest.mark.asyncio
async def test_resume_without_checkpoint_falls_back_to_fresh_run():
    """thread 无 checkpoint（values 为空）→ 全新执行"""
    snapshot = SimpleNamespace(values={}, next=())
    graph = _StubGraph(snapshot=snapshot)

    result = await _resolve_graph_input(graph, {}, _INITIAL_STATE)

    assert result is _INITIAL_STATE


@pytest.mark.asyncio
async def test_resume_with_completed_checkpoint_falls_back_to_fresh_run():
    """图已跑完（next 为空）→ 不做空输入恢复，全新执行更安全"""
    snapshot = SimpleNamespace(values={"evaluation_status": "completed"}, next=())
    graph = _StubGraph(snapshot=snapshot)

    result = await _resolve_graph_input(graph, {}, _INITIAL_STATE)

    assert result is _INITIAL_STATE


@pytest.mark.asyncio
async def test_resume_state_read_error_falls_back_to_fresh_run():
    """读取 checkpoint 状态异常 → 降级全新执行，不阻断评估"""
    graph = _StubGraph(error=RuntimeError("redis gone"))

    result = await _resolve_graph_input(graph, {}, _INITIAL_STATE)

    assert result is _INITIAL_STATE


# ── _ensure_worker_checkpointer：Celery worker 侧按任务重建 ──────────────────


@pytest.mark.asyncio
async def test_ensure_worker_checkpointer_noop_when_disabled(monkeypatch):
    """LANGGRAPH_ENABLED=false 时不触碰 checkpointer"""
    from app.tasks.evaluation_task import _ensure_worker_checkpointer

    monkeypatch.setattr(settings, "LANGGRAPH_ENABLED", False)

    import app.orchestration.checkpointer as cp

    async def boom(*args, **kwargs):
        raise AssertionError("disabled 时不应重建 checkpointer")

    monkeypatch.setattr(cp, "init_checkpointer", boom)

    await _ensure_worker_checkpointer()


@pytest.mark.asyncio
async def test_ensure_worker_checkpointer_rebuilds(monkeypatch):
    """启用时：关旧 → 重置图缓存 → 以配置的 URL/TTL 重建"""
    from app.tasks.evaluation_task import _ensure_worker_checkpointer

    monkeypatch.setattr(settings, "LANGGRAPH_ENABLED", True)

    import app.orchestration.checkpointer as cp
    import app.orchestration.graph as graph_mod

    calls: list = []

    async def fake_close():
        calls.append("close_checkpointer")

    async def fake_close_graph():
        calls.append("close_graph")

    async def fake_init(redis_url=None, ttl=None):
        calls.append(("init", redis_url, ttl))

    monkeypatch.setattr(cp, "close_checkpointer", fake_close)
    monkeypatch.setattr(cp, "init_checkpointer", fake_init)
    monkeypatch.setattr(graph_mod, "close_graph", fake_close_graph)

    await _ensure_worker_checkpointer()

    assert calls == [
        "close_checkpointer",
        "close_graph",
        ("init", settings.REDIS_CHECKPOINT_URL, settings.REDIS_CHECKPOINT_TTL),
    ]


# ── checkpointer：TTL 接通与关闭容错 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_init_checkpointer_passes_ttl_in_minutes(monkeypatch):
    """REDIS_CHECKPOINT_TTL（秒）换算为 AsyncRedisSaver 的 default_ttl（分钟）"""
    import langgraph.checkpoint.redis.aio as aio_mod

    from app.orchestration import checkpointer as cp

    monkeypatch.setattr(settings, "LANGGRAPH_ENABLED", True)
    captured = {}

    @asynccontextmanager
    async def fake_from_conn_string(redis_url=None, *, ttl=None, **kwargs):
        captured["redis_url"] = redis_url
        captured["ttl"] = ttl
        yield object()

    monkeypatch.setattr(aio_mod.AsyncRedisSaver, "from_conn_string", fake_from_conn_string)

    try:
        await cp.init_checkpointer(redis_url="redis://test:6379/1", ttl=86400)
        assert captured["redis_url"] == "redis://test:6379/1"
        assert captured["ttl"] == {"default_ttl": 1440}  # 86400s = 1440min
    finally:
        await cp.close_checkpointer()


@pytest.mark.asyncio
async def test_init_checkpointer_zero_ttl_means_no_expiry(monkeypatch):
    """ttl=0 → 不设置过期（ttl_config=None）"""
    import langgraph.checkpoint.redis.aio as aio_mod

    from app.orchestration import checkpointer as cp

    monkeypatch.setattr(settings, "LANGGRAPH_ENABLED", True)
    captured = {}

    @asynccontextmanager
    async def fake_from_conn_string(redis_url=None, *, ttl=None, **kwargs):
        captured["ttl"] = ttl
        yield object()

    monkeypatch.setattr(aio_mod.AsyncRedisSaver, "from_conn_string", fake_from_conn_string)

    try:
        await cp.init_checkpointer(redis_url="redis://test:6379/1", ttl=0)
        assert captured["ttl"] is None
    finally:
        await cp.close_checkpointer()


@pytest.mark.asyncio
async def test_close_checkpointer_swallows_close_errors():
    """关闭异常（如事件循环已切换）仅告警，全局引用必须被清理"""
    from app.orchestration import checkpointer as cp

    class _BoomStack:
        async def aclose(self):
            raise RuntimeError("Event loop is closed")

    cp._exit_stack = _BoomStack()
    cp._checkpointer = object()

    await cp.close_checkpointer()

    assert cp._checkpointer is None
    assert cp._exit_stack is None
