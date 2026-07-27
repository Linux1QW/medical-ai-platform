# -*- coding: utf-8 -*-
"""runtime 装配层测试：工厂开关、共享熔断/统计、托管预算、健康门控"""

import pytest
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.tools import runtime
from app.services.tools.base import BaseTool
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor
from app.services.tools.registry import ToolRegistry
from app.services.tools.robust_tool_executor import RobustToolExecutor
from app.services.tools.runtime import (
    ManagedToolBudget,
    create_tool_budget,
    create_tool_executor,
    get_tool_runtime_snapshot,
)

# ── 测试用工具定义 ──────────────────────────────────────────────────────────────


class SimpleArgs(BaseModel):
    query: str = Field(description="测试查询")


class CountingTool(BaseTool):
    """记录实际执行次数，用于验证门控是否真正跳过执行"""
    name = "counting_tool"
    description = "Counts executions"
    args_schema = SimpleArgs
    call_count = 0

    async def execute(self, args, context):
        CountingTool.call_count += 1
        return {"result": "ok"}


class FailTool(BaseTool):
    """抛出不可重试异常（RuntimeError），每次调用记一次熔断失败"""
    name = "fail_tool"
    description = "Always fails"
    args_schema = SimpleArgs

    async def execute(self, args, context):
        raise RuntimeError("boom")


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.reset()
    reg.register(CountingTool())
    reg.register(FailTool())
    CountingTool.call_count = 0
    yield reg
    reg.reset()


@pytest.fixture(autouse=True)
def clean_runtime_state():
    """清理进程级共享状态，避免测试间熔断/统计/会话互相污染"""
    runtime._shared_circuit_breakers.clear()
    runtime._shared_stats.clear()
    runtime.budget_manager._sessions.clear()
    yield
    runtime._shared_circuit_breakers.clear()
    runtime._shared_stats.clear()
    runtime.budget_manager._sessions.clear()


# ── 执行器工厂 ────────────────────────────────────────────────────────────────


def test_factory_returns_robust_when_hardened(registry, monkeypatch):
    """TOOL_EXECUTOR_HARDENED=True 时返回加固执行器"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)
    executor = create_tool_executor(registry)
    assert isinstance(executor, RobustToolExecutor)
    # 使用进程级共享熔断/统计字典
    assert executor._circuit_breakers is runtime._shared_circuit_breakers
    assert executor._stats is runtime._shared_stats


def test_factory_returns_light_when_disabled(registry, monkeypatch):
    """TOOL_EXECUTOR_HARDENED=False 时回退轻量执行器"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", False)
    executor = create_tool_executor(registry)
    assert isinstance(executor, ToolExecutor)
    assert not isinstance(executor, RobustToolExecutor)


@pytest.mark.asyncio
async def test_circuit_breaker_shared_across_executor_instances(registry, monkeypatch):
    """熔断器状态跨 executor 实例累计：新建实例仍处于熔断"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_MAX_RETRIES", 0)
    monkeypatch.setattr(settings, "TOOL_CIRCUIT_BREAKER_THRESHOLD", 2)

    executor1 = create_tool_executor(registry)
    for _ in range(2):
        result = await executor1.execute("fail_tool", '{"query": "t"}')
        assert result["error"]["code"] == "execution_error"

    # 模拟下一次评估重建 executor —— 共享熔断器应立即拒绝
    executor2 = create_tool_executor(registry)
    result = await executor2.execute("fail_tool", '{"query": "t"}')
    assert result["ok"] is False
    assert result["error"]["code"] == "circuit_open"


@pytest.mark.asyncio
async def test_stats_shared_across_executor_instances(registry, monkeypatch):
    """调用统计跨 executor 实例累计"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)

    executor1 = create_tool_executor(registry)
    await executor1.execute("counting_tool", '{"query": "t"}')
    executor2 = create_tool_executor(registry)
    await executor2.execute("counting_tool", '{"query": "t"}')

    assert runtime._shared_stats["counting_tool"].total_calls == 2


@pytest.mark.asyncio
async def test_traces_stay_per_run(registry, monkeypatch):
    """trace 保持单次运行独立，不跨实例共享"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)

    executor1 = create_tool_executor(registry)
    await executor1.execute("counting_tool", '{"query": "t"}')
    executor2 = create_tool_executor(registry)

    assert len(executor1.get_traces()) == 1
    assert len(executor2.get_traces()) == 0


# ── 预算工厂 ──────────────────────────────────────────────────────────────────


def test_budget_factory_returns_plain_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "TOOL_BUDGET_MANAGER_ENABLED", False)
    budget = create_tool_budget({"counting_tool": 3}, "run-1")
    assert type(budget) is ToolBudget


def test_budget_factory_returns_managed_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "TOOL_BUDGET_MANAGER_ENABLED", True)
    budget = create_tool_budget({"counting_tool": 3}, "run-1")
    assert isinstance(budget, ManagedToolBudget)


def test_managed_budget_check_consume(monkeypatch):
    """托管预算：配额耗尽后 check 返回 False，且管理器侧同步记账"""
    monkeypatch.setattr(settings, "TOOL_BUDGET_MANAGER_ENABLED", True)
    budget = create_tool_budget({"counting_tool": 2}, "run-2")

    assert budget.check("counting_tool") is True
    budget.consume("counting_tool")
    budget.consume("counting_tool")
    assert budget.check("counting_tool") is False

    session = runtime.budget_manager.get_or_create_session("run-2")
    assert session.total_calls == 2
    assert session.remaining("counting_tool") == 0


def test_managed_budget_session_reuse(monkeypatch):
    """相同 run_id 复用同一会话，预算跨实例累计"""
    monkeypatch.setattr(settings, "TOOL_BUDGET_MANAGER_ENABLED", True)
    budget1 = create_tool_budget({"counting_tool": 3}, "run-3")
    budget1.consume("counting_tool")
    budget2 = create_tool_budget({"counting_tool": 3}, "run-3")
    budget2.consume("counting_tool")

    session = runtime.budget_manager.get_or_create_session("run-3")
    assert session.total_calls == 2


@pytest.mark.asyncio
async def test_managed_budget_with_executor(registry, monkeypatch):
    """托管预算接入执行器：超额调用返回 budget_exceeded"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)
    monkeypatch.setattr(settings, "TOOL_BUDGET_MANAGER_ENABLED", True)

    executor = create_tool_executor(registry)
    budget = create_tool_budget({"counting_tool": 1}, "run-4")

    result1 = await executor.execute("counting_tool", '{"query": "t"}', budget=budget)
    assert result1["ok"] is True
    result2 = await executor.execute("counting_tool", '{"query": "t"}', budget=budget)
    assert result2["ok"] is False
    assert result2["error"]["code"] == "budget_exceeded"
    assert CountingTool.call_count == 1


# ── 健康门控 ──────────────────────────────────────────────────────────────────


class _FakeStatus:
    value = "unavailable"


class FakeHealthChecker:
    """最小健康检查器桩：可控 is_healthy 返回值"""

    def __init__(self, healthy: bool):
        self._healthy = healthy

    def is_healthy(self, tool_name: str) -> bool:
        return self._healthy

    def get_status(self, tool_name: str):
        return _FakeStatus()

    def get_degraded_result(self, tool_name: str) -> dict:
        return {"degraded": True, "evidence": []}


@pytest.mark.asyncio
async def test_health_gate_blocks_unhealthy_tool(registry):
    """工具不健康时跳过执行，返回降级结果"""
    executor = RobustToolExecutor(registry, health_checker=FakeHealthChecker(healthy=False))
    result = await executor.execute("counting_tool", '{"query": "t"}')

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_unhealthy"
    assert result["degraded"] is True
    assert result["data"] == {"degraded": True, "evidence": []}
    assert CountingTool.call_count == 0  # 未真正执行
    assert executor.get_traces()[0]["status"] == "unhealthy_skipped"


@pytest.mark.asyncio
async def test_health_gate_passes_healthy_tool(registry):
    """工具健康时正常执行"""
    executor = RobustToolExecutor(registry, health_checker=FakeHealthChecker(healthy=True))
    result = await executor.execute("counting_tool", '{"query": "t"}')

    assert result["ok"] is True
    assert CountingTool.call_count == 1


# ── 可观测性快照 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_snapshot_structure(registry, monkeypatch):
    """快照包含熔断/统计/预算/健康四类信息"""
    monkeypatch.setattr(settings, "TOOL_EXECUTOR_HARDENED", True)
    executor = create_tool_executor(registry)
    await executor.execute("counting_tool", '{"query": "t"}')

    snapshot = get_tool_runtime_snapshot()
    assert snapshot["hardened_executor"] is True
    assert "counting_tool" in snapshot["circuit_breakers"]
    assert snapshot["tool_stats"]["counting_tool"]["total_calls"] == 1
    assert "global_total_calls" in snapshot["budget"]
    assert snapshot["health"] is None  # 健康探测默认未启动
