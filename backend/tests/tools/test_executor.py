"""ToolExecutor 单元测试"""

import asyncio

import pytest
from prometheus_client import REGISTRY
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor, ToolExecutorBridge
from app.services.tools.registry import ToolRegistry
from app.services.tools.robust_tool_executor import RobustToolExecutor

# ── 测试用工具定义 ──────────────────────────────────────────────────────────────


class SimpleArgs(BaseModel):
    query: str = Field(description="测试查询")


class SuccessTool(BaseTool):
    name = "success_tool"
    description = "A tool that always succeeds"
    args_schema = SimpleArgs

    async def execute(self, args, context):
        return {"result": "ok"}


class SlowTool(BaseTool):
    name = "slow_tool"
    description = "A tool that times out"
    timeout_seconds = 1
    args_schema = SimpleArgs

    async def execute(self, args, context):
        await asyncio.sleep(10)
        return {"result": "too late"}


class ErrorTool(BaseTool):
    name = "error_tool"
    description = "A tool that raises"
    critical = True
    args_schema = SimpleArgs

    async def execute(self, args, context):
        raise RuntimeError("boom")


class VerboseTool(BaseTool):
    name = "verbose_tool"
    description = "Returns very long result"
    args_schema = SimpleArgs

    async def execute(self, args, context):
        return {"data": "x" * 10000}


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """创建并注册测试工具的 registry"""
    reg = ToolRegistry()
    reg.reset()
    reg.register(SuccessTool())
    reg.register(SlowTool())
    reg.register(ErrorTool())
    reg.register(VerboseTool())
    yield reg
    reg.reset()


@pytest.fixture
def executor(registry):
    """创建 ToolExecutor"""
    return ToolExecutor(registry, max_result_chars=6000)


# ── 测试用例 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_success(executor):
    """正常执行返回 ok=True"""
    result = await executor.execute("success_tool", '{"query": "test"}')
    assert result["ok"] is True
    assert result["data"]["result"] == "ok"
    assert result["error"] is None
    assert result["degraded"] is False


@pytest.mark.asyncio
async def test_execute_unknown_tool(executor):
    """未注册工具返回 unknown_tool 错误"""
    result = await executor.execute("nonexistent_tool", '{"query": "test"}')
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"


@pytest.mark.asyncio
async def test_execute_invalid_json(executor):
    """非法 JSON 参数返回 invalid_arguments 错误"""
    result = await executor.execute("success_tool", "not valid json {{{")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_execute_validation_error(executor):
    """Pydantic 参数校验失败返回 validation_error"""
    # SimpleArgs 需要 query 字段，传入空对象
    result = await executor.execute("success_tool", "{}")
    assert result["ok"] is False
    assert result["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_execute_timeout(executor):
    """工具超时返回 timeout 错误"""
    result = await executor.execute("slow_tool", '{"query": "test"}')
    assert result["ok"] is False
    assert result["error"]["code"] == "timeout"
    assert result["degraded"] is False  # slow_tool.critical == False


@pytest.mark.asyncio
async def test_execute_exception(executor):
    """工具内部异常返回 execution_error"""
    result = await executor.execute("error_tool", '{"query": "test"}')
    assert result["ok"] is False
    assert result["error"]["code"] == "execution_error"
    assert "boom" in result["error"]["message"]


@pytest.mark.asyncio
async def test_budget_exceeded(registry):
    """预算耗尽返回 budget_exceeded"""
    executor = ToolExecutor(registry)
    budget = ToolBudget({"success_tool": 1})

    # 第一次执行成功
    result1 = await executor.execute(
        "success_tool", '{"query": "test"}', budget=budget
    )
    assert result1["ok"] is True

    # 第二次执行预算耗尽
    result2 = await executor.execute(
        "success_tool", '{"query": "test"}', budget=budget
    )
    assert result2["ok"] is False
    assert result2["error"]["code"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_result_truncation(executor):
    """超长结果自动截断"""
    result = await executor.execute("verbose_tool", '{"query": "test"}')
    assert result["ok"] is True
    # 结果应被截断
    assert result["data"].get("truncated") is True


@pytest.mark.asyncio
async def test_trace_recording(executor):
    """执行后 trace 记录完整"""
    executor.clear_traces()
    await executor.execute("success_tool", '{"query": "hello"}')

    traces = executor.get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace["tool_name"] == "success_tool"
    assert trace["status"] == "success"
    assert trace["trace_id"].startswith("tool-")
    assert "elapsed_ms" in trace


@pytest.mark.asyncio
async def test_critical_tool_degraded(executor):
    """critical=True 的工具失败时 degraded=True"""
    result = await executor.execute("error_tool", '{"query": "test"}')
    assert result["ok"] is False
    assert result["degraded"] is True  # error_tool.critical == True


# ── Prometheus 工具指标 ─────────────────────────────────────────────────────────


def _calls_total(tool: str, agent: str, status: str) -> float:
    return REGISTRY.get_sample_value(
        "tool_calls_total", {"tool": tool, "agent": agent, "status": status}
    ) or 0.0


def _duration_count(tool: str) -> float:
    return REGISTRY.get_sample_value(
        "tool_call_duration_seconds_count", {"tool": tool}
    ) or 0.0


class TestPrometheusToolMetrics:
    @pytest.mark.asyncio
    async def test_success_increments_counter_and_histogram(self, executor):
        """成功执行打点 tool_calls_total{status=success} 与耗时直方图"""
        before = _calls_total("success_tool", "patient_agent", "success")
        before_dur = _duration_count("success_tool")
        ctx = ToolContext(agent_name="patient_agent")
        await executor.execute("success_tool", '{"query": "t"}', context=ctx)
        assert _calls_total("success_tool", "patient_agent", "success") == before + 1
        assert _duration_count("success_tool") == before_dur + 1

    @pytest.mark.asyncio
    async def test_no_context_agent_label_empty(self, executor):
        """无 context 时 agent 标签为空串"""
        before = _calls_total("success_tool", "", "success")
        await executor.execute("success_tool", '{"query": "t"}')
        assert _calls_total("success_tool", "", "success") == before + 1

    @pytest.mark.asyncio
    async def test_error_status_recorded(self, executor):
        """执行异常打点 status=error"""
        before = _calls_total("error_tool", "", "error")
        await executor.execute("error_tool", '{"query": "t"}')
        assert _calls_total("error_tool", "", "error") == before + 1

    @pytest.mark.asyncio
    async def test_robust_executor_also_records(self, registry):
        """RobustToolExecutor 同样打点（独立类自有 _record_trace）"""
        robust = RobustToolExecutor(registry, enable_retry=False)
        before = _calls_total("success_tool", "knowledge_agent", "success")
        ctx = ToolContext(agent_name="knowledge_agent")
        await robust.execute("success_tool", '{"query": "t"}', context=ctx)
        assert _calls_total("success_tool", "knowledge_agent", "success") == before + 1


# ── ToolExecutorBridge ──────────────────────────────────────────────────────────


class TestToolExecutorBridge:
    @pytest.mark.asyncio
    async def test_bridge_binds_context_and_budget(self, executor):
        """桥接器绑定 context/budget：白名单与预算在两参调用下生效"""
        ctx = ToolContext(agent_name="patient_agent", allowed_tools=frozenset({"success_tool"}))
        budget = ToolBudget({"success_tool": 1})
        bridge = ToolExecutorBridge(executor, ctx, budget)

        r1 = await bridge.execute("success_tool", '{"query": "t"}')
        assert r1["ok"] is True

        # 预算绑定生效：第二次超预算
        r2 = await bridge.execute("success_tool", '{"query": "t"}')
        assert r2["ok"] is False
        assert r2["error"]["code"] == "budget_exceeded"

        # 白名单绑定生效：非白名单工具被拒
        r3 = await bridge.execute("error_tool", '{"query": "t"}')
        assert r3["ok"] is False
        assert r3["error"]["code"] == "tool_forbidden"
