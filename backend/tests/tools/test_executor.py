"""ToolExecutor 单元测试"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor
from app.services.tools.registry import ToolRegistry

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
