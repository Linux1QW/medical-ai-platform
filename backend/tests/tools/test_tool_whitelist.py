# -*- coding: utf-8 -*-
"""角色工具白名单测试：策略映射、开关、执行器强制校验"""

import pytest
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.tools import register_all_tools
from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.executor import ToolExecutor
from app.services.tools.policy import AGENT_TOOL_WHITELIST, get_allowed_tools
from app.services.tools.registry import ToolRegistry
from app.services.tools.robust_tool_executor import RobustToolExecutor

# ── 测试用工具 ──────────────────────────────────────────────────────────────


class SimpleArgs(BaseModel):
    query: str = Field(description="测试查询")


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echoes input"
    args_schema = SimpleArgs

    async def execute(self, args, context):
        return {"echo": args.query}


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.reset()
    reg.register(EchoTool())
    yield reg
    reg.reset()


# ── 策略映射 ────────────────────────────────────────────────────────────────


def test_whitelist_only_contains_registered_tools():
    """白名单中的工具名必须都是真实注册的工具，防止拼写漂移"""
    reg = ToolRegistry()
    reg.reset()
    register_all_tools(reg)
    registered = set(reg.list_tools())
    reg.reset()

    for agent_name, allowed in AGENT_TOOL_WHITELIST.items():
        unknown = allowed - registered
        assert not unknown, f"{agent_name} 白名单含未注册工具: {unknown}"


def test_get_allowed_tools_known_and_unknown_agent():
    assert get_allowed_tools("knowledge_agent") == AGENT_TOOL_WHITELIST["knowledge_agent"]
    assert "verify_citation" in get_allowed_tools("knowledge_agent")
    assert "check_score_consistency" in get_allowed_tools("reflection_agent")
    # 未登记角色不限制（向后兼容）
    assert get_allowed_tools("some_new_agent") is None


def test_get_allowed_tools_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(settings, "TOOL_ROLE_WHITELIST_ENABLED", False)
    assert get_allowed_tools("knowledge_agent") is None
    assert get_allowed_tools("reflection_agent") is None


# ── 执行器强制校验 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_robust_executor_blocks_forbidden_tool(registry):
    executor = RobustToolExecutor(registry)
    context = ToolContext(agent_name="reflection_agent", allowed_tools={"summarize_evaluation"})

    result = await executor.execute("echo_tool", '{"query": "hi"}', context=context)

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_forbidden"
    assert "reflection_agent" in result["error"]["message"]


@pytest.mark.asyncio
async def test_robust_executor_allows_whitelisted_tool(registry):
    executor = RobustToolExecutor(registry)
    context = ToolContext(agent_name="knowledge_agent", allowed_tools={"echo_tool"})

    result = await executor.execute("echo_tool", '{"query": "hi"}', context=context)

    assert result["ok"] is True
    assert result["data"] == {"echo": "hi"}


@pytest.mark.asyncio
async def test_robust_executor_none_whitelist_unrestricted(registry):
    """allowed_tools=None（未登记角色/开关关闭）时不限制"""
    executor = RobustToolExecutor(registry)
    context = ToolContext(agent_name="legacy_agent", allowed_tools=None)

    result = await executor.execute("echo_tool", '{"query": "hi"}', context=context)

    assert result["ok"] is True


@pytest.mark.asyncio
async def test_lightweight_executor_blocks_forbidden_tool(registry):
    """TOOL_EXECUTOR_HARDENED=False 回退路径同样强制白名单"""
    executor = ToolExecutor(registry)
    context = ToolContext(agent_name="reflection_agent", allowed_tools={"summarize_evaluation"})

    result = await executor.execute("echo_tool", '{"query": "hi"}', context=context)

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_forbidden"


@pytest.mark.asyncio
async def test_lightweight_executor_without_context_unrestricted(registry):
    """context=None 的历史调用方不受影响"""
    executor = ToolExecutor(registry)

    result = await executor.execute("echo_tool", '{"query": "hi"}', context=None)

    assert result["ok"] is True
