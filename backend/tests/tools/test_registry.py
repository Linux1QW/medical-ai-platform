"""ToolRegistry 单元测试"""

import pytest
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.registry import ToolRegistry


# ── 测试用工具定义 ──────────────────────────────────────────────────────────────


class DummyArgs(BaseModel):
    query: str = Field(description="测试查询")


class DummyTool(BaseTool):
    name = "dummy_tool"
    description = "A dummy tool for testing"
    args_schema = DummyArgs

    async def execute(self, args, context):
        return {"result": "dummy"}


class AnotherTool(BaseTool):
    name = "another_tool"
    description = "Another dummy tool"
    args_schema = DummyArgs

    async def execute(self, args, context):
        return {"result": "another"}


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_registry():
    """每个测试前重置注册表，确保隔离"""
    registry = ToolRegistry()
    registry.reset()
    yield registry


# ── 测试用例 ─────────────────────────────────────────────────────────────────────


def test_register_tool_success(reset_registry):
    """注册工具成功"""
    registry = reset_registry
    tool = DummyTool()
    registry.register(tool)
    assert "dummy_tool" in registry.list_tools()


def test_register_duplicate_idempotent(reset_registry):
    """重复注册同名工具时幂等跳过，不抛异常"""
    registry = reset_registry
    tool1 = DummyTool()
    registry.register(tool1)

    tool2 = DummyTool()
    registry.register(tool2)  # 不应抛异常
    assert len(registry.list_tools()) == 1  # 仍只有一个工具
    assert registry.get("dummy_tool") is tool1  # 保留第一个实例


def test_get_registered_tool(reset_registry):
    """获取已注册工具返回实例"""
    registry = reset_registry
    tool = DummyTool()
    registry.register(tool)
    retrieved = registry.get("dummy_tool")
    assert retrieved is tool
    assert isinstance(retrieved, DummyTool)


def test_get_unregistered_tool(reset_registry):
    """获取未注册工具返回 None"""
    registry = reset_registry
    result = registry.get("nonexistent_tool")
    assert result is None


def test_list_tools(reset_registry):
    """返回所有已注册工具名列表"""
    registry = reset_registry
    registry.register(DummyTool())
    registry.register(AnotherTool())

    tools = registry.list_tools()
    assert "dummy_tool" in tools
    assert "another_tool" in tools
    assert len(tools) == 2


def test_get_openai_schemas_all(reset_registry):
    """无参数时返回全部 schema"""
    registry = reset_registry
    registry.register(DummyTool())
    registry.register(AnotherTool())

    schemas = registry.get_openai_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"dummy_tool", "another_tool"}


def test_get_openai_schemas_filtered(reset_registry):
    """指定 tool_names 时只返回对应 schema"""
    registry = reset_registry
    registry.register(DummyTool())
    registry.register(AnotherTool())

    schemas = registry.get_openai_schemas(tool_names=["dummy_tool"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "dummy_tool"


def test_reset(reset_registry):
    """reset() 清空注册表"""
    registry = reset_registry
    registry.register(DummyTool())
    assert len(registry.list_tools()) == 1

    registry.reset()
    assert len(registry.list_tools()) == 0


def test_singleton():
    """ToolRegistry 是单例"""
    r1 = ToolRegistry()
    r2 = ToolRegistry()
    assert r1 is r2
    # 清理
    r1.reset()


def test_openai_schema_format(reset_registry):
    """验证 openai_schema() 输出符合 OpenAI Function Calling 格式"""
    registry = reset_registry
    tool = DummyTool()
    registry.register(tool)

    schema = tool.openai_schema()
    assert schema["type"] == "function"
    assert "function" in schema
    fn = schema["function"]
    assert fn["name"] == "dummy_tool"
    assert fn["description"] == "A dummy tool for testing"
    assert "parameters" in fn
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "query" in params["properties"]
    assert "required" in params
