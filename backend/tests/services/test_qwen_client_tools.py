"""qwen_client Tool Calling 单元测试

使用 mock 模拟 Qwen API 响应，测试 call_qwen_with_tools 函数的各种场景。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.qwen_client import (
    call_qwen_with_tools,
    call_qwen_chat,
    ToolCallResult,
    ToolCallTrace,
)


# ── Mock 辅助 ────────────────────────────────────────────────────────────────────


def make_mock_response(content=None, tool_calls=None):
    """构造模拟的 Qwen API response"""
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    # model_dump 用于追加到 messages
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (tool_calls or [])
        ],
    }
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_tool_call(tool_id, name, arguments):
    """构造模拟的 tool call"""
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_executor():
    """模拟工具执行器"""
    executor = AsyncMock()
    executor.execute.return_value = {"ok": True, "data": {"result": "tool_output"}, "error": None}
    return executor


@pytest.fixture
def mock_tools():
    """模拟 tools schema 列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]


# ── 测试用例 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_no_tool_call_direct_return(mock_sem, mock_api, mock_executor, mock_tools):
    """无 tool_calls 时直接返回 content"""
    # 信号量 mock
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    # API 返回无 tool_calls 的响应
    mock_api.return_value = make_mock_response(content="最终答案")

    messages = [{"role": "user", "content": "测试问题"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=3, max_tool_calls=5,
    )

    assert isinstance(result, ToolCallResult)
    assert result.content == "最终答案"
    assert result.degraded is False
    assert len(result.tool_calls) == 0
    # 执行器不应被调用
    mock_executor.execute.assert_not_called()


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_single_tool_call(mock_sem, mock_api, mock_executor, mock_tools):
    """单轮工具调用 → 执行工具 → 返回最终结果"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc = make_tool_call("tc_1", "test_tool", '{"query": "test"}')

    # 第一次调用返回 tool_call，第二次返回最终结果
    mock_api.side_effect = [
        make_mock_response(content=None, tool_calls=[tc]),
        make_mock_response(content="基于工具结果的最终答案"),
    ]

    messages = [{"role": "user", "content": "测试问题"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=3, max_tool_calls=5,
    )

    assert result.content == "基于工具结果的最终答案"
    assert result.degraded is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "test_tool"
    mock_executor.execute.assert_called_once()


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_multi_round_tool_calls(mock_sem, mock_api, mock_executor, mock_tools):
    """多轮工具调用"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc1 = make_tool_call("tc_1", "test_tool", '{"query": "first"}')
    tc2 = make_tool_call("tc_2", "test_tool", '{"query": "second"}')

    mock_api.side_effect = [
        make_mock_response(content=None, tool_calls=[tc1]),
        make_mock_response(content=None, tool_calls=[tc2]),
        make_mock_response(content="多轮后最终答案"),
    ]

    messages = [{"role": "user", "content": "测试"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=5, max_tool_calls=10,
    )

    assert result.content == "多轮后最终答案"
    assert result.degraded is False
    assert len(result.tool_calls) == 2
    assert mock_executor.execute.call_count == 2


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_max_rounds_exceeded(mock_sem, mock_api, mock_executor, mock_tools):
    """超过最大轮数返回 degraded=True"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc = make_tool_call("tc_1", "test_tool", '{"query": "test"}')

    # 每次都返回 tool_call，永远不停止
    mock_api.return_value = make_mock_response(content=None, tool_calls=[tc])

    messages = [{"role": "user", "content": "测试"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=2, max_tool_calls=10,
    )

    assert result.degraded is True
    assert "轮数" in result.error or "round" in result.error.lower() or result.error is not None


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_max_calls_exceeded(mock_sem, mock_api, mock_executor, mock_tools):
    """超过最大调用数返回 degraded=True 或 budget_exceeded trace"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc = make_tool_call("tc_1", "test_tool", '{"query": "test"}')

    # 每轮返回 tool_call
    mock_api.side_effect = [
        make_mock_response(content=None, tool_calls=[tc]),
        make_mock_response(content=None, tool_calls=[tc]),
        make_mock_response(content=None, tool_calls=[tc]),
    ]

    messages = [{"role": "user", "content": "测试"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=3, max_tool_calls=1,  # 最多1次调用
    )

    # 应该有 budget_exceeded 的 trace
    budget_traces = [t for t in result.tool_calls if t.status == "budget_exceeded"]
    assert len(budget_traces) >= 1


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_tool_execution_error(mock_sem, mock_api, mock_executor, mock_tools):
    """工具执行失败返回结构化错误给 LLM"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc = make_tool_call("tc_1", "test_tool", '{"query": "test"}')

    mock_api.side_effect = [
        make_mock_response(content=None, tool_calls=[tc]),
        make_mock_response(content="工具失败后的降级回答"),
    ]
    mock_executor.execute.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "execution_error", "message": "Tool failed"},
    }

    messages = [{"role": "user", "content": "测试"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=3, max_tool_calls=5,
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].status == "error"


@pytest.mark.asyncio
@patch("app.services.qwen_client._call_qwen_api_with_tools", new_callable=AsyncMock)
@patch("app.services.qwen_client._get_semaphore")
async def test_invalid_tool_arguments(mock_sem, mock_api, mock_executor, mock_tools):
    """工具参数非法时的处理"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    tc = make_tool_call("tc_1", "test_tool", "invalid json {{{")

    mock_api.side_effect = [
        make_mock_response(content=None, tool_calls=[tc]),
        make_mock_response(content="参数非法后的回答"),
    ]
    # executor 返回 validation_error
    mock_executor.execute.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "validation_error", "message": "Invalid args"},
    }

    messages = [{"role": "user", "content": "测试"}]
    result = await call_qwen_with_tools(
        messages, tools=mock_tools, tool_executor=mock_executor,
        max_tool_rounds=3, max_tool_calls=5,
    )

    # 应该记录了 error trace
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].status == "error"


@pytest.mark.asyncio
@patch("app.services.qwen_client.client")
@patch("app.services.qwen_client._get_semaphore")
async def test_call_qwen_chat_unchanged(mock_sem, mock_client):
    """验证 call_qwen_chat() 行为不变（回归）"""
    sem = AsyncMock()
    sem.acquire.return_value = None
    sem._value = 10
    mock_sem.return_value = sem

    # Mock API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "chat response"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "你好"}]
    result = await call_qwen_chat(messages, temperature=0.5, max_tokens=100)

    assert result == "chat response"
    mock_client.chat.completions.create.assert_called_once()
