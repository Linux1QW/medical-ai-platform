# -*- coding: utf-8 -*-
"""run_agent 工作器节点的运行时审计测试 — 耗时记录与 NodeError 发射"""

from datetime import datetime
from unittest.mock import patch

import pytest

from app.orchestration.graph import run_agent
from app.orchestration.state import AgentResultEnvelope, EvaluationContext


class _StubAdapter:
    """返回预设 envelope 或抛出预设异常的适配器桩"""

    def __init__(self, envelope=None, exc=None):
        self._envelope = envelope
        self._exc = exc

    async def run(self, context):
        if self._exc is not None:
            raise self._exc
        return self._envelope


def _state(agent_name="inquiry"):
    return {
        "agent_name": agent_name,
        "step_id": f"step_{agent_name}",
        "context": EvaluationContext(),
        "run_id": "run-test",
    }


@pytest.mark.asyncio
async def test_success_records_duration_and_no_errors():
    envelope = AgentResultEnvelope(agent_name="inquiry", status="success", score=80)
    with patch(
        "app.orchestration.adapters.registry.get_adapter",
        return_value=_StubAdapter(envelope=envelope),
    ):
        result = await run_agent(_state())

    assert result["node_errors"] == []
    exec_result = result["execution_results"][0]
    assert exec_result.duration_ms is not None and exec_result.duration_ms >= 0
    # 时间戳为 ISO 字符串（保证 JSON 可序列化），且可解析
    assert datetime.fromisoformat(exec_result.started_at) <= datetime.fromisoformat(
        exec_result.finished_at
    )


@pytest.mark.asyncio
async def test_adapter_exception_emits_node_error():
    with patch(
        "app.orchestration.adapters.registry.get_adapter",
        return_value=_StubAdapter(exc=ValueError("bad output")),
    ):
        result = await run_agent(_state("humanistic"))

    envelope = result["agent_results"][0]
    assert envelope.status == "error"
    errors = result["node_errors"]
    assert len(errors) == 1
    assert errors[0].node_name == "run_agent:humanistic"
    assert errors[0].error_type == "ValueError"
    assert "bad output" in errors[0].error_message
    # 错误路径同样记录耗时
    assert result["execution_results"][0].duration_ms is not None


@pytest.mark.asyncio
async def test_adapter_internal_error_envelope_also_recorded():
    """适配器内部兜底返回 status=error 时也应记入 node_errors，保证 trace 树完整"""
    envelope = AgentResultEnvelope(
        agent_name="diagnosis",
        status="error",
        analysis="Agent执行异常: RuntimeError",
        review_reason="diagnosis_error: boom",
    )
    with patch(
        "app.orchestration.adapters.registry.get_adapter",
        return_value=_StubAdapter(envelope=envelope),
    ):
        result = await run_agent(_state("diagnosis"))

    errors = result["node_errors"]
    assert len(errors) == 1
    assert errors[0].node_name == "run_agent:diagnosis"
    assert errors[0].error_type == "AgentError"
    assert "diagnosis_error" in errors[0].error_message
