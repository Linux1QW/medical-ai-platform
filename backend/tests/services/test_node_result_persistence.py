# -*- coding: utf-8 -*-
"""run 级 trace 树构建测试 — _build_node_results 从 final state 生成审计行"""

from app.orchestration.state import (
    AgentResultEnvelope,
    ExecutionResult,
    NodeError,
)
from app.services.evaluation_service import _build_node_results

RUN_ID = "11111111-2222-3333-4444-555555555555"


def _exec_result(agent_name, status="success", trace=None, **kwargs):
    envelope = AgentResultEnvelope(
        agent_name=agent_name,
        status=status,
        trace=trace or {},
    )
    return ExecutionResult(
        step_id=f"step_{agent_name}",
        agent_name=agent_name,
        status=status,
        envelope=envelope,
        **kwargs,
    )


def test_empty_state_produces_no_rows():
    assert _build_node_results(RUN_ID, {}) == []


def test_execution_results_become_agent_rows_with_trace():
    state = {
        "execution_results": [
            _exec_result(
                "knowledge",
                trace={"tool_trace": [{"tool": "rag_search", "calls": 2}]},
                duration_ms=1234,
                started_at="2026-05-26T10:00:00",
                finished_at="2026-05-26T10:00:01",
            ),
            _exec_result("inquiry", duration_ms=500),
        ],
    }
    rows = _build_node_results(RUN_ID, state)

    assert len(rows) == 2
    knowledge_row = next(r for r in rows if r.node_name == "agent:knowledge")
    assert knowledge_row.run_id == RUN_ID
    assert knowledge_row.status == "success"
    assert knowledge_row.duration_ms == 1234
    # 工具调用明细随 trace 落库（不再仅 knowledge 一路进 Evaluation.rag_trace_data）
    assert knowledge_row.result_summary["trace"]["tool_trace"][0]["tool"] == "rag_search"
    assert knowledge_row.started_at is not None
    assert knowledge_row.finished_at is not None


def test_error_agent_row_carries_error_type_without_duplicate_row():
    state = {
        "execution_results": [_exec_result("diagnosis", status="error")],
        "node_errors": [
            NodeError(
                node_name="run_agent:diagnosis",
                error_type="TimeoutError",
                error_message="llm timeout",
            )
        ],
    }
    rows = _build_node_results(RUN_ID, state)

    # agent 行已覆盖该错误，不再产生独立 error 行
    assert len(rows) == 1
    assert rows[0].node_name == "agent:diagnosis"
    assert rows[0].status == "error"
    assert rows[0].error_type == "TimeoutError"


def test_agent_results_fallback_when_no_execution_results():
    """旧 dispatch_and_run 路径只产生 agent_results，也应落审计行"""
    state = {
        "agent_results": [
            AgentResultEnvelope(agent_name="humanistic", status="success", score=85),
        ],
    }
    rows = _build_node_results(RUN_ID, state)

    assert len(rows) == 1
    assert rows[0].node_name == "agent:humanistic"
    assert rows[0].result_summary["score"] == 85
    assert rows[0].duration_ms is None


def test_non_agent_node_error_gets_own_row():
    state = {
        "execution_results": [_exec_result("inquiry")],
        "node_errors": [
            NodeError(
                node_name="aggregate_results",
                error_type="KeyError",
                error_message="missing dimension",
                attempt=2,
            )
        ],
    }
    rows = _build_node_results(RUN_ID, state)

    assert len(rows) == 2
    error_row = next(r for r in rows if r.node_name == "aggregate_results")
    assert error_row.status == "error"
    assert error_row.attempt == 2
    assert error_row.error_type == "KeyError"
    assert error_row.result_summary == {"error_message": "missing dimension"}


def test_invalid_timestamp_string_is_tolerated():
    state = {
        "execution_results": [
            _exec_result("inquiry", started_at="not-a-timestamp"),
        ],
    }
    rows = _build_node_results(RUN_ID, state)
    assert rows[0].started_at is None
