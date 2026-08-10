"""Tests for the LangGraph-based multi-agent coach graph.

Updated to use the new LangGraph runtime (replaces hand-written CoachGraph).
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agent_runtime.contracts import (
    CoachContextView,
    CoachSuggestion,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.agent_runtime.critic import CriticAgent
from app.agent_runtime.evidence import EvidenceAgent
from app.agent_runtime.graph import (
    CoachDependencies,
    build_coach_graph,
    invoke_coach_graph,
)
from app.agent_runtime.state import CoachGraphState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_test_view(**overrides: Any) -> CoachContextView:
    defaults: dict[str, Any] = dict(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(age=45, gender="男", chief_complaint="头痛"),
        messages=[VisibleMessage(sequence=1, role="doctor", content="你好，哪里不舒服？")],
    )
    defaults.update(overrides)
    return CoachContextView(**defaults)


def _make_state(**overrides: Any) -> CoachGraphState:
    defaults: dict[str, Any] = dict(
        context=_make_test_view(),
        latest_message="你好，哪里不舒服？",
        turn=1,
    )
    defaults.update(overrides)
    return CoachGraphState(**defaults)


# ── Graph pipeline tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_full_pipeline() -> None:
    """Full pipeline with valid context produces a suggestion."""
    graph = build_coach_graph()
    view = _make_test_view()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="你好", turn=1, timeout_seconds=10,
    )

    assert result["blocked"] is False
    assert result["final_suggestion"] is not None
    assert result["final_suggestion"].intent == "rapport"
    assert result["status"] == "done"


@pytest.mark.asyncio
async def test_graph_blocks_unsafe_intent() -> None:
    """Message classified as unsafe → blocked."""
    graph = build_coach_graph()
    view = _make_test_view()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="我想伤害自己", turn=1, timeout_seconds=10,
    )

    assert result["intent_result"] is not None
    assert result["intent_result"].intent == "unsafe"
    assert result["final_suggestion"] is None


@pytest.mark.asyncio
async def test_graph_intent_classification() -> None:
    """'你好' → intent='rapport'."""
    graph = build_coach_graph()
    view = _make_test_view()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="你好", turn=1, timeout_seconds=10,
    )

    assert result["intent_result"].intent == "rapport"
    assert result["blocked"] is False


@pytest.mark.asyncio
async def test_graph_evidence_fn_called() -> None:
    """Provide evidence_fn, verify it's called."""
    called_with: list[tuple[str, str]] = []

    def evidence_fn(intent: str, message: str) -> list[dict[str, Any]]:
        called_with.append((intent, message))
        return [{"source": "rubric", "text": "evidence"}]

    deps = CoachDependencies(evidence_fn=evidence_fn)
    graph = build_coach_graph(dependencies=deps)
    view = _make_test_view()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="你好", turn=1, timeout_seconds=10,
    )

    assert len(called_with) == 1
    assert called_with[0][0] == "rapport"
    assert result["status"] == "done"


@pytest.mark.asyncio
async def test_graph_node_trace_recorded() -> None:
    """Verify trace_refs has entries for each node."""
    graph = build_coach_graph()
    view = _make_test_view()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="你好", turn=1, timeout_seconds=10,
    )

    refs = result.get("trace_refs", [])
    # Should have at least intent and persist refs
    assert any("intent" in r for r in refs)
    assert any("persist" in r for r in refs)


@pytest.mark.asyncio
async def test_graph_empty_messages() -> None:
    """Context with no messages still works."""
    view = _make_test_view(messages=[])
    graph = build_coach_graph()
    result = await invoke_coach_graph(
        graph, context=view, latest_message="你好", turn=1, timeout_seconds=10,
    )

    assert result["blocked"] is False
    assert result["status"] == "done"
    assert result["final_suggestion"] is not None


# ── Critic agent tests ────────────────────────────────────────────────────────


def test_critic_evaluate_passes_clean() -> None:
    """Clean suggestion passes critic."""
    critic = CriticAgent()
    suggestion = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=uuid4(),
        turn_no=1,
        intent="rapport",
        stage="rapport",
        suggested_question="[Coach] Consider asking about: rapport",
        rationale_summary="Intent: rapport | Plan: Build rapport",
        confidence=0.8,
        risk_level="low",
    )
    result = critic.evaluate(suggestion)
    assert result.passed is True
    assert result.findings == []


def test_critic_evaluate_fails_hidden_leak() -> None:
    """Suggestion with hidden pattern fails critic."""
    critic = CriticAgent()
    suggestion = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=uuid4(),
        turn_no=1,
        intent="rapport",
        stage="rapport",
        suggested_question="The gold_standard is hidden here",
        rationale_summary="Intent: rapport",
        confidence=0.8,
        risk_level="low",
    )
    result = critic.evaluate(suggestion)
    assert result.passed is False
    assert any(f.category == "hidden_leak" for f in result.findings)


# ── Evidence agent tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evidence_agent_blocked_skills() -> None:
    """EvidenceAgent rejects non-allowed skill."""
    agent = EvidenceAgent()
    result = await agent.search("dangerous_tool", "test query")
    assert "error" in result
    assert "not allowed" in result["error"]
