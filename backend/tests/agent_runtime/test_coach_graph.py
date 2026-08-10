"""Tests for the deterministic multi-agent coach graph."""
from __future__ import annotations

import asyncio
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
from app.agent_runtime.graph import CoachGraph, CoachGraphState

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
        context_view=_make_test_view(),
        latest_message="你好，哪里不舒服？",
        turn_no=1,
    )
    defaults.update(overrides)
    return CoachGraphState(**defaults)


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Graph pipeline tests ──────────────────────────────────────────────────────


def test_graph_full_pipeline() -> None:
    """Full pipeline with valid context produces a suggestion."""
    graph = CoachGraph()
    state = _make_state()
    result = _run(graph.run(state))

    assert result.blocked is False
    assert result.final_suggestion is not None
    assert result.final_suggestion.intent == "rapport"
    assert result.final_suggestion.session_id == state.session_id
    assert result.final_suggestion.turn_no == 1


def test_graph_blocks_unsafe_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Message classified as unsafe → blocked."""
    import app.agent_runtime.graph as graph_mod

    monkeypatch.setattr(graph_mod, "classify_intent", lambda msg, **kw: "unsafe")

    graph = CoachGraph()
    state = _make_state(latest_message="some unsafe message")
    result = _run(graph.run(state))

    assert result.blocked is True
    assert "Unsafe intent" in result.block_reason


def test_graph_intent_classification() -> None:
    """'你好' → intent='rapport'."""
    graph = CoachGraph()
    state = _make_state(latest_message="你好")
    result = _run(graph.run(state))

    assert result.intent == "rapport"
    assert result.blocked is False


def test_graph_safety_blocks_hidden_leak() -> None:
    """Working memory with non-visible source → blocked."""
    from app.services.memory.working import SlotObservation, WorkingMemoryState

    wm = WorkingMemoryState()
    # Add a slot with source_sequence=99 — not in visible messages (only seq=1 exists)
    wm.apply(
        SlotObservation(
            key="test_slot",
            value="hidden_value",
            polarity="positive",
            turn=1,
            source_sequence=99,
        )
    )

    graph = CoachGraph()
    state = _make_state(working_memory=wm)
    result = _run(graph.run(state))

    assert result.blocked is True
    assert "Hidden context violation" in result.block_reason


def test_graph_critic_blocks_diagnostic() -> None:
    """Draft containing diagnostic phrasing → blocked by critic node."""
    graph = CoachGraph()
    state = _make_state()

    # Manually set up a draft with diagnostic phrasing
    state.intent = "rapport"
    state.draft_suggestion = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=state.session_id,
        turn_no=1,
        intent="rapport",
        stage="rapport",
        suggested_question="你应该诊断这是感冒病",
        rationale_summary="Intent: rapport",
        confidence=0.7,
        risk_level="low",
    )

    # Run only critic → finalize → persist
    result = graph._critic_node(state)
    assert result.blocked is True
    assert "diagnostic phrasing" in result.block_reason


def test_graph_critic_blocks_hidden_fact() -> None:
    """Draft containing 'expected_diagnosis' → blocked by critic node."""
    graph = CoachGraph()
    state = _make_state()

    state.intent = "rapport"
    state.draft_suggestion = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=state.session_id,
        turn_no=1,
        intent="rapport",
        stage="rapport",
        suggested_question="Consider the expected_diagnosis here",
        rationale_summary="Intent: rapport",
        confidence=0.7,
        risk_level="low",
    )

    result = graph._critic_node(state)
    assert result.blocked is True
    assert "hidden-fact leakage" in result.block_reason


def test_graph_evidence_fn_called() -> None:
    """Provide evidence_fn, verify it's called."""
    called_with: list[tuple[str, str]] = []

    def evidence_fn(intent: str, message: str) -> list[dict[str, Any]]:
        called_with.append((intent, message))
        return [{"source": "rubric", "text": "evidence"}]

    graph = CoachGraph(evidence_fn=evidence_fn)
    state = _make_state()
    result = _run(graph.run(state))

    assert len(called_with) == 1
    assert called_with[0][0] == "rapport"
    assert result.evidence_results == [{"source": "rubric", "text": "evidence"}]


def test_graph_node_trace_recorded() -> None:
    """Verify node_trace has entries for each node."""
    graph = CoachGraph()
    state = _make_state()
    result = _run(graph.run(state))

    node_names = [entry["node"] for entry in result.node_trace if entry["status"] == "started"]
    expected = [
        "load_context",
        "intent",
        "memory",
        "safety",
        "planner",
        "evidence",
        "draft",
        "critic",
        "finalize",
        "persist",
    ]
    assert node_names == expected


def test_graph_empty_messages() -> None:
    """Context with no messages still works."""
    view = _make_test_view(messages=[])
    graph = CoachGraph()
    state = _make_state(context_view=view, latest_message="你好")
    result = _run(graph.run(state))

    assert result.blocked is False
    assert result.compiled_context is not None
    assert result.final_suggestion is not None


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


def test_evidence_agent_blocked_skills() -> None:
    """EvidenceAgent rejects non-allowed skill."""
    agent = EvidenceAgent()
    result = agent.search("dangerous_tool", "test query")
    assert "error" in result
    assert "not allowed" in result["error"]
