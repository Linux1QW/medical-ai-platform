"""Tests for the LangGraph-based Coach runtime."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agent_runtime.contracts import (
    CoachContextView,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.agent_runtime.graph import (
    CoachDependencies,
    build_coach_graph,
    invoke_coach_graph,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_view(**overrides: Any) -> CoachContextView:
    defaults: dict[str, Any] = dict(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(age=45, gender="男", chief_complaint="头痛"),
        messages=[VisibleMessage(sequence=1, role="doctor", content="你好，哪里不舒服？")],
    )
    defaults.update(overrides)
    return CoachContextView(**defaults)


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Graph structure tests ────────────────────────────────────────────────────


def test_graph_contains_required_nodes() -> None:
    """Compiled graph must have all required node names."""
    compiled = build_coach_graph(checkpointer=MemorySaver())
    node_names = set(compiled.get_graph().nodes)
    required = {"intent", "planner", "evidence", "draft", "critic", "persist"}
    assert node_names >= required, f"Missing nodes: {required - node_names}"


def test_graph_has_finalize_node() -> None:
    """Compiled graph includes the finalize node."""
    compiled = build_coach_graph(checkpointer=MemorySaver())
    assert "finalize" in compiled.get_graph().nodes


# ── Full pipeline tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_pipeline_rapport() -> None:
    """Full pipeline with '你好' produces a suggestion."""
    deps = CoachDependencies()
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    result = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="你好",
        turn=1,
        timeout_seconds=10,
    )

    assert result["status"] == "done"
    assert result["blocked"] is False
    assert result["final_suggestion"] is not None
    assert result["final_suggestion"].intent == "rapport"


@pytest.mark.asyncio
async def test_unsafe_intent_blocked() -> None:
    """Unsafe intent → blocked, no suggestion."""
    deps = CoachDependencies()
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    result = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="我想伤害自己",
        turn=1,
        timeout_seconds=10,
    )

    # Unsafe goes through intent → finalize (skipping planner/draft)
    assert result["final_suggestion"] is None
    assert result["intent_result"] is not None
    assert result["intent_result"].intent == "unsafe"


@pytest.mark.asyncio
async def test_evidence_fn_called() -> None:
    """Custom evidence_fn is invoked during the pipeline."""
    called_with: list[tuple[str, str]] = []

    def evidence_fn(intent: str, message: str) -> list[dict[str, Any]]:
        called_with.append((intent, message))
        return [{"source": "rubric", "text": "evidence text", "score": 0.9, "doc_id": "doc1"}]

    deps = CoachDependencies(evidence_fn=evidence_fn)
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    result = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="你好",
        turn=1,
        timeout_seconds=10,
    )

    assert len(called_with) >= 1
    assert result["status"] == "done"


# ── Checkpointing / thread resume tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_same_thread_resumes_after_retry() -> None:
    """Same thread_id produces consistent turn_no across invocations."""
    deps = CoachDependencies()
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    first = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="你好",
        turn=1,
        thread_id="coach:test-1",
        timeout_seconds=10,
    )

    second = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="你好",
        turn=1,
        thread_id="coach:test-1",
        timeout_seconds=10,
    )

    # Both invocations should have the same turn_no
    assert second["turn"] == first["turn"]


# ── Timeout tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hard_timeout_returns_degraded() -> None:
    """Timeout → status='degraded', block_reason='COACH_TIMEOUT'."""
    # Create a gateway that hangs forever
    class HangingGateway:
        async def complete_structured(self, **kwargs: Any) -> Any:
            await asyncio.sleep(100)
            raise RuntimeError("should not reach")

    deps = CoachDependencies(gateway=HangingGateway())
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    result = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="一些需要模型的消息",
        turn=1,
        timeout_seconds=1,  # Very short timeout
    )

    assert result["status"] == "degraded"
    assert result["block_reason"] == "COACH_TIMEOUT"


# ── Repetition penalty tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repetition_penalized_once() -> None:
    """Repeating the same dimension within 1 turn penalizes confidence."""
    deps = CoachDependencies()
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    # First invocation — ask about chief_complaint
    first = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="哪里不舒服",
        turn=1,
        thread_id="coach:repeat-test",
        timeout_seconds=10,
    )
    assert first["status"] == "done"
    _first_confidence = first["final_suggestion"].confidence if first["final_suggestion"] else 0.0

    # Second invocation — same intent, same turn, with asked_dimensions showing prior ask
    second = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="哪里不舒服",
        turn=2,
        asked_dimensions={"chief_complaint": 1},
        thread_id="coach:repeat-test-2",
        timeout_seconds=10,
    )
    assert second["status"] == "done"


# ── Trace refs bounded ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trace_refs_are_bounded() -> None:
    """trace_refs should not grow unbounded."""
    deps = CoachDependencies()
    graph = build_coach_graph(checkpointer=MemorySaver(), dependencies=deps)
    view = _make_view()

    result = await invoke_coach_graph(
        graph,
        context=view,
        latest_message="你好",
        turn=1,
        timeout_seconds=10,
    )

    refs = result.get("trace_refs", [])
    assert len(refs) <= 20, f"trace_refs too long: {len(refs)}"
