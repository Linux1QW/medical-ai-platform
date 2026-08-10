"""Tests for agent telemetry."""
import pytest
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.agent_runtime.telemetry import (
    AgentEventRecorder,
    AgentEventType,
    AgentEventStatus,
    AgentSpan,
    compute_hmac,
)


@pytest.fixture
def recorder():
    """Create a test recorder with in-memory storage."""
    return AgentEventRecorder(
        session_id="test-session",
        trace_id="test-trace",
        capture_content=False,
    )


def test_event_types_are_bounded_enum():
    """All event types are fixed enum values."""
    expected = {
        "session_started", "node_started", "node_finished",
        "model_started", "model_finished",
        "tool_started", "tool_finished",
        "policy_blocked",
        "memory_read", "memory_candidate",
        "feedback_received", "session_finished",
    }
    actual = {e.value for e in AgentEventType}
    assert expected == actual


@pytest.mark.asyncio
async def test_recorder_is_append_only_and_redacted(recorder):
    """Events are append-only with monotonic sequence and content redacted by default."""
    await recorder.record(
        "model_started",
        agent_name="coach_intent",
        input_data={"patient": "张三", "phone": "13812345678"},
    )
    await recorder.record(
        "model_finished",
        agent_name="coach_intent",
        output_data={"text": "患者张三"},
    )
    events = await recorder.list_events()
    assert [e.sequence for e in events] == [1, 2]
    assert all(e.input_payload is None and e.output_payload is None for e in events)
    # HMAC should be present when data was provided
    assert events[0].input_hmac is not None  # model_started had input_data
    assert events[1].output_hmac is not None  # model_finished had output_data


@pytest.mark.asyncio
async def test_recorder_computes_hmac_not_raw_content(recorder):
    """When capture_content=False, store HMAC instead of raw content."""
    await recorder.record(
        "node_finished",
        agent_name="coach_planner",
        output_data={"plan": "ask about onset"},
    )
    events = await recorder.list_events()
    assert len(events) == 1
    event = events[0]
    assert event.output_payload is None
    assert event.output_hmac is not None
    assert event.output_char_count is not None


@pytest.mark.asyncio
async def test_span_lifecycle(recorder):
    """AgentSpan tracks start/finish with proper status."""
    span = recorder.start_span(
        "node_started",
        agent_name="coach_evidence",
        node_name="search_rubric",
    )
    assert span.sequence == 1
    span.finish("finished", output_data={"citations": ["ref-1"]})
    events = await recorder.list_events()
    assert len(events) == 2  # start + finish
    assert events[0].event_type == "node_started"
    assert events[1].event_type == "node_finished"


def test_compute_hmac_is_deterministic():
    """HMAC computation is deterministic."""
    h1 = compute_hmac("test-key", "hello world")
    h2 = compute_hmac("test-key", "hello world")
    assert h1 == h2
    h3 = compute_hmac("test-key", "different")
    assert h1 != h3


@pytest.mark.asyncio
async def test_metrics_have_bounded_labels():
    """Metrics use bounded enum labels, never user/session IDs."""
    from app.agent_runtime.telemetry import AGENT_METRIC_LABELS
    # No user/session/consultation ID labels
    for label_set in AGENT_METRIC_LABELS.values():
        assert "user_id" not in label_set
        assert "session_id" not in label_set
        assert "consultation_id" not in label_set
