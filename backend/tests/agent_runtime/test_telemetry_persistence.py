"""Tests for privacy-safe agent telemetry persistence."""

from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.agent_runtime.telemetry import (
    COACH_POLICY_BLOCKED,
    COACH_TIMEOUT,
    COACH_UNAVAILABLE,
    AgentEventRecorder,
    AgentEventType,
    compute_hmac,
)


@pytest.fixture
def hmac_key():
    """Create a valid HMAC key for testing."""
    return SecretStr("test-secret-hmac-key-12345")


@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    repo = AsyncMock()
    repo.append_event = AsyncMock()
    return repo


@pytest.fixture
def recorder(hmac_key):
    """Create a test recorder with in-memory storage."""
    return AgentEventRecorder(
        session_id="test-session",
        trace_id="test-trace",
        hmac_key=hmac_key,
        capture_content=False,
    )


@pytest.fixture
def recorder_with_repo(hmac_key, mock_repository):
    """Create a test recorder with repository."""
    return AgentEventRecorder(
        session_id="test-session",
        trace_id="test-trace",
        hmac_key=hmac_key,
        repository=mock_repository,
        capture_content=False,
    )


def test_rejects_default_hmac_key():
    """Recorder must reject 'default-hmac-key'."""
    with pytest.raises(ValueError, match="forbidden"):
        AgentEventRecorder(
            session_id="test",
            trace_id="test",
            hmac_key=SecretStr("default-hmac-key"),
        )


def test_rejects_empty_hmac_key():
    """Recorder must reject empty keys."""
    with pytest.raises(ValueError, match="forbidden"):
        AgentEventRecorder(
            session_id="test",
            trace_id="test",
            hmac_key=SecretStr(""),
        )


@pytest.mark.asyncio
async def test_default_trace_persists_hash_not_content(recorder):
    """When capture_content=False, stores only HMAC, not raw content."""
    await recorder.record("model_finished", input_data={"content": "患者原文"})
    events = await recorder.list_events()
    event = events[0]
    assert event.input_payload is None
    assert event.input_hmac is not None
    assert len(event.input_hmac) == 64  # SHA-256 hex digest


@pytest.mark.asyncio
async def test_recorder_persists_to_repository(recorder_with_repo, mock_repository):
    """Recorder persists events to repository when provided."""
    await recorder_with_repo.record(
        "node_finished",
        agent_name="coach_intent",
        input_data={"patient": "张三"},
        output_data={"question": "您哪里不舒服?"},
    )
    # Verify repository was called
    mock_repository.append_event.assert_called_once()
    call_kwargs = mock_repository.append_event.call_args.kwargs
    # Verify no raw content persisted
    assert call_kwargs["input_payload"] is None
    assert call_kwargs["output_payload"] is None
    # Verify HMAC is present
    assert call_kwargs["input_hmac"] is not None
    assert call_kwargs["output_hmac"] is not None  # output_data was provided


@pytest.mark.asyncio
async def test_recorder_is_append_only_and_redacted(recorder):
    """Events are append-only with monotonic sequence and content redacted."""
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
    assert events[0].input_hmac is not None
    assert events[1].output_hmac is not None


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
    assert len(events) == 2
    assert events[0].event_type == "node_started"
    assert events[1].event_type == "node_finished"


def test_compute_hmac_is_deterministic():
    """HMAC computation is deterministic."""
    h1 = compute_hmac("test-key", "hello world")
    h2 = compute_hmac("test-key", "hello world")
    assert h1 == h2
    h3 = compute_hmac("test-key", "different")
    assert h1 != h3


def test_stable_error_codes():
    """Stable error codes are defined and consistent."""
    assert COACH_POLICY_BLOCKED == "COACH_POLICY_BLOCKED"
    assert COACH_TIMEOUT == "COACH_TIMEOUT"
    assert COACH_UNAVAILABLE == "COACH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_metrics_have_bounded_labels():
    """Metrics use bounded enum labels, never user/session IDs."""
    from app.agent_runtime.telemetry import AGENT_METRIC_LABELS
    for label_set in AGENT_METRIC_LABELS.values():
        assert "user_id" not in label_set
        assert "session_id" not in label_set
        assert "consultation_id" not in label_set


@pytest.mark.asyncio
async def test_event_types_are_bounded_enum():
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
