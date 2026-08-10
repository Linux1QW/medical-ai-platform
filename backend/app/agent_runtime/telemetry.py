"""Unified agent telemetry: privacy-safe event recording and bounded metrics."""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentEventType(str, Enum):
    """Fixed set of agent event types."""
    SESSION_STARTED = "session_started"
    NODE_STARTED = "node_started"
    NODE_FINISHED = "node_finished"
    MODEL_STARTED = "model_started"
    MODEL_FINISHED = "model_finished"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    POLICY_BLOCKED = "policy_blocked"
    MEMORY_READ = "memory_read"
    MEMORY_CANDIDATE = "memory_candidate"
    FEEDBACK_RECEIVED = "feedback_received"
    SESSION_FINISHED = "session_finished"


class AgentEventStatus(str, Enum):
    """Status of an agent event."""
    STARTED = "started"
    FINISHED = "finished"
    ERROR = "error"
    BLOCKED = "blocked"


# Bounded metric labels — never include user/session/consultation IDs
AGENT_METRIC_LABELS: dict[str, list[str]] = {
    "agent_node_duration_seconds": ["agent_name", "node_name", "status"],
    "agent_decisions_total": ["agent_name", "intent", "status"],
    "agent_policy_blocks_total": ["agent_name", "reason"],
    "agent_context_tokens": ["agent_name"],
    "agent_memory_reads_total": ["skill_dimension"],
    "agent_feedback_total": ["feedback_value"],
}


def compute_hmac(key: str, data: str) -> str:
    """Compute HMAC-SHA256 for content fingerprinting."""
    return hmac.new(
        key.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@dataclass
class RecordedEvent:
    """A recorded agent event."""
    sequence: int
    event_type: str
    agent_name: str | None = None
    node_name: str | None = None
    status: str | None = None
    input_payload: Any = None
    output_payload: Any = None
    input_hmac: str | None = None
    output_hmac: str | None = None
    input_char_count: int | None = None
    output_char_count: int | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentSpan:
    """A span tracking start/finish of an agent operation."""
    recorder: AgentEventRecorder
    sequence: int
    event_type: str
    agent_name: str
    node_name: str | None = None
    _started: bool = False

    def finish(
        self,
        status: str,
        *,
        output_data: Any = None,
        error_code: str | None = None,
    ) -> None:
        """Finish the span, recording the completion event."""
        # Map start event types to their corresponding finish types
        event_type_map = {
            "node_started": "node_finished",
            "model_started": "model_finished",
            "tool_started": "tool_finished",
        }
        actual_type = event_type_map.get(self.event_type, self.event_type)

        # Directly record synchronously to ensure the event is appended immediately
        self.recorder._record_sync(
            actual_type,
            agent_name=self.agent_name,
            node_name=self.node_name,
            output_data=output_data,
            status=status,
            error_code=error_code,
        )


class AgentEventRecorder:
    """Privacy-safe agent event recorder.

    When capture_content=False (default), stores only HMAC and character count.
    All events are append-only with monotonic sequence numbers.
    """

    def __init__(
        self,
        *,
        session_id: str,
        trace_id: str,
        capture_content: bool = False,
        hmac_key: str = "default-hmac-key",
    ) -> None:
        self.session_id = session_id
        self.trace_id = trace_id
        self.capture_content = capture_content
        self.hmac_key = hmac_key
        self._sequence = 0
        self._events: list[RecordedEvent] = []

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _compute_hmac(self, data: Any) -> tuple[str | None, int | None]:
        """Compute HMAC and char count for data."""
        if data is None:
            return None, None
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)
        char_count = len(serialized)
        hmac_value = compute_hmac(self.hmac_key, serialized)
        return hmac_value, char_count

    def _record_sync(
        self,
        event_type: str,
        *,
        agent_name: str | None = None,
        node_name: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        status: str | None = None,
        error_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecordedEvent:
        """Synchronous internal record method for use by AgentSpan.finish()."""
        seq = self._next_sequence()
        input_hmac, input_chars = self._compute_hmac(input_data)
        output_hmac, output_chars = self._compute_hmac(output_data)

        event = RecordedEvent(
            sequence=seq,
            event_type=event_type,
            agent_name=agent_name,
            node_name=node_name,
            status=status,
            input_payload=input_data if self.capture_content else None,
            output_payload=output_data if self.capture_content else None,
            input_hmac=input_hmac,
            output_hmac=output_hmac,
            input_char_count=input_chars,
            output_char_count=output_chars,
            error_code=error_code,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    async def record(
        self,
        event_type: str,
        *,
        agent_name: str | None = None,
        node_name: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        status: str | None = None,
        error_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecordedEvent:
        """Record an agent event."""
        return self._record_sync(
            event_type,
            agent_name=agent_name,
            node_name=node_name,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error_code=error_code,
            metadata=metadata,
        )

    async def list_events(self) -> list[RecordedEvent]:
        """List all recorded events (append-only)."""
        return list(self._events)

    def start_span(
        self,
        event_type: str,
        *,
        agent_name: str,
        node_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSpan:
        """Start a new span for tracking operation lifecycle."""
        seq = self._next_sequence()
        event = RecordedEvent(
            sequence=seq,
            event_type=event_type,
            agent_name=agent_name,
            node_name=node_name,
            status="started",
            metadata=metadata or {},
        )
        self._events.append(event)
        return AgentSpan(
            recorder=self,
            sequence=seq,
            event_type=event_type,
            agent_name=agent_name,
            node_name=node_name,
        )
