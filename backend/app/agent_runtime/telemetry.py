"""Unified agent telemetry: privacy-safe event recording with repository-backed persistence."""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import SecretStr


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


# Stable error codes for coach safety gate
COACH_POLICY_BLOCKED = "COACH_POLICY_BLOCKED"
COACH_TIMEOUT = "COACH_TIMEOUT"
COACH_UNAVAILABLE = "COACH_UNAVAILABLE"


# Bounded metric labels — never include user/session/consultation IDs
AGENT_METRIC_LABELS: dict[str, list[str]] = {
    "agent_node_duration_seconds": ["agent_name", "node_name", "status"],
    "agent_decisions_total": ["agent_name", "intent", "status"],
    "agent_policy_blocks_total": ["agent_name", "reason"],
    "agent_context_tokens": ["agent_name"],
    "agent_memory_reads_total": ["skill_dimension"],
    "agent_feedback_total": ["feedback_value"],
    "coach_safety_checks_total": ["check_category", "result"],
    "coach_safety_gate_decisions": ["decision"],
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
        event_type_map = {
            "node_started": "node_finished",
            "model_started": "model_finished",
            "tool_started": "tool_finished",
        }
        actual_type = event_type_map.get(self.event_type, self.event_type)

        self.recorder._record_sync(
            actual_type,
            agent_name=self.agent_name,
            node_name=self.node_name,
            output_data=output_data,
            status=status,
            error_code=error_code,
        )


class AgentEventRecorder:
    """Privacy-safe agent event recorder with repository-backed persistence.

    - Requires SecretStr HMAC key (no default-hmac-key allowed).
    - Persists hash NOT content: input_payload=None, input_hmac is 64-char SHA-256.
    - Transactional sequence allocation unique per trace.
    """

    def __init__(
        self,
        *,
        session_id: str,
        trace_id: str,
        hmac_key: SecretStr,
        repository: Any | None = None,
        capture_content: bool = False,
    ) -> None:
        # Reject default/weak keys
        raw_key = hmac_key.get_secret_value()
        if not raw_key or raw_key == "default-hmac-key":
            raise ValueError(
                "hmac_key must be a non-empty secret; 'default-hmac-key' is forbidden"
            )
        self.session_id = session_id
        self.trace_id = trace_id
        self.hmac_key = raw_key
        self.repository = repository
        self.capture_content = capture_content
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
        """Synchronous internal record method."""
        seq = self._next_sequence()
        input_hmac, input_chars = self._compute_hmac(input_data)
        output_hmac, output_chars = self._compute_hmac(output_data)

        event = RecordedEvent(
            sequence=seq,
            event_type=event_type,
            agent_name=agent_name,
            node_name=node_name,
            status=status,
            # Never persist raw content — always None
            input_payload=None,
            output_payload=None,
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
        """Record an agent event, persisting to repository if available."""
        event = self._record_sync(
            event_type,
            agent_name=agent_name,
            node_name=node_name,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error_code=error_code,
            metadata=metadata,
        )

        # Persist to repository if available
        if self.repository is not None:
            await self.repository.append_event(
                trace_id=self.trace_id,
                session_id=self.session_id,
                event_type=event_type,
                status=status or "finished",
                agent_name=agent_name,
                node_name=node_name,
                input_payload=None,  # Never persist raw content
                output_payload=None,
                input_hmac=event.input_hmac,
                output_hmac=event.output_hmac,
                error_code=error_code,
                metadata_json=metadata,
            )

        return event

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
