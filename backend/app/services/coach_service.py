"""Coach service: orchestrates the coach graph for API consumption."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from app.agent_runtime.contracts import (
    CoachContextView,
    CoachSuggestion,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.agent_runtime.graph import CoachGraph, CoachGraphState


@dataclass
class CoachSession:
    """In-memory coach session state."""
    session_id: UUID
    consultation_id: int
    doctor_id: int
    turn_no: int = 0
    status: str = "idle"  # idle, thinking, suggestion, degraded, error
    last_suggestion: CoachSuggestion | None = None
    idempotency_cache: dict[str, CoachSuggestion] = field(default_factory=dict)
    feedback_log: list[dict[str, Any]] = field(default_factory=list)


class CoachService:
    """Service layer for coach operations.

    Manages sessions, runs the coach graph, and produces SSE events.
    Coach is gated by COACH_ENABLED environment variable (default: False).
    """

    def __init__(self) -> None:
        self._sessions: dict[int, CoachSession] = {}  # consultation_id → session
        self._enabled: bool = False  # Coach default OFF

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def get_or_create_session(self, consultation_id: int, doctor_id: int) -> CoachSession:
        """Get or create a coach session for a consultation."""
        if consultation_id not in self._sessions:
            self._sessions[consultation_id] = CoachSession(
                session_id=uuid4(),
                consultation_id=consultation_id,
                doctor_id=doctor_id,
            )
        return self._sessions[consultation_id]

    def get_state(self, consultation_id: int) -> dict[str, Any]:
        """Get current coach state for a consultation."""
        session = self._sessions.get(consultation_id)
        if session is None:
            return {
                "consultation_id": consultation_id,
                "status": "disabled" if not self._enabled else "idle",
                "current_stage": None,
                "turn_no": 0,
            }
        return {
            "consultation_id": consultation_id,
            "status": session.status if self._enabled else "disabled",
            "current_stage": None,
            "turn_no": session.turn_no,
        }

    async def stream_suggestion(
        self,
        *,
        consultation_id: int,
        doctor_id: int,
        latest_message: str,
        idempotency_key: str,
        patient_age: int = 45,
        patient_gender: str = "男",
        chief_complaint: str = "问诊",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream coach suggestion via SSE events.

        Events: thinking → suggestion (or error) → done
        Supports idempotency: same key returns cached result.
        Supports Last-Event-ID replay.
        """
        if not self._enabled:
            yield {"event": "error", "data": {"message": "Coach is disabled"}, "id": None}
            yield {"event": "done", "data": {}, "id": None}
            return

        session = self.get_or_create_session(consultation_id, doctor_id)

        # Idempotency check
        if idempotency_key in session.idempotency_cache:
            cached = session.idempotency_cache[idempotency_key]
            yield {"event": "suggestion", "data": cached.model_dump(mode="json"), "id": str(uuid4())}
            yield {"event": "done", "data": {}, "id": None}
            return

        session.status = "thinking"
        session.turn_no += 1
        yield {"event": "thinking", "data": {"turn_no": session.turn_no}, "id": str(uuid4())}

        try:
            # Build context view
            context_view = CoachContextView(
                consultation_id=consultation_id,
                doctor_id=doctor_id,
                visible_patient=VisiblePatientProfile(
                    age=patient_age,
                    gender=patient_gender,
                    chief_complaint=chief_complaint,
                ),
                messages=[
                    VisibleMessage(
                        sequence=session.turn_no,
                        role="doctor",
                        content=latest_message,
                    ),
                ],
            )

            # Run coach graph
            graph = CoachGraph()
            state = CoachGraphState(
                context_view=context_view,
                latest_message=latest_message,
                turn_no=session.turn_no,
                session_id=session.session_id,
            )

            result = await graph.run(state)

            if result.blocked:
                session.status = "degraded"
                yield {"event": "error", "data": {"message": result.block_reason}, "id": str(uuid4())}
            elif result.final_suggestion:
                session.status = "suggestion"
                session.last_suggestion = result.final_suggestion
                session.idempotency_cache[idempotency_key] = result.final_suggestion
                yield {
                    "event": "suggestion",
                    "data": result.final_suggestion.model_dump(mode="json"),
                    "id": str(uuid4()),
                }
            else:
                session.status = "degraded"
                yield {"event": "error", "data": {"message": "No suggestion produced"}, "id": str(uuid4())}

        except Exception as e:
            session.status = "error"
            yield {"event": "error", "data": {"message": str(e)}, "id": str(uuid4())}

        yield {"event": "done", "data": {}, "id": None}

    def record_feedback(
        self,
        suggestion_id: UUID,
        feedback: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Record feedback on a suggestion."""
        for session in self._sessions.values():
            if session.last_suggestion and session.last_suggestion.suggestion_id == suggestion_id:
                session.feedback_log.append({
                    "suggestion_id": str(suggestion_id),
                    "feedback": feedback,
                    "reason": reason,
                })
                return {
                    "suggestion_id": suggestion_id,
                    "feedback": feedback,
                    "recorded": True,
                }
        return {
            "suggestion_id": suggestion_id,
            "feedback": feedback,
            "recorded": False,
        }
