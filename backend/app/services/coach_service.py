"""Coach service: orchestrates the coach graph for API consumption.

Uses CoachRepository for persistence, compile_coach_context for context
building, and invoke_coach_graph for graph execution.

COACH_ENABLED is read from settings at startup (immutable).
Disabled → stable 409 / feature-disabled contract.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_runtime.context import compile_coach_context
from app.agent_runtime.contracts import (
    CoachContextView,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.agent_runtime.graph import build_coach_graph, invoke_coach_graph
from app.core.config import settings
from app.repositories.coach import CoachRepository

logger = logging.getLogger(__name__)


class CoachDisabledError(Exception):
    """Raised when Coach feature is disabled."""

    def __init__(self) -> None:
        super().__init__("Coach feature is disabled")


class CoachService:
    """Service layer for coach operations.

    - All persistence goes through CoachRepository.
    - Context is compiled via compile_coach_context.
    - Graph execution uses invoke_coach_graph with hard timeout.
    - COACH_ENABLED is read from settings (no mutable set_enabled()).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = CoachRepository(db)
        self._enabled: bool = settings.COACH_ENABLED

    @property
    def enabled(self) -> bool:
        return self._enabled

    def require_enabled(self) -> None:
        """Raise CoachDisabledError if Coach is not enabled."""
        if not self._enabled:
            raise CoachDisabledError()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    async def get_state(self, consultation_id: int) -> dict[str, Any]:
        """Get current coach state for a consultation."""
        if not self._enabled:
            return {
                "consultation_id": consultation_id,
                "status": "disabled",
                "current_stage": None,
                "turn_no": 0,
            }

        from sqlalchemy import select

        from app.models.coach_session import CoachSession

        stmt = select(CoachSession).where(
            CoachSession.consultation_id == consultation_id
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return {
                "consultation_id": consultation_id,
                "status": "idle",
                "current_stage": None,
                "turn_no": 0,
            }

        return {
            "consultation_id": consultation_id,
            "status": session.status if self._enabled else "disabled",
            "current_stage": None,
            "turn_no": session.state_version,
        }

    # ------------------------------------------------------------------
    # Stream suggestion (SSE)
    # ------------------------------------------------------------------

    async def stream_suggestion(
        self,
        *,
        consultation_id: int,
        doctor_id: int,
        latest_message: str,
        idempotency_key: str,
        last_event_id: str | None = None,
        patient_age: int = 45,
        patient_gender: str = "男",
        chief_complaint: str = "问诊",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream coach suggestion via SSE events.

        - Durable idempotency via repository unique keys.
        - Last-Event-ID replay from persisted events.
        - Heartbeat comments every 15 seconds during long model calls.
        """
        self.require_enabled()

        # 1. Get or create session
        session = await self.repo.get_or_create_session(
            consultation_id=consultation_id,
            doctor_id=doctor_id,
        )

        # 2. Handle Last-Event-ID replay
        if last_event_id:
            replay_events = await self._replay_from_event(
                session_id=session.id,
                last_event_id=last_event_id,
            )
            if replay_events is not None:
                for evt in replay_events:
                    yield evt
                # After replay, check if there's already a completed decision
                existing = await self.repo.get_decision_by_idempotency(
                    session.id, idempotency_key
                )
                if existing and existing.stage == "completed":
                    yield {"event": "done", "data": {}, "id": None}
                    return
                # Continue with new turn after replay
            # If last_event_id is invalid (not found), we proceed with fresh stream

        # 3. Reserve turn (idempotent)
        decision = await self.repo.reserve_turn(session.id, idempotency_key)

        # If decision already completed → replay from stored events
        if decision.stage == "completed" and decision.suggestion_json:
            # Replay all events for this decision
            events: list = await self.repo.list_events_after(session.id, 0)
            for stored_event in events:
                yield {
                    "event": stored_event.event_type,
                    "data": stored_event.data_json or {},
                    "id": stored_event.event_id,
                }
            yield {"event": "done", "data": {}, "id": None}
            return

        # 4. Emit thinking event
        thinking_event = await self.repo.append_stream_event(
            session_id=session.id,
            event_type="thinking",
            data={"turn_no": decision.turn_no},
        )
        yield {
            "event": "thinking",
            "data": {"turn_no": decision.turn_no},
            "id": thinking_event.event_id,
        }

        # 5. Build context and run graph with heartbeat
        try:
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
                        sequence=decision.turn_no,
                        role="doctor",
                        content=latest_message,
                    ),
                ],
            )

            compiled = compile_coach_context(context_view)

            # Run graph with heartbeat
            graph = build_coach_graph()
            graph_task = asyncio.create_task(
                invoke_coach_graph(
                    graph,
                    context=compiled,
                    latest_message=latest_message,
                    turn=decision.turn_no,
                    session_id=UUID(session.public_id),
                    thread_id=session.thread_id,
                    timeout_seconds=settings.COACH_HARD_TIMEOUT_SECONDS,
                )
            )

            # Heartbeat loop: emit heartbeat every 15s while graph runs
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(session.id, graph_task)
            )

            try:
                result = await graph_task
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            # 6. Process result
            if result.get("blocked"):
                error_event = await self.repo.append_stream_event(
                    session_id=session.id,
                    event_type="error",
                    data={"message": result.get("block_reason", "COACH_ERROR")},
                )
                yield {
                    "event": "error",
                    "data": {"message": result.get("block_reason", "COACH_ERROR")},
                    "id": error_event.event_id,
                }
                # Update decision
                await self.repo.save_decision(
                    session_id=session.id,
                    decision_data={
                        "idempotency_key": idempotency_key,
                        "intent": result.get("intent_result", {}).get("intent", "unknown") if isinstance(result.get("intent_result"), dict) else "unknown",
                        "stage": "blocked",
                        "risk_level": "low",
                    },
                )
            elif result.get("final_suggestion"):
                suggestion_data = result["final_suggestion"]
                if hasattr(suggestion_data, "model_dump"):
                    suggestion_data = suggestion_data.model_dump(mode="json")

                suggestion_event = await self.repo.append_stream_event(
                    session_id=session.id,
                    event_type="suggestion",
                    data=suggestion_data,
                )
                yield {
                    "event": "suggestion",
                    "data": suggestion_data,
                    "id": suggestion_event.event_id,
                }

                # Update decision with result
                await self.repo.save_decision(
                    session_id=session.id,
                    decision_data={
                        "idempotency_key": idempotency_key,
                        "intent": result.get("intent_result", {}).get("intent", "unknown") if isinstance(result.get("intent_result"), dict) else "unknown",
                        "stage": "completed",
                        "suggestion_json": suggestion_data,
                        "suggestion_id": suggestion_data.get("suggestion_id", str(uuid4())),
                        "confidence": suggestion_data.get("confidence", 0.0),
                        "risk_level": suggestion_data.get("risk_level", "low"),
                    },
                )
            else:
                error_event = await self.repo.append_stream_event(
                    session_id=session.id,
                    event_type="error",
                    data={"message": "No suggestion produced"},
                )
                yield {
                    "event": "error",
                    "data": {"message": "No suggestion produced"},
                    "id": error_event.event_id,
                }
                await self.repo.save_decision(
                    session_id=session.id,
                    decision_data={
                        "idempotency_key": idempotency_key,
                        "intent": "unknown",
                        "stage": "degraded",
                        "risk_level": "low",
                    },
                )

        except Exception:
            logger.exception("Coach graph error for consultation=%s", consultation_id)
            error_event = await self.repo.append_stream_event(
                session_id=session.id,
                event_type="error",
                data={"message": "Internal coach error"},
            )
            yield {
                "event": "error",
                "data": {"message": "Internal coach error"},
                "id": error_event.event_id,
            }
            await self.repo.save_decision(
                session_id=session.id,
                decision_data={
                    "idempotency_key": idempotency_key,
                    "intent": "unknown",
                    "stage": "error",
                    "risk_level": "low",
                },
            )

        yield {"event": "done", "data": {}, "id": None}
        await self.db.commit()

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    async def record_feedback(
        self,
        suggestion_id: UUID,
        feedback: str,
        doctor_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Record feedback on a suggestion, verifying ownership."""
        from sqlalchemy import select

        from app.models.coach_decision import CoachDecision
        from app.models.coach_session import CoachSession

        # Find the decision by suggestion_id
        stmt = (
            select(CoachDecision)
            .join(CoachSession, CoachDecision.session_id == CoachSession.id)
            .where(
                CoachDecision.suggestion_id == str(suggestion_id),
                CoachSession.doctor_id == doctor_id,
            )
        )
        result = await self.db.execute(stmt)
        decision = result.scalar_one_or_none()

        if decision is None:
            return {
                "suggestion_id": suggestion_id,
                "feedback": feedback,
                "recorded": False,
            }

        from datetime import datetime
        decision.feedback_value = feedback
        decision.feedback_reason = reason
        decision.feedback_at = datetime.utcnow()
        await self.db.flush()
        await self.db.commit()

        return {
            "suggestion_id": suggestion_id,
            "feedback": feedback,
            "recorded": True,
        }

    # ------------------------------------------------------------------
    # Admin trace
    # ------------------------------------------------------------------

    async def get_trace(
        self,
        consultation_id: int,
    ) -> dict[str, Any]:
        """Get full trace for admin inspection."""
        from sqlalchemy import select

        from app.models.coach_decision import CoachDecision
        from app.models.coach_session import CoachSession

        session_stmt = select(CoachSession).where(
            CoachSession.consultation_id == consultation_id
        )
        result = await self.db.execute(session_stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return {"session": None, "decisions": [], "events": []}

        decisions_stmt = (
            select(CoachDecision)
            .where(CoachDecision.session_id == session.id)
            .order_by(CoachDecision.turn_no.asc())
        )
        decisions_result = await self.db.execute(decisions_stmt)
        decisions = list(decisions_result.scalars().all())

        events = await self.repo.list_events_after(session.id, 0, limit=500)

        return {
            "session": {
                "id": session.public_id,
                "consultation_id": session.consultation_id,
                "doctor_id": session.doctor_id,
                "mode": session.mode,
                "status": session.status,
                "state_version": session.state_version,
            },
            "decisions": [
                {
                    "turn_no": d.turn_no,
                    "intent": d.intent,
                    "stage": d.stage,
                    "confidence": d.confidence,
                    "risk_level": d.risk_level,
                    "feedback_value": d.feedback_value,
                }
                for d in decisions
            ],
            "events": [
                {
                    "event_id": e.event_id,
                    "sequence": e.sequence,
                    "event_type": e.event_type,
                    "data_json": e.data_json,
                }
                for e in events
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _replay_from_event(
        self,
        session_id: int,
        last_event_id: str,
    ) -> list[dict[str, Any]] | None:
        """Replay events after the given event UUID.

        Returns None if the event_id is not found (invalid Last-Event-ID).
        Returns list of SSE event dicts to replay.
        """
        from sqlalchemy import select

        from app.models.coach_stream_event import CoachStreamEvent

        # Find the event by event_id
        stmt = select(CoachStreamEvent).where(
            CoachStreamEvent.event_id == last_event_id
        )
        result = await self.db.execute(stmt)
        event = result.scalar_one_or_none()

        if event is None:
            return None

        # Verify session ownership
        if event.session_id != session_id:
            return None

        # Get all events after this sequence
        events = await self.repo.list_events_after(session_id, event.sequence)
        return [
            {
                "event": e.event_type,
                "data": e.data_json or {},
                "id": e.event_id,
            }
            for e in events
        ]

    async def _heartbeat_loop(
        self,
        session_id: int,
        graph_task: asyncio.Task,
    ) -> None:
        """Emit heartbeat comments every 15 seconds while graph is running."""
        while not graph_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(graph_task),
                    timeout=15.0,
                )
                break  # Graph completed
            except asyncio.TimeoutError:
                # Emit heartbeat
                await self.repo.append_stream_event(
                    session_id=session_id,
                    event_type="heartbeat",
                    data={"ts": asyncio.get_event_loop().time()},
                )
