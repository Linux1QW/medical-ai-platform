"""Coach service: orchestrates the coach graph for API consumption.

Uses CoachRepository for persistence, compile_coach_context for context
building, and invoke_coach_graph for graph execution.

COACH_ENABLED is read from settings at startup (immutable).
Disabled → stable 409 / feature-disabled contract.

Transaction order (durable SSE):
  Reserve Decision → Acquire Lease → Execute Graph →
  Open short TX → Save Decision → Save Events → Commit TX →
  Read committed Events → Send Events to client → Send done
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_runtime.graph import invoke_coach_graph
from app.core.config import settings
from app.repositories.coach import CoachRepository
from app.services.coach_context_builder import CoachContextBuilder
from app.services.coach_runtime_factory import CoachRuntimeFactory

logger = logging.getLogger(__name__)


class CoachDisabledError(Exception):
    """Raised when Coach feature is disabled."""

    def __init__(self) -> None:
        super().__init__("Coach feature is disabled")


class CoachDecisionLockedError(Exception):
    """Raised when a decision is already being processed by another worker."""

    def __init__(self) -> None:
        super().__init__("Decision is locked by another worker")


class CoachService:
    """Service layer for coach operations.

    - All persistence goes through CoachRepository.
    - Context is compiled via compile_coach_context.
    - Graph execution uses invoke_coach_graph with hard timeout.
    - COACH_ENABLED is read from settings (no mutable set_enabled()).
    - Transaction order: commit BEFORE sending events to client.
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
    # Stream suggestion (SSE) — durable transaction order
    # ------------------------------------------------------------------

    async def stream_suggestion(
        self,
        *,
        consultation_id: int,
        doctor_id: int,
        latest_message: str,
        idempotency_key: str,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream coach suggestion via SSE events.

        Durable transaction order:
        1. Get or create session
        2. Handle Last-Event-ID replay (decision-level)
        3. Reserve turn (idempotent)
        4. Acquire lease on decision
        5. Execute graph
        6. Open short transaction: save decision + events → commit
        7. Read committed events → yield to client
        8. Yield done

        Never sends suggestion or done before commit.
        """
        self.require_enabled()

        # 1. Get or create session
        session = await self.repo.get_or_create_session(
            consultation_id=consultation_id,
            doctor_id=doctor_id,
        )

        # 2. Handle Last-Event-ID replay (decision-level)
        if last_event_id:
            replay_events = await self._replay_decision_events(
                session_id=session.id,
                last_event_id=last_event_id,
                idempotency_key=idempotency_key,
            )
            if replay_events is not None:
                for evt in replay_events:
                    yield evt
                # Check if decision is already completed
                existing = await self.repo.get_decision_by_idempotency(
                    session.id, idempotency_key
                )
                if existing and existing.stage in ("completed", "blocked", "error", "degraded"):
                    yield {"event": "done", "data": {}, "id": None}
                    return
                # Continue with new turn after replay
            # If last_event_id is invalid, proceed with fresh stream

        # 3. Reserve turn (idempotent)
        decision = await self.repo.reserve_turn(session.id, idempotency_key)

        # If decision already completed → replay from stored decision events
        if decision.stage in ("completed", "blocked", "error", "degraded"):
            events = await self.repo.list_decision_events_after(decision.id, 0)
            for stored_event in events:
                yield {
                    "event": stored_event.event_type,
                    "data": stored_event.data_json or {},
                    "id": stored_event.event_id,
                }
            yield {"event": "done", "data": {}, "id": None}
            return

        # If decision is running but lease expired → recover
        if decision.stage == "running":
            expired = await self.repo.get_expired_running_decisions(
                session_id=session.id
            )
            expired_ids = {d.id for d in expired}
            if decision.id in expired_ids:
                # Lease expired: write stable error state, allow retry
                await self._write_error_terminal_state(decision.id, "LEASE_EXPIRED")
                await self.db.commit()
                yield {
                    "event": "error",
                    "data": {"message": "LEASE_EXPIRED"},
                    "id": None,
                }
                yield {"event": "done", "data": {}, "id": None}
                return
            else:
                # Still locked by another worker
                raise CoachDecisionLockedError()

        # 4. Acquire lease
        lease_acquired = await self.repo.acquire_decision_lease(decision.id)
        if not lease_acquired:
            raise CoachDecisionLockedError()
        await self.db.commit()  # Commit the lease acquisition

        # 5. Execute graph and collect results
        collected_events, result, final_stage = await self._execute_graph_and_collect(
            session=session,
            decision=decision,
            consultation_id=consultation_id,
            doctor_id=doctor_id,
            latest_message=latest_message,
        )

        # 7. Open short transaction: save decision + events → commit
        persisted_events = await self._persist_decision_results(
            session=session,
            decision=decision,
            idempotency_key=idempotency_key,
            collected_events=collected_events,
            result=result,
            final_stage=final_stage,
        )

        # COMMIT the transaction
        await self.db.commit()

        # 8. Read committed events and send to client
        for event in persisted_events:
            yield {
                "event": event.event_type,
                "data": event.data_json or {},
                "id": event.event_id,
            }

        # 9. Send done
        yield {"event": "done", "data": {}, "id": None}

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

        # -- Projection: map ORM → front-end contract (excludes PII) --
        return {
            "session": {
                "session_id": session.public_id,
                "consultation_id": session.consultation_id,
                "status": session.status,
                "created_at": (
                    session.started_at.isoformat() if session.started_at else None
                ),
            },
            "decisions": [
                {
                    "decision_id": d.suggestion_id or str(d.id),
                    "agent": d.intent,
                    "action": d.stage,
                    "rationale": (
                        (d.suggestion_json or {}).get("rationale_summary")
                        if isinstance(d.suggestion_json, dict)
                        else None
                    ),
                    "timestamp": (
                        d.created_at.isoformat() if d.created_at else None
                    ),
                }
                for d in decisions
            ],
            "events": [
                {
                    "event_id": e.event_id,
                    "agent": (
                        (e.data_json or {}).get("agent")
                        if isinstance(e.data_json, dict)
                        else None
                    ),
                    "event_type": e.event_type,
                    "message": (
                        (e.data_json or {}).get("message")
                        or (e.data_json or {}).get("suggestion_id")
                        if isinstance(e.data_json, dict)
                        else None
                    ),
                    "timestamp": (
                        e.created_at.isoformat() if e.created_at else None
                    ),
                }
                for e in events
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _replay_decision_events(
        self,
        session_id: int,
        last_event_id: str,
        idempotency_key: str,
    ) -> list[dict[str, Any]] | None:
        """Replay events after the given event UUID, scoped to the decision.

        Only replays events belonging to the decision identified by idempotency_key.
        Returns None if the event_id is not found (invalid Last-Event-ID).
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

        # Find the decision for this idempotency key
        decision = await self.repo.get_decision_by_idempotency(
            session_id, idempotency_key
        )
        if decision is None:
            return None

        # Only replay events for THIS decision, not the whole session
        events = await self.repo.list_decision_events_after(
            decision.id, event.sequence
        )
        return [
            {
                "event": e.event_type,
                "data": e.data_json or {},
                "id": e.event_id,
            }
            for e in events
        ]

    async def _persist_decision_results(
        self,
        *,
        session: Any,
        decision: Any,
        idempotency_key: str,
        collected_events: list[dict[str, Any]],
        result: dict[str, Any],
        final_stage: str,
    ) -> list[Any]:
        """Persist decision + events in a short transaction, release lease."""
        intent = "unknown"
        if result and isinstance(result.get("intent_result"), dict):
            intent = result["intent_result"].get("intent", "unknown")

        suggestion_json = None
        suggestion_id = None
        confidence = 0.0
        risk_level = "low"
        for evt in collected_events:
            if evt["event_type"] == "suggestion":
                suggestion_json = evt["data"]
                suggestion_id = evt["data"].get("suggestion_id", str(uuid4()))
                confidence = evt["data"].get("confidence", 0.0)
                risk_level = evt["data"].get("risk_level", "low")
                break

        await self.repo.save_decision(
            session_id=session.id,
            decision_data={
                "idempotency_key": idempotency_key,
                "intent": intent,
                "stage": final_stage,
                "suggestion_json": suggestion_json,
                "suggestion_id": suggestion_id,
                "confidence": confidence,
                "risk_level": risk_level,
            },
        )

        persisted_events = []
        for evt_data in collected_events:
            event = await self.repo.append_stream_event(
                session_id=session.id,
                event_type=evt_data["event_type"],
                data=evt_data["data"],
                decision_id=decision.id,
            )
            persisted_events.append(event)

        await self.repo.release_decision_lease(decision.id, final_stage)
        return persisted_events

    async def _execute_graph_and_collect(
        self,
        *,
        session: Any,
        decision: Any,
        consultation_id: int,
        doctor_id: int,
        latest_message: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """Execute the coach graph and collect events + final stage.

        Returns (collected_events, result_dict, final_stage).
        """
        collected_events: list[dict[str, Any]] = [{
            "event_type": "thinking",
            "data": {"turn_no": decision.turn_no},
        }]
        result: dict[str, Any] = {}
        final_stage: str = "error"

        try:
            from sqlalchemy import select as sa_select

            from app.models.consultation import Consultation as ConsultationModel

            stmt = sa_select(ConsultationModel).where(
                ConsultationModel.id == consultation_id
            )
            consult_result = await self.db.execute(stmt)
            consultation = consult_result.scalar_one_or_none()
            if consultation is None:
                raise ValueError(f"Consultation id={consultation_id} not found")

            context_builder = CoachContextBuilder()
            context_view = await context_builder.build(
                self.db, consultation, doctor_id,
            )

            runtime_factory = CoachRuntimeFactory()
            runtime = await runtime_factory.create()

            graph_task = asyncio.create_task(
                invoke_coach_graph(
                    runtime.graph,
                    context=context_view,
                    latest_message=latest_message,
                    turn=decision.turn_no,
                    session_id=UUID(session.public_id),
                    thread_id=session.thread_id,
                    timeout_seconds=settings.COACH_HARD_TIMEOUT_SECONDS,
                )
            )

            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(session.id, decision.id, graph_task)
            )

            try:
                result = await graph_task
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            # Process result into events
            if result.get("blocked"):
                final_stage = "blocked"
                collected_events.append({
                    "event_type": "error",
                    "data": {"message": result.get("block_reason", "COACH_ERROR")},
                })
            elif result.get("final_suggestion"):
                final_stage = "completed"
                suggestion_data = result["final_suggestion"]
                if hasattr(suggestion_data, "model_dump"):
                    suggestion_data = suggestion_data.model_dump(mode="json")
                collected_events.append({
                    "event_type": "suggestion", "data": suggestion_data,
                })
            else:
                final_stage = "degraded"
                collected_events.append({
                    "event_type": "error",
                    "data": {"message": "No suggestion produced"},
                })

        except Exception:
            logger.exception("Coach graph error for consultation=%s", consultation_id)
            final_stage = "error"
            collected_events.append({
                "event_type": "error",
                "data": {"message": "Internal coach error"},
            })

        return collected_events, result, final_stage

    async def _write_error_terminal_state(
        self,
        decision_id: int,
        error_message: str,
    ) -> None:
        """Write a stable error terminal state for an expired decision."""

        from sqlalchemy import select

        from app.models.coach_decision import CoachDecision

        stmt = (
            select(CoachDecision)
            .where(CoachDecision.id == decision_id)
            .with_for_update()
        )
        result = await self.db.execute(stmt)
        decision = result.scalar_one()
        decision.stage = "error"
        decision.lease_expires_at = None
        decision.intent = decision.intent or "unknown"
        await self.db.flush()

    async def _heartbeat_loop(
        self,
        session_id: int,
        decision_id: int,
        graph_task: asyncio.Task,
    ) -> None:
        """Emit heartbeat events every 15 seconds while graph is running.

        Heartbeats are buffered and will be persisted with the decision's events.
        """
        while not graph_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(graph_task),
                    timeout=15.0,
                )
                break  # Graph completed
            except asyncio.TimeoutError:
                # Heartbeat is appended to the repository but will be
                # committed with the main transaction later
                await self.repo.append_stream_event(
                    session_id=session_id,
                    event_type="heartbeat",
                    data={"ts": asyncio.get_event_loop().time()},
                    decision_id=decision_id,
                )
