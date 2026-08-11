"""Transactional repository for coach session runtime.

Provides atomic operations for session management, turn reservation
with idempotency, and SSE stream event persistence.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.coach_decision import CoachDecision
from app.models.coach_session import CoachSession
from app.models.coach_stream_event import CoachStreamEvent

# Default lease duration for running decisions
DECISION_LEASE_SECONDS = 120


class CoachRepository:
    """Handles transactional persistence for coach sessions."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def get_or_create_session(
        self,
        consultation_id: int,
        doctor_id: int,
        mode: str = "off",
    ) -> CoachSession:
        """Return existing session for consultation or create a new one.

        Uses INSERT … ON CONFLICT DO NOTHING pattern via unique consultation_id.
        """
        stmt = select(CoachSession).where(
            CoachSession.consultation_id == consultation_id
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            return existing

        thread_id = f"thread-{uuid.uuid4().hex[:16]}"
        new_session = CoachSession(
            public_id=str(uuid.uuid4()),
            consultation_id=consultation_id,
            doctor_id=doctor_id,
            mode=mode,
            status="active",
            thread_id=thread_id,
            state_version=0,
        )
        self.db.add(new_session)
        try:
            await self.db.flush()
        except IntegrityError:
            # Another worker created it concurrently — fetch the existing row
            await self.db.rollback()
            result = await self.db.execute(stmt)
            existing = result.scalar_one()
            return existing
        return new_session

    # ------------------------------------------------------------------
    # Turn reservation (idempotent)
    # ------------------------------------------------------------------

    async def reserve_turn(
        self,
        session_id: int,
        idempotency_key: str,
    ) -> CoachDecision:
        """Reserve a new turn inside a row-level locked transaction.

        - Acquires SELECT … FOR UPDATE on the session row.
        - Checks for duplicate idempotency_key → returns existing decision.
        - Otherwise increments state_version and creates a new decision.
        - Handles concurrent insert race via IntegrityError fallback.
        """
        # Check for existing decision with same idempotency key first (fast path)
        existing_stmt = select(CoachDecision).where(
            and_(
                CoachDecision.session_id == session_id,
                CoachDecision.idempotency_key == idempotency_key,
            )
        )
        existing_result = await self.db.execute(existing_stmt)
        existing_decision = existing_result.scalar_one_or_none()
        if existing_decision is not None:
            return existing_decision

        # Lock the session row
        session_stmt = (
            select(CoachSession)
            .where(CoachSession.id == session_id)
            .with_for_update()
        )
        session_result = await self.db.execute(session_stmt)
        coach_session: CoachSession = session_result.scalar_one()

        # Increment state_version
        coach_session.state_version += 1
        coach_session.last_turn_at = datetime.utcnow()

        # Determine next turn_no
        turn_no_stmt = (
            select(func.coalesce(func.max(CoachDecision.turn_no), 0))
            .where(CoachDecision.session_id == session_id)
        )
        turn_result = await self.db.execute(turn_no_stmt)
        max_turn: int = turn_result.scalar() or 0
        next_turn = max_turn + 1

        decision = CoachDecision(
            session_id=session_id,
            turn_no=next_turn,
            idempotency_key=idempotency_key,
            intent="pending",
            stage="reserved",
        )
        self.db.add(decision)
        try:
            await self.db.flush()
        except IntegrityError:
            # Another worker inserted the same idempotency key concurrently.
            # Rollback our changes and return the existing decision.
            await self.db.rollback()
            rollback_result = await self.db.execute(existing_stmt)
            return rollback_result.scalar_one()
        return decision

    async def get_decision_by_idempotency(
        self,
        session_id: int,
        idempotency_key: str,
    ) -> Optional[CoachDecision]:
        """Look up a decision by its idempotency key within a session."""
        stmt = select(CoachDecision).where(
            and_(
                CoachDecision.session_id == session_id,
                CoachDecision.idempotency_key == idempotency_key,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def save_decision(
        self,
        session_id: int,
        decision_data: dict,
    ) -> CoachDecision:
        """Persist or update a decision with full result payload.

        If `idempotency_key` is in decision_data and a matching decision exists,
        update it; otherwise create a new one.
        """
        idempotency_key = decision_data.get("idempotency_key")
        if idempotency_key:
            existing = await self.get_decision_by_idempotency(session_id, idempotency_key)
            if existing is not None:
                for key, value in decision_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                await self.db.flush()
                return existing

        # Create new decision
        decision = CoachDecision(
            session_id=session_id,
            **decision_data,
        )
        self.db.add(decision)
        await self.db.flush()
        return decision

    # ------------------------------------------------------------------
    # Stream events
    # ------------------------------------------------------------------

    async def append_stream_event(
        self,
        session_id: int,
        event_type: str,
        data: Optional[dict] = None,
        decision_id: Optional[int] = None,
    ) -> CoachStreamEvent:
        """Append a stream event with transactional sequence allocation.

        Uses SELECT MAX(sequence) FOR UPDATE to guarantee monotonic ordering.
        decision_id is required for durable event-to-decision association.
        """
        if decision_id is None:
            raise ValueError("decision_id is required for append_stream_event")

        # Lock and get next sequence
        seq_stmt = (
            select(func.coalesce(func.max(CoachStreamEvent.sequence), 0))
            .where(CoachStreamEvent.session_id == session_id)
            .with_for_update()
        )
        result = await self.db.execute(seq_stmt)
        max_seq: int = result.scalar() or 0
        next_seq = max_seq + 1

        event = CoachStreamEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            decision_id=decision_id,
            sequence=next_seq,
            event_type=event_type,
            data_json=data,
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def list_events_after(
        self,
        session_id: int,
        after_sequence: int,
        limit: int = 100,
    ) -> list[CoachStreamEvent]:
        """List events with sequence > after_sequence for the given session.

        Rejects Last-Event-ID from another session by filtering on session_id.
        Returns events ordered by sequence ascending.
        """
        stmt = (
            select(CoachStreamEvent)
            .where(
                and_(
                    CoachStreamEvent.session_id == session_id,
                    CoachStreamEvent.sequence > after_sequence,
                )
            )
            .order_by(CoachStreamEvent.sequence.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_decision_events_after(
        self,
        decision_id: int,
        after_sequence: int,
        limit: int = 100,
    ) -> list[CoachStreamEvent]:
        """List events for a specific decision with sequence > after_sequence.

        Returns events ordered by sequence ascending.
        """
        stmt = (
            select(CoachStreamEvent)
            .where(
                and_(
                    CoachStreamEvent.decision_id == decision_id,
                    CoachStreamEvent.sequence > after_sequence,
                )
            )
            .order_by(CoachStreamEvent.sequence.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Decision lease management
    # ------------------------------------------------------------------

    async def acquire_decision_lease(
        self,
        decision_id: int,
        lease_seconds: int = DECISION_LEASE_SECONDS,
    ) -> bool:
        """Acquire a lease on a decision for processing.

        Returns True if the lease was acquired, False if already leased
        and the existing lease has not expired.
        """
        now = datetime.utcnow()
        stmt = (
            select(CoachDecision)
            .where(CoachDecision.id == decision_id)
            .with_for_update()
        )
        result = await self.db.execute(stmt)
        decision = result.scalar_one()

        # If already leased and not expired, cannot acquire
        if (
            decision.lease_expires_at is not None
            and decision.lease_expires_at > now
            and decision.stage == "running"
        ):
            return False

        decision.locked_at = now
        decision.lease_expires_at = now + timedelta(seconds=lease_seconds)
        decision.stage = "running"
        await self.db.flush()
        return True

    async def release_decision_lease(
        self,
        decision_id: int,
        final_stage: str,
    ) -> None:
        """Release the lease on a decision and set final stage."""
        stmt = (
            select(CoachDecision)
            .where(CoachDecision.id == decision_id)
            .with_for_update()
        )
        result = await self.db.execute(stmt)
        decision = result.scalar_one()
        decision.stage = final_stage
        decision.lease_expires_at = None
        await self.db.flush()

    async def get_expired_running_decisions(
        self,
        session_id: Optional[int] = None,
    ) -> list[CoachDecision]:
        """Find decisions that are running but have expired leases."""
        now = datetime.utcnow()
        conditions = [
            CoachDecision.stage == "running",
            CoachDecision.lease_expires_at < now,
        ]
        if session_id is not None:
            conditions.append(CoachDecision.session_id == session_id)

        stmt = (
            select(CoachDecision)
            .where(and_(*conditions))
            .order_by(CoachDecision.created_at.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
