"""Integration tests: Coach disconnect, replay, and idempotency.

Covers:
- Completion before disconnect (replay of completed decision)
- Last-Event-ID reconnect (decision-level replay)
- Same idempotency key repeated (idempotent behavior)
- Multiple decisions per session
- Decision lease management
- Expired lease recovery
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models.base import Base
from app.models.coach_session import CoachSession
from app.repositories.coach import CoachRepository


@pytest.fixture
async def engine():
    """Create an async engine for each test."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture
def session_factory(engine):
    """Create a session factory."""
    return async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )


async def _create_completed_decision(
    session_factory: async_sessionmaker,
    idem_key: str = "idem-key-1",
) -> tuple[int, int]:
    """Helper: create a session with a completed decision and events."""
    async with session_factory() as worker:
        repo = CoachRepository(worker)

        s = CoachSession(
            public_id=str(uuid.uuid4()),
            consultation_id=abs(hash(uuid.uuid4())) % 10**9,
            doctor_id=1,
            mode="on_demand",
            status="active",
            thread_id=f"thread-{uuid.uuid4().hex[:16]}",
            state_version=0,
        )
        worker.add(s)
        await worker.flush()

        decision = await repo.reserve_turn(s.id, idem_key)
        await repo.append_stream_event(
            s.id, "thinking", {"turn_no": 1}, decision_id=decision.id
        )
        await repo.append_stream_event(
            s.id, "suggestion", {"text": "first suggestion"}, decision_id=decision.id
        )
        decision.stage = "completed"
        decision.intent = "rapport"
        decision.suggestion_json = {"text": "first suggestion"}
        await worker.commit()

        return s.id, decision.id


class TestCompletedDecisionReplay:
    """Replay a completed decision's events."""

    async def test_replay_all_events_for_completed_decision(self, session_factory):
        """After disconnect, replay returns all events for the decision."""
        session_id, decision_id = await _create_completed_decision(session_factory)

        async with session_factory() as worker:
            repo = CoachRepository(worker)
            events = await repo.list_decision_events_after(decision_id, 0)

            assert len(events) == 2
            assert events[0].event_type == "thinking"
            assert events[1].event_type == "suggestion"

    async def test_replay_from_last_event_id(self, session_factory):
        """Replay from a specific sequence returns only subsequent events."""
        session_id, decision_id = await _create_completed_decision(session_factory)

        async with session_factory() as worker:
            repo = CoachRepository(worker)
            all_events = await repo.list_decision_events_after(decision_id, 0)
            first_seq = all_events[0].sequence

            events = await repo.list_decision_events_after(decision_id, first_seq)
            assert len(events) == 1
            assert events[0].event_type == "suggestion"


class TestIdempotencyKey:
    """Same idempotency key must return existing decision."""

    async def test_same_key_returns_existing_decision(self, session_factory):
        """get_decision_by_idempotency finds the existing decision."""
        session_id, decision_id = await _create_completed_decision(
            session_factory, idem_key="idem-lookup-key"
        )

        async with session_factory() as worker:
            repo = CoachRepository(worker)
            existing = await repo.get_decision_by_idempotency(session_id, "idem-lookup-key")

            assert existing is not None
            assert existing.id == decision_id
            assert existing.stage == "completed"

    async def test_different_key_creates_new_decision(self, session_factory):
        """Different idempotency key creates a new decision."""
        session_id, decision_id = await _create_completed_decision(session_factory)

        async with session_factory() as worker:
            repo = CoachRepository(worker)
            new_decision = await repo.reserve_turn(session_id, "different-key")

            assert new_decision.id != decision_id
            assert new_decision.turn_no == 2
            await worker.rollback()


class TestMultipleDecisionsPerSession:
    """A session can have multiple decisions with isolated events."""

    async def test_multiple_decisions_isolated_events(self, session_factory):
        """Events from different decisions don't mix."""
        session_id, decision_id_1 = await _create_completed_decision(
            session_factory, idem_key="multi-decision-1"
        )

        async with session_factory() as worker:
            repo = CoachRepository(worker)

            # Create second decision in same session
            decision_2 = await repo.reserve_turn(session_id, "multi-decision-2")
            await repo.append_stream_event(
                session_id, "thinking", {"turn_no": 2},
                decision_id=decision_2.id
            )
            await repo.append_stream_event(
                session_id, "suggestion", {"text": "second suggestion"},
                decision_id=decision_2.id
            )
            decision_2.stage = "completed"
            await worker.commit()

            # Verify decision 1 events are isolated
            events_1 = await repo.list_decision_events_after(decision_id_1, 0)
            assert len(events_1) == 2
            assert all(e.decision_id == decision_id_1 for e in events_1)

            # Verify decision 2 events are isolated
            events_2 = await repo.list_decision_events_after(decision_2.id, 0)
            assert len(events_2) == 2
            assert all(e.decision_id == decision_2.id for e in events_2)


class TestDecisionLease:
    """Decision lease management for long-running operations."""

    async def test_acquire_lease(self, session_factory):
        """Acquiring a lease sets locked_at and lease_expires_at."""
        async with session_factory() as worker:
            repo = CoachRepository(worker)

            s = CoachSession(
                public_id=str(uuid.uuid4()),
                consultation_id=abs(hash(uuid.uuid4())) % 10**9,
                doctor_id=1,
                mode="on_demand",
                status="active",
                thread_id=f"thread-{uuid.uuid4().hex[:16]}",
                state_version=0,
            )
            worker.add(s)
            await worker.flush()

            decision = await repo.reserve_turn(s.id, "lease-test-key")
            acquired = await repo.acquire_decision_lease(decision.id, lease_seconds=60)

            assert acquired is True
            await worker.refresh(decision)
            assert decision.stage == "running"
            assert decision.locked_at is not None
            assert decision.lease_expires_at is not None

            await worker.rollback()

    async def test_cannot_acquire_already_leased_decision(self, session_factory):
        """Cannot acquire lease on an already leased decision."""
        async with session_factory() as worker:
            repo = CoachRepository(worker)

            s = CoachSession(
                public_id=str(uuid.uuid4()),
                consultation_id=abs(hash(uuid.uuid4())) % 10**9,
                doctor_id=1,
                mode="on_demand",
                status="active",
                thread_id=f"thread-{uuid.uuid4().hex[:16]}",
                state_version=0,
            )
            worker.add(s)
            await worker.flush()

            decision = await repo.reserve_turn(s.id, "lease-conflict-key")
            acquired_1 = await repo.acquire_decision_lease(decision.id, lease_seconds=60)
            assert acquired_1 is True

            acquired_2 = await repo.acquire_decision_lease(decision.id, lease_seconds=60)
            assert acquired_2 is False

            await worker.rollback()

    async def test_release_lease(self, session_factory):
        """Releasing a lease clears lease_expires_at and sets final stage."""
        async with session_factory() as worker:
            repo = CoachRepository(worker)

            s = CoachSession(
                public_id=str(uuid.uuid4()),
                consultation_id=abs(hash(uuid.uuid4())) % 10**9,
                doctor_id=1,
                mode="on_demand",
                status="active",
                thread_id=f"thread-{uuid.uuid4().hex[:16]}",
                state_version=0,
            )
            worker.add(s)
            await worker.flush()

            decision = await repo.reserve_turn(s.id, "lease-release-key")
            await repo.acquire_decision_lease(decision.id, lease_seconds=60)
            await repo.release_decision_lease(decision.id, "completed")

            await worker.refresh(decision)
            assert decision.stage == "completed"
            assert decision.lease_expires_at is None

            await worker.rollback()

    async def test_expired_lease_can_be_reacquired(self, session_factory):
        """An expired lease can be acquired again."""
        async with session_factory() as worker:
            repo = CoachRepository(worker)

            s = CoachSession(
                public_id=str(uuid.uuid4()),
                consultation_id=abs(hash(uuid.uuid4())) % 10**9,
                doctor_id=1,
                mode="on_demand",
                status="active",
                thread_id=f"thread-{uuid.uuid4().hex[:16]}",
                state_version=0,
            )
            worker.add(s)
            await worker.flush()

            decision = await repo.reserve_turn(s.id, "lease-expired-key")
            await repo.acquire_decision_lease(decision.id, lease_seconds=60)

            # Manually expire the lease
            decision.lease_expires_at = datetime.utcnow() - timedelta(seconds=10)
            await worker.flush()

            acquired = await repo.acquire_decision_lease(decision.id, lease_seconds=60)
            assert acquired is True

            await worker.rollback()

    async def test_get_expired_running_decisions(self, session_factory):
        """get_expired_running_decisions finds decisions with expired leases."""
        async with session_factory() as worker:
            repo = CoachRepository(worker)

            s = CoachSession(
                public_id=str(uuid.uuid4()),
                consultation_id=abs(hash(uuid.uuid4())) % 10**9,
                doctor_id=1,
                mode="on_demand",
                status="active",
                thread_id=f"thread-{uuid.uuid4().hex[:16]}",
                state_version=0,
            )
            worker.add(s)
            await worker.flush()

            decision = await repo.reserve_turn(s.id, "lease-find-expired-key")
            await repo.acquire_decision_lease(decision.id, lease_seconds=60)

            # Manually expire the lease
            decision.lease_expires_at = datetime.utcnow() - timedelta(seconds=10)
            await worker.flush()

            expired = await repo.get_expired_running_decisions(session_id=s.id)
            assert len(expired) == 1
            assert expired[0].id == decision.id

            await worker.rollback()
