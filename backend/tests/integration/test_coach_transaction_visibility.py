"""Integration tests: Coach transaction visibility.

Verifies that after a client receives 'done', all Decision and Event data
is immediately visible to a separate database session (Worker B).

This tests the durable SSE guarantee: commit BEFORE sending to client.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models.base import Base
from app.models.coach_decision import CoachDecision
from app.models.coach_session import CoachSession
from app.models.coach_stream_event import CoachStreamEvent
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


async def _create_session_and_decision(
    session_factory: async_sessionmaker,
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

        decision = await repo.reserve_turn(s.id, f"key-{uuid.uuid4().hex[:8]}")
        await repo.append_stream_event(
            s.id, "thinking", {"turn_no": decision.turn_no},
            decision_id=decision.id,
        )
        await repo.append_stream_event(
            s.id, "suggestion", {"text": "test suggestion"},
            decision_id=decision.id,
        )
        decision.stage = "completed"
        decision.intent = "rapport"
        decision.suggestion_json = {"text": "test suggestion"}
        await worker.commit()

        return s.id, decision.id


class TestTransactionVisibility:
    """After commit, all data must be visible to a separate session."""

    async def test_decision_visible_after_commit(self, session_factory):
        """Worker B must see the Decision immediately after Worker A commits."""
        session_id, decision_id = await _create_session_and_decision(session_factory)

        async with session_factory() as worker_b:
            stmt = select(CoachDecision).where(CoachDecision.id == decision_id)
            result = await worker_b.execute(stmt)
            decision = result.scalar_one_or_none()

            assert decision is not None, "Decision must be visible after commit"
            assert decision.stage == "completed"
            assert decision.intent == "rapport"

    async def test_all_events_visible_after_commit(self, session_factory):
        """Worker B must see ALL events for the decision after commit."""
        session_id, decision_id = await _create_session_and_decision(session_factory)

        async with session_factory() as worker_b:
            stmt = (
                select(CoachStreamEvent)
                .where(CoachStreamEvent.decision_id == decision_id)
                .order_by(CoachStreamEvent.sequence.asc())
            )
            result = await worker_b.execute(stmt)
            events = list(result.scalars().all())

            assert len(events) == 2, f"Expected 2 events, got {len(events)}"
            assert events[0].event_type == "thinking"
            assert events[1].event_type == "suggestion"

    async def test_events_linked_to_decision(self, session_factory):
        """All events must have decision_id set."""
        session_id, decision_id = await _create_session_and_decision(session_factory)

        async with session_factory() as worker_b:
            stmt = select(CoachStreamEvent).where(
                CoachStreamEvent.session_id == session_id
            )
            result = await worker_b.execute(stmt)
            events = list(result.scalars().all())

            for event in events:
                assert event.decision_id == decision_id

    async def test_list_decision_events_after_works(self, session_factory):
        """list_decision_events_after returns correct events."""
        session_id, decision_id = await _create_session_and_decision(session_factory)

        async with session_factory() as worker_b:
            repo_b = CoachRepository(worker_b)
            events = await repo_b.list_decision_events_after(decision_id, 0)

            assert len(events) == 2
            assert events[0].sequence == 1
            assert events[1].sequence == 2

    async def test_partial_replay_decision_scoped(self, session_factory):
        """Replay after sequence=1 returns only subsequent events."""
        session_id, decision_id = await _create_session_and_decision(session_factory)

        async with session_factory() as worker_b:
            repo_b = CoachRepository(worker_b)
            events = await repo_b.list_decision_events_after(decision_id, 1)

            assert len(events) == 1
            assert events[0].event_type == "suggestion"
            assert events[0].sequence == 2
