"""Tests for CoachRepository transactional operations."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.coach_decision import CoachDecision
from app.models.coach_session import CoachSession
from app.repositories.coach import CoachRepository


@pytest.fixture
def repo(db_session: AsyncSession) -> CoachRepository:
    """Create a CoachRepository instance."""
    return CoachRepository(db_session)


@pytest.fixture
async def session(db_session: AsyncSession) -> CoachSession:
    """Create a test coach session with unique consultation_id."""
    s = CoachSession(
        public_id=str(uuid.uuid4()),
        consultation_id=abs(hash(uuid.uuid4())) % 10**9,
        doctor_id=1,
        mode="on_demand",
        status="active",
        thread_id=f"thread-{uuid.uuid4().hex[:16]}",
        state_version=0,
    )
    db_session.add(s)
    await db_session.flush()
    return s


@pytest.fixture
async def second_session(db_session: AsyncSession) -> CoachSession:
    """Create a second test coach session for cross-session tests."""
    s = CoachSession(
        public_id=str(uuid.uuid4()),
        consultation_id=abs(hash(uuid.uuid4())) % 10**9,
        doctor_id=1,
        mode="shadow",
        status="active",
        thread_id=f"thread-{uuid.uuid4().hex[:16]}",
        state_version=0,
    )
    db_session.add(s)
    await db_session.flush()
    return s


class TestGetOrCreateSession:
    """Tests for get_or_create_session."""

    async def test_creates_new_session(self, repo: CoachRepository):
        """Should create a new session for a new consultation."""
        consultation_id = abs(hash(uuid.uuid4())) % 10**9
        s = await repo.get_or_create_session(consultation_id=consultation_id, doctor_id=1, mode="on_demand")
        assert s.id is not None
        assert s.consultation_id == consultation_id
        assert s.mode == "on_demand"
        assert s.state_version == 0

    async def test_returns_existing_session(self, repo: CoachRepository, session: CoachSession):
        """Should return existing session for same consultation_id."""
        s = await repo.get_or_create_session(
            consultation_id=session.consultation_id,
            doctor_id=session.doctor_id,
            mode=session.mode,
        )
        assert s.id == session.id


class TestReserveTurn:
    """Tests for reserve_turn idempotency."""

    async def test_same_idempotency_key_returns_existing_decision(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Sequential reserve_turn with same key should return same decision."""
        first = await repo.reserve_turn(session.id, "same-key")
        second = await repo.reserve_turn(session.id, "same-key")
        assert first.id == second.id
        assert first.turn_no == second.turn_no

    async def test_concurrent_idempotency_with_separate_sessions(
        self, session: CoachSession, async_engine
    ):
        """Concurrent reserve_turn with separate sessions - first commits, then second sees it.

        Note: Uses unique cleanup to avoid polluting other tests.
        """
        session_factory = async_sessionmaker(
            bind=async_engine, class_=AsyncSession, expire_on_commit=False
        )

        idempotency_key = f"concurrent-{uuid.uuid4().hex[:8]}"

        # First worker reserves and commits
        async with session_factory() as s1:
            r1 = CoachRepository(s1)
            first = await r1.reserve_turn(session.id, idempotency_key)
            await s1.commit()

        # Second worker should find the existing decision
        async with session_factory() as s2:
            r2 = CoachRepository(s2)
            second = await r2.reserve_turn(session.id, idempotency_key)
            await s2.commit()

        assert first.id == second.id
        assert first.turn_no == second.turn_no

        # Cleanup: delete the committed decision to avoid polluting other tests
        async with session_factory() as s_cleanup:
            from sqlalchemy import delete
            await s_cleanup.execute(
                delete(CoachDecision).where(CoachDecision.idempotency_key == idempotency_key)
            )
            await s_cleanup.commit()

    async def test_different_keys_create_separate_turns(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Different idempotency keys should create separate decisions."""
        d1 = await repo.reserve_turn(session.id, "key-1")
        d2 = await repo.reserve_turn(session.id, "key-2")
        assert d1.id != d2.id
        assert d1.turn_no != d2.turn_no
        assert d2.turn_no == d1.turn_no + 1

    async def test_state_version_increments(
        self, repo: CoachRepository, db_session: AsyncSession, session: CoachSession
    ):
        """Each new turn should increment session state_version."""
        await repo.reserve_turn(session.id, "v-key-1")
        await db_session.refresh(session)
        assert session.state_version == 1

        await repo.reserve_turn(session.id, "v-key-2")
        await db_session.refresh(session)
        assert session.state_version == 2


class TestGetDecisionByIdempotency:
    """Tests for get_decision_by_idempotency."""

    async def test_returns_existing_decision(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Should find decision by idempotency key."""
        d = await repo.reserve_turn(session.id, "lookup-key")
        found = await repo.get_decision_by_idempotency(session.id, "lookup-key")
        assert found is not None
        assert found.id == d.id

    async def test_returns_none_for_missing_key(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Should return None for non-existent key."""
        found = await repo.get_decision_by_idempotency(session.id, "nonexistent")
        assert found is None


class TestStreamEvents:
    """Tests for append_stream_event and list_events_after."""

    async def test_events_have_monotonic_sequence(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Events should have strictly increasing sequence numbers."""
        e1 = await repo.append_stream_event(session.id, "suggestion", {"text": "a"})
        e2 = await repo.append_stream_event(session.id, "heartbeat", None)
        e3 = await repo.append_stream_event(session.id, "error", {"msg": "fail"})

        assert e1.sequence == 1
        assert e2.sequence == 2
        assert e3.sequence == 3

    async def test_list_events_after_replays_in_order(
        self, repo: CoachRepository, session: CoachSession
    ):
        """list_events_after should return events in sequence order."""
        await repo.append_stream_event(session.id, "suggestion", {"idx": 1})
        await repo.append_stream_event(session.id, "suggestion", {"idx": 2})
        await repo.append_stream_event(session.id, "suggestion", {"idx": 3})

        events = await repo.list_events_after(session.id, after_sequence=0)
        assert len(events) == 3
        assert events[0].sequence == 1
        assert events[1].sequence == 2
        assert events[2].sequence == 3

    async def test_list_events_after_partial_replay(
        self, repo: CoachRepository, session: CoachSession
    ):
        """Should only return events after the given sequence."""
        await repo.append_stream_event(session.id, "suggestion", {"idx": 1})
        await repo.append_stream_event(session.id, "suggestion", {"idx": 2})
        await repo.append_stream_event(session.id, "suggestion", {"idx": 3})

        events = await repo.list_events_after(session.id, after_sequence=2)
        assert len(events) == 1
        assert events[0].sequence == 3

    async def test_last_event_id_rejected_across_sessions(
        self,
        repo: CoachRepository,
        session: CoachSession,
        second_session: CoachSession,
    ):
        """Events from session A should not leak into session B's replay."""
        # Add events to session A
        await repo.append_stream_event(session.id, "suggestion", {"session": "A"})
        await repo.append_stream_event(session.id, "suggestion", {"session": "A"})

        # Add events to session B
        await repo.append_stream_event(second_session.id, "suggestion", {"session": "B"})

        # Query session B events — should only see session B's events
        events_b = await repo.list_events_after(second_session.id, after_sequence=0)
        assert len(events_b) == 1
        assert events_b[0].data_json == {"session": "B"}

        # Query session A events — should only see session A's events
        events_a = await repo.list_events_after(session.id, after_sequence=0)
        assert len(events_a) == 2
        for e in events_a:
            assert e.data_json == {"session": "A"}


class TestTwoSessionBehavior:
    """Tests for multi-session isolation."""

    async def test_separate_sessions_have_independent_sequences(
        self,
        repo: CoachRepository,
        session: CoachSession,
        second_session: CoachSession,
    ):
        """Each session should have its own sequence counter."""
        e_a1 = await repo.append_stream_event(session.id, "suggestion", None)
        e_b1 = await repo.append_stream_event(second_session.id, "suggestion", None)
        e_a2 = await repo.append_stream_event(session.id, "suggestion", None)

        assert e_a1.sequence == 1
        assert e_b1.sequence == 1  # Independent counter
        assert e_a2.sequence == 2

    async def test_separate_sessions_have_independent_turns(
        self,
        repo: CoachRepository,
        session: CoachSession,
        second_session: CoachSession,
    ):
        """Each session should have its own turn counter."""
        d_a1 = await repo.reserve_turn(session.id, "a-key-1")
        d_b1 = await repo.reserve_turn(second_session.id, "b-key-1")
        d_a2 = await repo.reserve_turn(session.id, "a-key-2")

        assert d_a1.turn_no == 1
        assert d_b1.turn_no == 1  # Independent counter
        assert d_a2.turn_no == 2
