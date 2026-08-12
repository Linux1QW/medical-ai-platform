"""Tests for evaluation_dispatch_service — Transactional Outbox service layer

TDD Phase 1: These tests should FAIL until the implementation is complete.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session():
    """In-memory SQLite async session for unit tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        # Create all tables from Base metadata
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _make_run_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime(2026, 8, 10, 12, 0, 0)


# ── Test: enqueue_dispatch creates outbox row in pending state ────────────────


@pytest.mark.asyncio
async def test_enqueue_dispatch_creates_pending_outbox(db_session: AsyncSession):
    """enqueue_dispatch should add an outbox row with status=pending, without committing."""
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import enqueue_dispatch

    run_id = _make_run_id()
    trace_ctx = {"trace_id": "abc123", "span_id": "span456"}

    outbox = await enqueue_dispatch(
        db_session,
        run_id=run_id,
        consultation_id=42,
        trace_context=trace_ctx,
    )

    assert outbox.run_id == run_id
    assert outbox.status == "pending"
    assert outbox.payload == {
        "run_id": run_id,
        "consultation_id": 42,
        "trace_context": trace_ctx,
    }
    assert outbox.attempt == 0
    # Should be flushable (queryable in same session)
    from sqlalchemy import select
    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    assert result.scalar_one() is not None


# ── Test: run + outbox rollback together ──────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_dispatch_rollback_with_run(db_session: AsyncSession):
    """If the enclosing transaction rolls back, the outbox row must NOT persist."""
    from sqlalchemy import select

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.models.evaluation_run import EvaluationRun
    from app.services.evaluation_dispatch_service import enqueue_dispatch

    run_id = _make_run_id()

    # Create a run in the same transaction
    run = EvaluationRun(
        id=run_id,
        consultation_id=99,
        graph_version="v1",
        scoring_policy_version="v1",
        checkpoint_thread_id=f"eval:{run_id}",
        status="queued",
        attempt=0,
    )
    db_session.add(run)
    await db_session.flush()

    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=99, trace_context={})

    # Rollback the entire transaction
    await db_session.rollback()

    # Outbox row should not exist
    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    assert result.scalar_one_or_none() is None


# ── Test: duplicate enqueue hits unique constraint ────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_dispatch_duplicate_run_id_raises(db_session: AsyncSession):
    """Enqueuing twice for the same run_id should violate the unique constraint."""
    from app.services.evaluation_dispatch_service import enqueue_dispatch

    run_id = _make_run_id()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={})

    with pytest.raises(Exception):  # noqa: B017
        await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={})


# ── Test: claim_dispatch_batch leases pending rows ────────────────────────────


@pytest.mark.asyncio
async def test_claim_dispatch_batch_leases_pending(db_session: AsyncSession):
    """claim_dispatch_batch should transition pending → leased and set lease fields."""
    from sqlalchemy import select

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import claim_dispatch_batch, enqueue_dispatch

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    leases = await claim_dispatch_batch(
        db_session,
        worker_id="worker-1",
        now=now,
        batch_size=10,
        lease_seconds=30,
    )

    assert len(leases) == 1
    assert leases[0].run_id == run_id
    assert leases[0].lease_owner == "worker-1"

    # Verify DB state
    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    row = result.scalar_one()
    assert row.status == "leased"
    assert row.lease_owner == "worker-1"
    assert row.lease_expires_at == now + timedelta(seconds=30)


# ── Test: expired lease can be reclaimed ──────────────────────────────────────


@pytest.mark.asyncio
async def test_claim_dispatch_batch_reclaims_expired_lease(db_session: AsyncSession):
    """Rows with expired leases should be reclaimable by another worker."""
    from sqlalchemy import update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import claim_dispatch_batch, enqueue_dispatch

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    # Manually set to leased with expired lease
    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(
            status="leased",
            lease_owner="worker-old",
            lease_expires_at=now - timedelta(seconds=60),
        )
    )
    await db_session.flush()

    leases = await claim_dispatch_batch(
        db_session,
        worker_id="worker-new",
        now=now,
        batch_size=10,
        lease_seconds=30,
    )

    assert len(leases) == 1
    assert leases[0].lease_owner == "worker-new"


# ── Test: backoff calculation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_calculation_capped_at_60(db_session: AsyncSession):
    """Backoff should be min(2**attempts, 60) + jitter."""
    from app.services.evaluation_dispatch_service import compute_backoff

    # attempts=0 → 2^0=1
    assert compute_backoff(attempts=0, jitter=0.0) == 1.0
    # attempts=5 → 2^5=32
    assert compute_backoff(attempts=5, jitter=0.0) == 32.0
    # attempts=6 → min(64, 60)=60
    assert compute_backoff(attempts=6, jitter=0.0) == 60.0
    # attempts=100 → capped at 60
    assert compute_backoff(attempts=100, jitter=0.0) == 60.0
    # jitter is additive
    assert compute_backoff(attempts=0, jitter=0.5) == 1.5


# ── Test: 1499 attempts + under 24h → still retrying ─────────────────────────


@pytest.mark.asyncio
async def test_reject_dispatch_1499_attempts_continues(db_session: AsyncSession):
    """At 1499 attempts and under 24h age, reject should NOT dead_letter."""
    from sqlalchemy import update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import (
        claim_dispatch_batch,
        enqueue_dispatch,
        reject_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    # Set attempt=1499, created_at=1h ago
    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(attempt=1499, created_at=now - timedelta(hours=1))
    )
    await db_session.flush()

    # Claim then reject
    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )
    assert len(leases) == 1

    new_status = await reject_dispatch(
        db_session,
        event_id=leases[0].event_id,
        lease_owner="w1",
        error_code="broker_unavailable",
        now=now,
    )
    # Should still be pending (not dead_letter)
    assert new_status == "pending"


# ── Test: 1500 attempts → dead_letter ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_dispatch_1500_attempts_dead_letters(db_session: AsyncSession):
    """At 1500 attempts, reject should dead_letter the outbox row."""
    from sqlalchemy import update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import (
        claim_dispatch_batch,
        enqueue_dispatch,
        reject_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(attempt=1500, created_at=now - timedelta(hours=1))
    )
    await db_session.flush()

    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )
    new_status = await reject_dispatch(
        db_session,
        event_id=leases[0].event_id,
        lease_owner="w1",
        error_code="broker_unavailable",
        now=now,
    )
    assert new_status == "dead_letter"


# ── Test: 24h age → dead_letter ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_dispatch_24h_age_dead_letters(db_session: AsyncSession):
    """Even with low attempt count, 24h age should dead_letter."""
    from sqlalchemy import update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import (
        claim_dispatch_batch,
        enqueue_dispatch,
        reject_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    # attempt=5 but created 25h ago
    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(attempt=5, created_at=now - timedelta(hours=25))
    )
    await db_session.flush()

    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )
    new_status = await reject_dispatch(
        db_session,
        event_id=leases[0].event_id,
        lease_owner="w1",
        error_code="broker_unavailable",
        now=now,
    )
    assert new_status == "dead_letter"


# ── Test: cancel_dispatch transitions pending/leased/published → cancelled ────


@pytest.mark.asyncio
async def test_cancel_dispatch_transitions_to_cancelled(db_session: AsyncSession):
    """cancel_dispatch should atomically cancel pending/leased/published outbox rows."""
    from sqlalchemy import select

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import cancel_dispatch, enqueue_dispatch

    run_id = _make_run_id()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={})

    cancelled = await cancel_dispatch(db_session, run_id=run_id)
    assert cancelled is True

    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    row = result.scalar_one()
    assert row.status == "cancelled"


# ── Test: cancel_dispatch with leased row ─────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_dispatch_from_leased(db_session: AsyncSession):
    """cancel_dispatch should also cancel leased rows."""
    from sqlalchemy import select

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import (
        cancel_dispatch,
        claim_dispatch_batch,
        enqueue_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)
    await claim_dispatch_batch(db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30)

    cancelled = await cancel_dispatch(db_session, run_id=run_id)
    assert cancelled is True

    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    row = result.scalar_one()
    assert row.status == "cancelled"


# ── Test: requeue_stale_dispatch preconditions ────────────────────────────────


@pytest.mark.asyncio
async def test_requeue_stale_dispatch_requires_published_status(db_session: AsyncSession):
    """requeue_stale_dispatch should only work on published outbox rows."""
    from app.services.evaluation_dispatch_service import enqueue_dispatch, requeue_stale_dispatch

    run_id = _make_run_id()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={})

    # Outbox is pending, not published → should fail
    result = await requeue_stale_dispatch(db_session, run_id=run_id)
    assert result is False


# ── Test: reject_dispatch dead_letter also fails run ──────────────────────────


@pytest.mark.asyncio
async def test_reject_dispatch_dead_letter_marks_run_failed(db_session: AsyncSession):
    """When outbox dead_letters and run is queued, run should become failed."""
    from sqlalchemy import select, update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.models.evaluation_run import EvaluationRun
    from app.services.evaluation_dispatch_service import (
        claim_dispatch_batch,
        enqueue_dispatch,
        reject_dispatch,
    )

    run_id = _make_run_id()
    now = _now()

    # Create run in queued state
    run = EvaluationRun(
        id=run_id, consultation_id=1, graph_version="v1",
        scoring_policy_version="v1", checkpoint_thread_id=f"eval:{run_id}",
        status="queued", attempt=0,
    )
    db_session.add(run)
    await db_session.flush()

    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    # Set attempt=1500 to force dead_letter
    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(attempt=1500, created_at=now - timedelta(hours=1))
    )
    await db_session.flush()

    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )
    await reject_dispatch(
        db_session,
        event_id=leases[0].event_id,
        lease_owner="w1",
        error_code="broker_unavailable",
        now=now,
    )

    # Run should be failed
    result = await db_session.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))
    failed_run = result.scalar_one()
    assert failed_run.status == "failed"
    assert failed_run.error_type == "dispatch_exhausted"


# ── Test: payload allowlist rejects PII fields ────────────────────────────────


@pytest.mark.asyncio
async def test_payload_allowlist_rejects_pii(db_session: AsyncSession):
    """Payload should reject fields not in the allowlist (PII)."""
    from app.services.evaluation_dispatch_service import enqueue_dispatch

    run_id = _make_run_id()
    bad_trace = {"patient_name": "张三", "trace_id": "ok"}

    with pytest.raises(ValueError, match="not allowed"):
        await enqueue_dispatch(
            db_session, run_id=run_id, consultation_id=1, trace_context=bad_trace
        )


# ── Test: acknowledge_dispatch sets published state ───────────────────────────


@pytest.mark.asyncio
async def test_acknowledge_dispatch_sets_published(db_session: AsyncSession):
    """acknowledge_dispatch should transition leased → published."""
    from sqlalchemy import select

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import (
        acknowledge_dispatch,
        claim_dispatch_batch,
        enqueue_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)
    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )

    await acknowledge_dispatch(
        db_session,
        event_id=leases[0].event_id,
        lease_owner="w1",
        celery_task_id="celery-task-xyz",
        published_at=now,
    )

    result = await db_session.execute(
        select(EvaluationDispatchOutbox).where(EvaluationDispatchOutbox.run_id == run_id)
    )
    row = result.scalar_one()
    assert row.status == "published"
    assert row.last_task_id == "celery-task-xyz"
    assert row.published_at == now


# ── Test: acknowledge with wrong owner raises DispatchLeaseLost ───────────────


@pytest.mark.asyncio
async def test_acknowledge_dispatch_wrong_owner_raises(db_session: AsyncSession):
    """acknowledge_dispatch with wrong lease_owner should raise DispatchLeaseLost."""
    from app.services.evaluation_dispatch_service import (
        DispatchLeaseLost,
        acknowledge_dispatch,
        claim_dispatch_batch,
        enqueue_dispatch,
    )

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)
    leases = await claim_dispatch_batch(
        db_session, worker_id="w1", now=now, batch_size=10, lease_seconds=30
    )

    with pytest.raises(DispatchLeaseLost):
        await acknowledge_dispatch(
            db_session,
            event_id=leases[0].event_id,
            lease_owner="wrong-worker",
            celery_task_id="task-123",
            published_at=now,
        )


# ── Test: purge_terminal_dispatches ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_purge_terminal_dispatches(db_session: AsyncSession):
    """purge_terminal_dispatches should delete published/cancelled/dead_letter older than cutoff."""
    from sqlalchemy import select, update

    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import enqueue_dispatch, purge_terminal_dispatches

    run_id = _make_run_id()
    now = _now()
    await enqueue_dispatch(db_session, run_id=run_id, consultation_id=1, trace_context={}, now=now)

    # Set to published, 8 days old
    await db_session.execute(
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(status="published", created_at=now - timedelta(days=8))
    )
    await db_session.flush()

    deleted = await purge_terminal_dispatches(
        db_session, older_than=now - timedelta(days=7)
    )
    assert deleted == 1

    result = await db_session.execute(select(EvaluationDispatchOutbox))
    assert result.scalar_one_or_none() is None


# ── Test: claim uses FOR UPDATE SKIP LOCKED with deterministic order ─────────


def test_claim_uses_skip_locked_and_deterministic_order():
    """claim_dispatch_batch must use FOR UPDATE SKIP LOCKED with deterministic ordering."""
    from datetime import datetime, timezone

    from sqlalchemy.dialects import mysql

    from app.services.evaluation_dispatch_service import build_claim_statement

    now = datetime.now(timezone.utc)
    stmt = build_claim_statement(now=now, batch_size=20)
    compiled = stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True})
    sql = str(compiled).upper()
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "ORDER BY" in sql
