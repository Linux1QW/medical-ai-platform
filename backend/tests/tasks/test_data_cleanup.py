"""Tests for data_cleanup — retention policy for outbox and runs

TDD Phase 1: These tests should FAIL until the implementation is complete.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base


@pytest_asyncio.fixture
async def db_session():
    """In-memory SQLite async session for unit tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _now() -> datetime:
    return datetime(2026, 8, 10, 12, 0, 0)


# ── Test: 7-day-old published outbox rows are deleted ────────────────────────


@pytest.mark.asyncio
async def test_purge_old_published_outbox(db_session: AsyncSession):
    """Published outbox rows older than 7 days should be purged."""
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import purge_terminal_dispatches

    now = _now()
    run_id = str(uuid.uuid4())

    old_row = EvaluationDispatchOutbox(
        event_id=str(uuid.uuid4()),
        run_id=run_id,
        status="published",
        task_name="run_evaluation",
        payload={"run_id": run_id, "consultation_id": 1},
        attempt=1,
        created_at=now - timedelta(days=8),
        updated_at=now - timedelta(days=8),
        published_at=now - timedelta(days=8),
    )
    db_session.add(old_row)
    await db_session.flush()

    deleted = await purge_terminal_dispatches(db_session, older_than=now - timedelta(days=7))
    assert deleted == 1


# ── Test: 7-day-old cancelled/dead_letter outbox rows are deleted ────────────


@pytest.mark.asyncio
async def test_purge_old_cancelled_dead_letter_outbox(db_session: AsyncSession):
    """Cancelled and dead_letter outbox rows older than 7 days should be purged."""
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import purge_terminal_dispatches

    now = _now()

    for status in ("cancelled", "dead_letter"):
        run_id = str(uuid.uuid4())
        row = EvaluationDispatchOutbox(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            status=status,
            task_name="run_evaluation",
            payload={"run_id": run_id, "consultation_id": 1},
            attempt=1,
            created_at=now - timedelta(days=8),
            updated_at=now - timedelta(days=8),
        )
        db_session.add(row)
    await db_session.flush()

    deleted = await purge_terminal_dispatches(db_session, older_than=now - timedelta(days=7))
    assert deleted == 2


# ── Test: pending/leased outbox rows are NOT deleted ─────────────────────────


@pytest.mark.asyncio
async def test_pending_leased_outbox_not_purged(db_session: AsyncSession):
    """Pending and leased outbox rows should never be purged."""
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import purge_terminal_dispatches

    now = _now()

    for status in ("pending", "leased"):
        run_id = str(uuid.uuid4())
        row = EvaluationDispatchOutbox(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            status=status,
            task_name="run_evaluation",
            payload={"run_id": run_id, "consultation_id": 1},
            attempt=0,
            created_at=now - timedelta(days=30),
            updated_at=now - timedelta(days=30),
        )
        db_session.add(row)
    await db_session.flush()

    deleted = await purge_terminal_dispatches(db_session, older_than=now - timedelta(days=7))
    assert deleted == 0


# ── Test: recent terminal outbox rows are NOT deleted ────────────────────────


@pytest.mark.asyncio
async def test_recent_terminal_outbox_not_purged(db_session: AsyncSession):
    """Terminal outbox rows within 7 days should NOT be purged."""
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.services.evaluation_dispatch_service import purge_terminal_dispatches

    now = _now()
    run_id = str(uuid.uuid4())

    row = EvaluationDispatchOutbox(
        event_id=str(uuid.uuid4()),
        run_id=run_id,
        status="published",
        task_name="run_evaluation",
        payload={"run_id": run_id, "consultation_id": 1},
        attempt=1,
        created_at=now - timedelta(days=3),
        updated_at=now - timedelta(days=3),
        published_at=now - timedelta(days=3),
    )
    db_session.add(row)
    await db_session.flush()

    deleted = await purge_terminal_dispatches(db_session, older_than=now - timedelta(days=7))
    assert deleted == 0


# ── Test: run cleanup only deletes failed/cancelled runs without Evaluation ──


@pytest.mark.asyncio
async def test_run_cleanup_only_failed_cancelled_without_evaluation(db_session: AsyncSession):
    """Only failed/cancelled runs older than 180 days with no Evaluation should be deleted."""
    from app.models.evaluation_run import EvaluationRun
    from app.tasks.data_cleanup import cleanup_unreported_runs

    now = _now()

    # Old failed run without evaluation → should be deleted
    run1 = EvaluationRun(
        id=str(uuid.uuid4()), consultation_id=1, graph_version="v1",
        scoring_policy_version="v1", checkpoint_thread_id="c1",
        status="failed", attempt=3,
        created_at=now - timedelta(days=200),
        finished_at=now - timedelta(days=200),
    )
    # Old cancelled run without evaluation → should be deleted
    run2 = EvaluationRun(
        id=str(uuid.uuid4()), consultation_id=2, graph_version="v1",
        scoring_policy_version="v1", checkpoint_thread_id="c2",
        status="cancelled", attempt=0,
        created_at=now - timedelta(days=200),
        finished_at=now - timedelta(days=200),
    )
    # Old completed run → should NOT be deleted
    run3 = EvaluationRun(
        id=str(uuid.uuid4()), consultation_id=3, graph_version="v1",
        scoring_policy_version="v1", checkpoint_thread_id="c3",
        status="completed", attempt=1,
        created_at=now - timedelta(days=200),
        finished_at=now - timedelta(days=200),
    )
    # Recent failed run → should NOT be deleted
    run4 = EvaluationRun(
        id=str(uuid.uuid4()), consultation_id=4, graph_version="v1",
        scoring_policy_version="v1", checkpoint_thread_id="c4",
        status="failed", attempt=1,
        created_at=now - timedelta(days=10),
        finished_at=now - timedelta(days=10),
    )
    db_session.add_all([run1, run2, run3, run4])
    await db_session.flush()

    deleted = await cleanup_unreported_runs(db_session, now=now, retention_days=180)
    assert deleted == 2


# ── Test: audit auto-delete is off by default ────────────────────────────────


@pytest.mark.asyncio
async def test_audit_auto_delete_default_off():
    """Audit auto-delete should be disabled by default."""
    from app.core.config import Settings

    s = Settings()
    assert s.AUDIT_LOG_AUTO_DELETE_ENABLED is False


# ── Test: audit cleanup only when explicitly enabled ─────────────────────────


@pytest.mark.asyncio
async def test_audit_cleanup_requires_explicit_enable(db_session: AsyncSession):
    """Audit log cleanup should only run when auto_delete=True and policy_id is set."""
    from app.tasks.data_cleanup import maybe_cleanup_audit_logs

    now = _now()

    # Default: auto_delete=False → should not delete anything
    result = await maybe_cleanup_audit_logs(
        db_session,
        now=now,
        auto_delete_enabled=False,
        policy_id="",
        retention_days=90,
    )
    assert result["deleted"] == 0
    assert result["skipped"] is True
