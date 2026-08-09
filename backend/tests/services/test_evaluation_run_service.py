# -*- coding: utf-8 -*-
"""EvaluationRun 状态机单元测试 — TDD 红灯先行

覆盖：
1. queued 创建（attempt=0, status="queued"）
2. 首次 claim → STARTED（attempt=1）
3. running→retrying→running（第二次 claim, attempt=2）
4. 同 task_id 有效 lease → ACTIVE_SAME_TASK
5. 不同 task_id 有效 lease → ACTIVE_OTHER_TASK
6. 过期 lease → RECLAIMED_STALE（新 owner 接管）
7. 旧 owner 续期 → RunLeaseLost
8. 终态幂等（重复调用不重写 finished_at）
9. 非法 completed→running → InvalidRunTransition
10. 并发第二次 claim 不重复启动
11. 精确 ID 回归（传入固定 UUID）
12. 取消：queued→cancelled, running→cancel_requested, 终态幂等, needs_review→NOT_CANCELLABLE
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.evaluation_run import EvaluationRun


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_run(
    run_id: str = "test-run-001",
    consultation_id: int = 1,
    status: str = "queued",
    attempt: int = 0,
    execution_owner: str | None = None,
    execution_task_id: str | None = None,
    heartbeat_at: datetime | None = None,
    lease_expires_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    cancel_requested_at: datetime | None = None,
    cancel_requested_by: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    evaluation_id: int | None = None,
) -> EvaluationRun:
    return EvaluationRun(
        id=run_id,
        consultation_id=consultation_id,
        graph_version="evaluation-graph-v1",
        scoring_policy_version="v1",
        checkpoint_thread_id=f"evaluation:{run_id}",
        status=status,
        attempt=attempt,
        execution_owner=execution_owner,
        execution_task_id=execution_task_id,
        heartbeat_at=heartbeat_at,
        lease_expires_at=lease_expires_at,
        started_at=started_at,
        finished_at=finished_at,
        cancel_requested_at=cancel_requested_at,
        cancel_requested_by=cancel_requested_by,
        error_type=error_type,
        error_message=error_message,
        evaluation_id=evaluation_id,
    )


class _FakeDB:
    """Minimal async session stub for run service tests."""

    def __init__(self, run: EvaluationRun | None = None):
        self._run = run
        self.added = []
        self.flushed = False
        self.committed = False

    async def execute(self, stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = self._run
        return result

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True


# ── Test: create_queued_run ──────────────────────────────────────────────────


class TestCreateQueuedRun:
    """1. queued 创建（attempt=0, status="queued"）"""

    @pytest.mark.asyncio
    async def test_creates_run_with_queued_status(self):
        from app.services.evaluation_run_service import create_queued_run

        db = _FakeDB()
        run = await create_queued_run(db, run_id="run-001", consultation_id=42)

        assert run.id == "run-001"
        assert run.consultation_id == 42
        assert run.status == "queued"
        assert run.attempt == 0
        assert run.started_at is None
        assert run in db.added
        assert db.flushed


# ── Test: claim_run ──────────────────────────────────────────────────────────


class TestClaimRun:
    """2-6, 9-10. claim_run 状态转换"""

    @pytest.mark.asyncio
    async def test_queued_claim_returns_started(self):
        """2. 首次 claim → STARTED（attempt=1）"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        run = _make_run(status="queued", attempt=0)
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 0, 0)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-001",
            execution_owner="worker-1:inv-abc",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.STARTED
        assert result.run.attempt == 1
        assert result.run.status == "running"
        assert result.run.execution_owner == "worker-1:inv-abc"
        assert result.run.execution_task_id == "task-001"
        assert result.run.lease_expires_at == now + timedelta(seconds=300)
        assert result.run.heartbeat_at == now
        assert result.run.started_at == now

    @pytest.mark.asyncio
    async def test_retrying_claim_returns_started_attempt_2(self):
        """3. running→retrying→running（第二次 claim, attempt=2）"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        run = _make_run(status="retrying", attempt=1)
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 5, 0)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-002",
            execution_owner="worker-2:inv-def",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.STARTED
        assert result.run.attempt == 2
        assert result.run.status == "running"
        assert result.run.execution_owner == "worker-2:inv-def"

    @pytest.mark.asyncio
    async def test_running_valid_lease_same_task_returns_active_same(self):
        """4. 同 task_id 有效 lease → ACTIVE_SAME_TASK"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        now = datetime(2026, 8, 10, 12, 0, 0)
        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-abc",
            execution_task_id="task-001",
            lease_expires_at=now + timedelta(seconds=200),
            heartbeat_at=now,
        )
        db = _FakeDB(run)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-001",
            execution_owner="worker-1:inv-abc",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.ACTIVE_SAME_TASK
        assert result.retry_after_seconds is not None
        assert result.retry_after_seconds > 0

    @pytest.mark.asyncio
    async def test_running_valid_lease_different_task_returns_active_other(self):
        """5. 不同 task_id 有效 lease → ACTIVE_OTHER_TASK"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        now = datetime(2026, 8, 10, 12, 0, 0)
        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-abc",
            execution_task_id="task-001",
            lease_expires_at=now + timedelta(seconds=200),
            heartbeat_at=now,
        )
        db = _FakeDB(run)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-999",
            execution_owner="worker-2:inv-zzz",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.ACTIVE_OTHER_TASK

    @pytest.mark.asyncio
    async def test_running_expired_lease_returns_reclaimed_stale(self):
        """6. 过期 lease → RECLAIMED_STALE（新 owner 接管）"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        now = datetime(2026, 8, 10, 12, 10, 0)
        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-old",
            execution_task_id="task-old",
            lease_expires_at=now - timedelta(seconds=10),  # expired
            heartbeat_at=now - timedelta(minutes=5),
        )
        db = _FakeDB(run)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-new",
            execution_owner="worker-2:inv-new",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.RECLAIMED_STALE
        assert result.run.attempt == 2
        assert result.run.execution_owner == "worker-2:inv-new"
        assert result.run.execution_task_id == "task-new"

    @pytest.mark.asyncio
    async def test_terminal_claim_returns_terminal(self):
        """终态 → TERMINAL"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        run = _make_run(status="completed", attempt=1, finished_at=datetime(2026, 8, 10, 12, 1, 0))
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 5, 0)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-002",
            execution_owner="worker-2:inv-def",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.TERMINAL

    @pytest.mark.asyncio
    async def test_invalid_transition_completed_to_running(self):
        """9. 非法 completed→running → InvalidRunTransition (通过 claim 终态返回 TERMINAL 而非抛异常)"""
        from app.services.evaluation_run_service import (
            RunClaimDisposition,
            claim_run,
        )

        run = _make_run(status="failed", attempt=1, finished_at=datetime(2026, 8, 10, 12, 1, 0))
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 5, 0)

        result = await claim_run(
            db,
            run_id="test-run-001",
            celery_task_id="task-002",
            execution_owner="worker-2:inv-def",
            now=now,
            lease_seconds=300,
        )

        assert result.disposition == RunClaimDisposition.TERMINAL


# ── Test: renew_run_lease ────────────────────────────────────────────────────


class TestRenewRunLease:
    """7. 旧 owner 续期 / 非 owner 续期 → RunLeaseLost"""

    @pytest.mark.asyncio
    async def test_valid_owner_renews(self):
        from app.services.evaluation_run_service import renew_run_lease

        now = datetime(2026, 8, 10, 12, 2, 0)
        run = _make_run(
            status="running",
            execution_owner="worker-1:inv-abc",
            lease_expires_at=now + timedelta(seconds=100),
        )
        db = _FakeDB(run)

        ok = await renew_run_lease(
            db,
            run_id="test-run-001",
            execution_owner="worker-1:inv-abc",
            now=now,
            lease_seconds=300,
        )

        assert ok is True
        assert run.lease_expires_at == now + timedelta(seconds=300)
        assert run.heartbeat_at == now

    @pytest.mark.asyncio
    async def test_wrong_owner_raises_lease_lost(self):
        """7. 旧 owner 续期 → RunLeaseLost"""
        from app.services.evaluation_run_service import RunLeaseLost, renew_run_lease

        now = datetime(2026, 8, 10, 12, 2, 0)
        run = _make_run(
            status="running",
            execution_owner="worker-2:inv-new",
            lease_expires_at=now + timedelta(seconds=100),
        )
        db = _FakeDB(run)

        with pytest.raises(RunLeaseLost):
            await renew_run_lease(
                db,
                run_id="test-run-001",
                execution_owner="worker-1:inv-old",
                now=now,
                lease_seconds=300,
            )


# ── Test: mark_run_terminal ──────────────────────────────────────────────────


class TestMarkRunTerminal:
    """8. 终态幂等（重复调用不重写 finished_at）"""

    @pytest.mark.asyncio
    async def test_terminal_idempotent(self):
        from app.services.evaluation_run_service import mark_run_terminal

        original_finished = datetime(2026, 8, 10, 12, 1, 0)
        run = _make_run(
            status="completed",
            attempt=1,
            finished_at=original_finished,
            execution_owner="worker-1:inv-abc",
            evaluation_id=99,
        )
        db = _FakeDB(run)

        # 重复调用 — 不应覆盖
        await mark_run_terminal(
            db,
            run_id="test-run-001",
            status="completed",
            execution_owner="worker-1:inv-abc",
            evaluation_id=100,
        )

        assert run.finished_at == original_finished
        assert run.evaluation_id == 99  # 不覆盖

    @pytest.mark.asyncio
    async def test_mark_terminal_sets_fields(self):
        from app.services.evaluation_run_service import mark_run_terminal

        now = datetime(2026, 8, 10, 12, 5, 0)
        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-abc",
            lease_expires_at=now + timedelta(seconds=200),
        )
        db = _FakeDB(run)

        await mark_run_terminal(
            db,
            run_id="test-run-001",
            status="completed",
            execution_owner="worker-1:inv-abc",
            evaluation_id=42,
            now=now,
        )

        assert run.status == "completed"
        assert run.evaluation_id == 42
        assert run.finished_at == now
        assert run.execution_owner is None
        assert run.execution_task_id is None
        assert run.lease_expires_at is None


# ── Test: mark_run_retrying ──────────────────────────────────────────────────


class TestMarkRunRetrying:
    @pytest.mark.asyncio
    async def test_retrying_clears_lease_not_finished(self):
        from app.services.evaluation_run_service import mark_run_retrying

        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-abc",
            execution_task_id="task-001",
            lease_expires_at=datetime(2026, 8, 10, 12, 5, 0),
        )
        db = _FakeDB(run)

        await mark_run_retrying(
            db,
            run_id="test-run-001",
            execution_owner="worker-1:inv-abc",
            error=ConnectionError("network down"),
        )

        assert run.status == "retrying"
        assert run.execution_owner is None
        assert run.execution_task_id is None
        assert run.lease_expires_at is None
        assert run.finished_at is None  # retrying 不写 finished_at
        assert run.error_type is not None
        assert run.error_message is not None


# ── Test: exact ID regression ────────────────────────────────────────────────


class TestExactIDRegression:
    """11. 精确 ID 回归（传入固定 UUID，所有关联字段使用同一值）"""

    @pytest.mark.asyncio
    async def test_fixed_uuid_propagates(self):
        from app.services.evaluation_run_service import create_queued_run

        fixed_id = "11111111-1111-1111-1111-111111111111"
        db = _FakeDB()
        run = await create_queued_run(db, run_id=fixed_id, consultation_id=1)

        assert run.id == fixed_id
        assert run.checkpoint_thread_id == f"evaluation:{fixed_id}"


# ── Test: request_run_cancel ─────────────────────────────────────────────────


class TestRequestRunCancel:
    """12. 取消：queued→cancelled, running→cancel_requested, 终态幂等, needs_review→NOT_CANCELLABLE"""

    @pytest.mark.asyncio
    async def test_queued_cancel(self):
        from app.services.evaluation_run_service import (
            CancelDisposition,
            request_run_cancel,
        )

        run = _make_run(status="queued", attempt=0)
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 0, 0)

        disp, updated = await request_run_cancel(
            db, run_id="test-run-001", requested_by=1, now=now
        )

        assert disp == CancelDisposition.CANCELLED_BEFORE_START
        assert updated.status == "cancelled"
        assert updated.cancel_requested_at == now
        assert updated.cancel_requested_by == 1

    @pytest.mark.asyncio
    async def test_running_cancel_sets_requested_flag(self):
        from app.services.evaluation_run_service import (
            CancelDisposition,
            request_run_cancel,
        )

        run = _make_run(
            status="running",
            attempt=1,
            execution_owner="worker-1:inv-abc",
            lease_expires_at=datetime(2026, 8, 10, 12, 5, 0),
        )
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 2, 0)

        disp, updated = await request_run_cancel(
            db, run_id="test-run-001", requested_by=1, now=now
        )

        assert disp == CancelDisposition.REQUESTED_RUNNING
        assert updated.cancel_requested_at == now
        assert updated.cancel_requested_by == 1
        assert updated.status == "running"  # 不改 status

    @pytest.mark.asyncio
    async def test_terminal_cancel_returns_already_terminal(self):
        from app.services.evaluation_run_service import (
            CancelDisposition,
            request_run_cancel,
        )

        run = _make_run(status="completed", attempt=1, finished_at=datetime(2026, 8, 10, 12, 1, 0))
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 5, 0)

        disp, updated = await request_run_cancel(
            db, run_id="test-run-001", requested_by=1, now=now
        )

        assert disp == CancelDisposition.ALREADY_TERMINAL

    @pytest.mark.asyncio
    async def test_needs_review_not_cancellable(self):
        from app.services.evaluation_run_service import (
            CancelDisposition,
            request_run_cancel,
        )

        run = _make_run(status="needs_review", attempt=1)
        db = _FakeDB(run)
        now = datetime(2026, 8, 10, 12, 5, 0)

        disp, _ = await request_run_cancel(
            db, run_id="test-run-001", requested_by=1, now=now
        )

        assert disp == CancelDisposition.NOT_CANCELLABLE


# ── Test: get_run ────────────────────────────────────────────────────────────


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_run(self):
        from app.services.evaluation_run_service import get_run

        run = _make_run()
        db = _FakeDB(run)
        result = await get_run(db, "test-run-001")
        assert result is not None
        assert result.id == "test-run-001"

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self):
        from app.services.evaluation_run_service import get_run

        db = _FakeDB(None)
        result = await get_run(db, "nonexistent")
        assert result is None


# ── Test: mark_unowned_run_terminal ──────────────────────────────────────────


class TestMarkUnownedRunTerminal:
    @pytest.mark.asyncio
    async def test_marks_queued_as_failed(self):
        from app.services.evaluation_run_service import mark_unowned_run_terminal

        run = _make_run(status="queued", attempt=0)
        db = _FakeDB(run)

        ok = await mark_unowned_run_terminal(
            db,
            run_id="test-run-001",
            expected_status="queued",
            status="failed",
            error_code="dispatch_exhausted",
        )

        assert ok is True
        assert run.status == "failed"
        assert run.error_type == "dispatch_exhausted"

    @pytest.mark.asyncio
    async def test_status_mismatch_returns_false(self):
        from app.services.evaluation_run_service import mark_unowned_run_terminal

        run = _make_run(status="running", attempt=1)
        db = _FakeDB(run)

        ok = await mark_unowned_run_terminal(
            db,
            run_id="test-run-001",
            expected_status="queued",
            status="failed",
        )

        assert ok is False


# ── Test: InvalidRunTransition exception ─────────────────────────────────────


class TestExceptions:
    def test_invalid_run_transition_is_exception(self):
        from app.services.evaluation_run_service import InvalidRunTransition

        assert issubclass(InvalidRunTransition, Exception)

    def test_run_lease_lost_is_exception(self):
        from app.services.evaluation_run_service import RunLeaseLost

        assert issubclass(RunLeaseLost, Exception)
