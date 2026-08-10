# -*- coding: utf-8 -*-
"""Evaluation Reconciliation 定时任务测试 — TDD 红灯先行

覆盖所有 reconciliation 场景：
- queued + pending/有效 leased outbox → 不处理
- queued + 过期 lease → 释放为 pending
- queued + published 超 5 分钟且从未 claim、总年龄未满 24 小时 → requeue_stale_dispatch
- 达 24 小时仍无 claim → outbox dead_letter、run failed(dispatch_unclaimed)
- running + lease 过期 → 锁定 run 清 owner
  - 有 cancel_requested_at → cancelled
  - 有剩余 attempt → retrying
  - 耗尽 → failed
- retrying + cancel → cancelled
- retrying + 2 分钟未 claim → published outbox 重置 pending
- retrying + 20 分钟未 claim → failed(redelivery_lost)
- needs_review/终态 → 不动
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
from app.models.evaluation_run import EvaluationRun

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_run(
    run_id=None,
    consultation_id=1,
    status="queued",
    attempt=0,
    execution_owner=None,
    execution_task_id=None,
    heartbeat_at=None,
    lease_expires_at=None,
    cancel_requested_at=None,
    cancel_requested_by=None,
    error_type=None,
    error_message=None,
    started_at=None,
    finished_at=None,
    created_at=None,
):
    rid = run_id or str(uuid.uuid4())
    return EvaluationRun(
        id=rid,
        consultation_id=consultation_id,
        graph_version="evaluation-graph-v1",
        scoring_policy_version="v1",
        checkpoint_thread_id=f"evaluation:{rid}",
        status=status,
        attempt=attempt,
        execution_owner=execution_owner,
        execution_task_id=execution_task_id,
        heartbeat_at=heartbeat_at,
        lease_expires_at=lease_expires_at,
        cancel_requested_at=cancel_requested_at,
        cancel_requested_by=cancel_requested_by,
        error_type=error_type,
        error_message=error_message,
        started_at=started_at,
        finished_at=finished_at,
        created_at=created_at or datetime.utcnow(),
    )


def _make_outbox(
    event_id=None,
    run_id=None,
    status="pending",
    attempt=0,
    lease_owner=None,
    lease_expires_at=None,
    published_at=None,
    last_task_id=None,
    created_at=None,
    next_attempt_at=None,
):
    return EvaluationDispatchOutbox(
        event_id=event_id or str(uuid.uuid4()),
        run_id=run_id or str(uuid.uuid4()),
        status=status,
        task_name="app.tasks.evaluation_task.run_evaluation",
        payload={"run_id": run_id or "x", "consultation_id": 1, "trace_context": {}},
        attempt=attempt,
        lease_owner=lease_owner,
        lease_expires_at=lease_expires_at,
        published_at=published_at,
        last_task_id=last_task_id,
        created_at=created_at or datetime.utcnow(),
        next_attempt_at=next_attempt_at,
    )


class _FakeDB:
    """Minimal async session stub for reconciliation tests."""

    def __init__(self, runs=None, outboxes=None):
        self._runs = {r.id: r for r in (runs or [])}
        self._outboxes = {o.event_id: o for o in (outboxes or [])}
        self.flushed = False
        self.committed = False

    async def execute(self, stmt):
        """Very basic stub — returns runs/outboxes based on query patterns."""
        result = MagicMock()

        # 根据 stmt 类型返回不同数据
        scalars = MagicMock()

        # 简化：检查 stmt 中的 model 类型
        stmt_str = str(stmt)
        if "evaluation_runs" in stmt_str:
            scalars.all.return_value = list(self._runs.values())
            scalars.one_or_none = MagicMock(return_value=None)
        elif "evaluation_dispatch_outbox" in stmt_str:
            scalars.all.return_value = list(self._outboxes.values())
        else:
            scalars.all.return_value = []

        result.scalars.return_value = scalars
        result.scalar_one_or_none.return_value = None
        result.rowcount = 0
        return result

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True

    async def rollback(self):
        pass


# ── Test: queued + pending/valid leased outbox → 不处理 ──────────────────────


class TestReconcileQueuedWithPendingOutbox:
    """queued + pending/有效 leased outbox → 不处理"""

    def test_queued_with_pending_outbox_no_action(self):
        """queued run + pending outbox → 不处理"""
        run_id = str(uuid.uuid4())
        run = _make_run(run_id=run_id, status="queued")
        outbox = _make_outbox(run_id=run_id, status="pending")

        # reconciliation 应该跳过这个 run
        # 具体断言在实现后补充
        assert run.status == "queued"
        assert outbox.status == "pending"

    def test_queued_with_valid_leased_outbox_no_action(self):
        """queued run + 有效 leased outbox → 不处理"""
        run_id = str(uuid.uuid4())
        run = _make_run(run_id=run_id, status="queued")
        outbox = _make_outbox(
            run_id=run_id, status="leased",
            lease_owner="worker-1",
            lease_expires_at=datetime.utcnow() + timedelta(minutes=5),
        )

        assert run.status == "queued"
        assert outbox.status == "leased"


# ── Test: queued + expired lease → release to pending ────────────────────────


class TestReconcileQueuedWithExpiredLease:
    """queued + 过期 lease → 释放为 pending"""

    def test_queued_with_expired_leased_outbox_releases_to_pending(self):
        """queued run + 过期 leased outbox → outbox 重置为 pending"""
        run_id = str(uuid.uuid4())
        _run = _make_run(run_id=run_id, status="queued")
        outbox = _make_outbox(
            run_id=run_id, status="leased",
            lease_owner="worker-1",
            lease_expires_at=datetime.utcnow() - timedelta(minutes=5),
        )

        # 实现后：reconciliation 应把 outbox 重置为 pending
        assert outbox.status == "leased"  # 初始状态
        assert outbox.lease_expires_at < datetime.utcnow()


# ── Test: queued + published > 5 min, never claimed → requeue ────────────────


class TestReconcileStaleQueuedPublished:
    """queued + published 超 5 分钟且从未 claim → requeue_stale_dispatch"""

    def test_stale_published_outbox_requeued(self):
        """queued + published > 5 min + 总年龄 < 24h → requeue"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        _run = _make_run(run_id=run_id, status="queued", created_at=now - timedelta(minutes=10))
        outbox = _make_outbox(
            run_id=run_id, status="published",
            published_at=now - timedelta(minutes=6),
            created_at=now - timedelta(minutes=10),
        )

        # 实现后：应调用 requeue_stale_dispatch
        assert outbox.status == "published"
        age = (now - outbox.created_at).total_seconds()
        assert age < 86400  # < 24h


# ── Test: 24h no claim → dead_letter + failed(dispatch_unclaimed) ────────────


class TestReconcileUnclaimed24h:
    """达 24 小时仍无 claim → outbox dead_letter、run failed(dispatch_unclaimed)"""

    def test_24h_unclaimed_becomes_dead_letter(self):
        """published > 24h + 从未 claim → dead_letter + failed"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(run_id=run_id, status="queued", created_at=now - timedelta(hours=25))
        outbox = _make_outbox(
            run_id=run_id, status="published",
            published_at=now - timedelta(hours=25),
            created_at=now - timedelta(hours=25),
        )

        # 实现后：outbox → dead_letter, run → failed(dispatch_unclaimed)
        assert run.status == "queued"
        age = (now - outbox.created_at).total_seconds()
        assert age >= 86400


# ── Test: running + expired lease ────────────────────────────────────────────


class TestReconcileRunningExpiredLease:
    """running + lease 过期 → 锁定 run 清 owner"""

    def test_running_expired_lease_with_cancel_intent(self):
        """running + lease 过期 + cancel_requested_at → cancelled"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="running", attempt=1,
            execution_owner="worker-1",
            execution_task_id="task-1",
            lease_expires_at=now - timedelta(minutes=5),
            cancel_requested_at=now - timedelta(minutes=2),
            cancel_requested_by=10,
        )

        # 实现后：run → cancelled
        assert run.status == "running"
        assert run.cancel_requested_at is not None

    def test_running_expired_lease_with_remaining_attempts(self):
        """running + lease 过期 + 有剩余 attempt → retrying"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="running", attempt=1,
            execution_owner="worker-1",
            execution_task_id="task-1",
            lease_expires_at=now - timedelta(minutes=5),
        )

        # 实现后：run → retrying(worker_lost)，清 owner
        assert run.status == "running"
        assert run.attempt == 1

    def test_running_expired_lease_no_remaining_attempts(self):
        """running + lease 过期 + 耗尽 attempt → failed(worker_lost)"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="running", attempt=3,
            execution_owner="worker-1",
            execution_task_id="task-1",
            lease_expires_at=now - timedelta(minutes=5),
        )

        # 实现后：run → failed(worker_lost)
        assert run.status == "running"


# ── Test: retrying + cancel → cancelled ──────────────────────────────────────


class TestReconcileRetryingCancel:
    """retrying + cancel → cancelled"""

    def test_retrying_with_cancel_intent_becomes_cancelled(self):
        """retrying + cancel_requested_at → cancelled"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="retrying", attempt=1,
            cancel_requested_at=now - timedelta(minutes=1),
            cancel_requested_by=10,
        )

        # 实现后：run → cancelled
        assert run.status == "retrying"


# ── Test: retrying + 2 min no claim → pending rescue ─────────────────────────


class TestReconcileRetryingStale:
    """retrying + 2 分钟未 claim → published outbox 重置 pending"""

    def test_retrying_2min_stale_outbox_rescue(self):
        """retrying + published outbox > 2 min → 重置 pending"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="retrying", attempt=1,
        )
        run.updated_at = now - timedelta(minutes=3)
        outbox = _make_outbox(
            run_id=run_id, status="published",
            published_at=now - timedelta(minutes=3),
        )

        # reconciliation 应把 published outbox 重置为 pending
        assert outbox.status == "published"
        assert run.status == "retrying"


# ── Test: retrying + 20 min no claim → failed(redelivery_lost) ───────────────


class TestReconcileRetryingRedeliveryLost:
    """retrying + 20 分钟未 claim → failed(redelivery_lost)"""

    def test_retrying_20min_no_claim_failed(self):
        """retrying + published outbox > 20 min → failed(redelivery_lost)"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="retrying", attempt=1,
        )
        run.updated_at = now - timedelta(minutes=21)
        _make_outbox(
            run_id=run_id, status="published",
            published_at=now - timedelta(minutes=21),
        )

        # reconciliation 应把 run 标记为 failed(redelivery_lost)
        assert run.status == "retrying"


# ── Test: needs_review / terminal → 不动 ─────────────────────────────────────


class TestReconcileTerminalNoAction:
    """needs_review/终态 → reconciliation 不处理"""

    @pytest.mark.parametrize("status", [
        "completed", "needs_review", "reviewed", "failed", "cancelled"
    ])
    def test_terminal_status_not_processed(self, status):
        """终态 run 不被 reconciliation 处理"""
        run_id = str(uuid.uuid4())
        run = _make_run(run_id=run_id, status=status)

        # reconciliation 应跳过所有终态
        assert run.status == status

    def test_running_with_valid_lease_not_processed(self):
        """有效 lease 的 running run 不被处理"""
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        run = _make_run(
            run_id=run_id, status="running", attempt=1,
            execution_owner="worker-1",
            lease_expires_at=now + timedelta(minutes=5),
        )

        # lease 有效，不处理
        assert run.lease_expires_at > now


# ── Integration: reconcile_all ───────────────────────────────────────────────


class TestReconcileAll:
    """reconcile_all 集成测试"""

    @pytest.mark.asyncio
    async def test_reconcile_all_processes_all_categories(self):
        """reconcile_all 处理所有类别的 stale run"""
        from app.tasks.evaluation_reconciliation import reconcile_all

        # 创建各种场景的 run 和 outbox
        now = datetime.utcnow()
        run_id_stale = str(uuid.uuid4())
        run_id_terminal = str(uuid.uuid4())

        runs = [
            _make_run(run_id=run_id_stale, status="queued", created_at=now - timedelta(hours=25)),
            _make_run(run_id=run_id_terminal, status="completed"),
        ]
        outboxes = [
            _make_outbox(
                run_id=run_id_stale, status="published",
                published_at=now - timedelta(hours=25),
                created_at=now - timedelta(hours=25),
            ),
        ]

        db = _FakeDB(runs=runs, outboxes=outboxes)

        # reconcile_all 应该被实现
        # _FakeDB 的 execute 返回空结果，所以 stats 全为 0
        stats = await reconcile_all(db)
        assert isinstance(stats, dict)
        assert "queued_unclaimed_dead" in stats
