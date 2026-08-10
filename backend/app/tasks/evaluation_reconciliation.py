# -*- coding: utf-8 -*-
"""Evaluation Reconciliation — 每 60 秒运行的 stale run/outbox 恢复

Reconciliation 语义：
- queued + pending/有效 leased outbox → 不处理
- queued + 过期 lease → 释放为 pending
- queued + published 超 5 分钟且从未 claim、总年龄未满 24 小时 → requeue_stale_dispatch
- 达 24 小时仍无 claim → outbox dead_letter、run failed(dispatch_unclaimed)
- running + lease 过期 → 锁定 run 清 owner
  - 有 cancel_requested_at → cancelled
  - 有剩余 attempt → retrying(worker_lost)
  - 耗尽 → failed(worker_lost)
- retrying + cancel → cancelled
- retrying + 2 分钟未 claim → published outbox 重置 pending
- retrying + 20 分钟未 claim → failed(redelivery_lost)
- needs_review/终态 → 不处理
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.celery_app import celery_app
from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
from app.models.evaluation_run import EvaluationRun
from app.tasks.async_runtime import run_worker_coroutine

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# 最大业务 attempt（超过则不再重试）
_MAX_BUSINESS_ATTEMPTS = 3

# stale published 阈值（秒）
_STALE_PUBLISHED_SECONDS = 300  # 5 min

# 24 小时未 claim 阈值（秒）
_MAX_UNCLAIMED_SECONDS = 86400  # 24h

# retrying rescue 阈值（秒）
_RETRYING_RESCUE_SECONDS = 120  # 2 min

# retrying redelivery lost 阈值（秒）
_RETRYING_LOST_SECONDS = 1200  # 20 min

# 终态集合
_TERMINAL_STATUSES = frozenset({"completed", "needs_review", "reviewed", "failed", "cancelled"})


# ── Reconciliation Logic ─────────────────────────────────────────────────────


async def reconcile_all(db: AsyncSession) -> dict:  # noqa: C901
    """执行一轮 reconciliation，返回处理统计"""
    now = datetime.utcnow()
    stats = {
        "queued_lease_released": 0,
        "queued_stale_requeued": 0,
        "queued_unclaimed_dead": 0,
        "running_expired_cancelled": 0,
        "running_expired_retrying": 0,
        "running_expired_failed": 0,
        "retrying_cancelled": 0,
        "retrying_rescued": 0,
        "retrying_lost": 0,
    }

    # ── 1. queued runs ───────────────────────────────────────────────────────
    queued_runs = await _get_runs_by_status(db, "queued")
    for run in queued_runs:
        outbox = await _get_outbox_for_run(db, run.id)
        if outbox is None:
            continue

        age_seconds = (now - run.created_at).total_seconds()

        # pending → 不处理
        if outbox.status == "pending":
            continue

        # 有效 leased → 不处理
        if outbox.status == "leased":
            if outbox.lease_expires_at and outbox.lease_expires_at > now:
                continue
            # 过期 lease → 释放为 pending
            await _release_outbox_lease(db, outbox, now)
            stats["queued_lease_released"] += 1
            continue

        # published → 检查 stale/unclaimed
        if outbox.status == "published":
            published_age = (now - (outbox.published_at or outbox.created_at)).total_seconds()

            # 24 小时仍未 claim → dead_letter + failed
            if age_seconds >= _MAX_UNCLAIMED_SECONDS:
                await _dead_letter_outbox(db, outbox, now)
                await _fail_run_unclaimed(db, run, now)
                stats["queued_unclaimed_dead"] += 1
                continue

            # 5 分钟 stale → requeue
            if published_age >= _STALE_PUBLISHED_SECONDS:
                await _requeue_stale(db, run.id)
                stats["queued_stale_requeued"] += 1
                continue

        # published/leased 有效 → 不处理
        continue

    # ── 2. running runs with expired lease ────────────────────────────────────
    running_runs = await _get_runs_by_status(db, "running")
    for run in running_runs:
        # 有效 lease → 不处理
        if run.lease_expires_at and run.lease_expires_at > now:
            continue
        # 无 lease 信息 → 跳过（可能刚创建）
        if run.lease_expires_at is None and run.execution_owner is None:
            continue

        # 锁定 run 清 owner
        await _clear_run_lease(db, run)

        # 有 cancel_requested_at → cancelled
        if run.cancel_requested_at is not None:
            await _cancel_run(db, run, now)
            stats["running_expired_cancelled"] += 1
            continue

        # 有剩余 attempt → retrying
        if run.attempt < _MAX_BUSINESS_ATTEMPTS:
            await _retry_run(db, run, "worker_lost")
            stats["running_expired_retrying"] += 1
            continue

        # 耗尽 → failed
        await _fail_run(db, run, "worker_lost", now)
        stats["running_expired_failed"] += 1

    # ── 3. retrying runs ─────────────────────────────────────────────────────
    retrying_runs = await _get_runs_by_status(db, "retrying")
    for run in retrying_runs:
        # 有取消意图 → cancelled
        if run.cancel_requested_at is not None:
            await _cancel_run_no_owner(db, run, now)
            stats["retrying_cancelled"] += 1
            continue

        outbox = await _get_outbox_for_run(db, run.id)
        if outbox is None:
            continue

        # 计算 retrying 状态持续时间
        retrying_age = (now - (run.updated_at or run.created_at)).total_seconds()

        # 20 分钟仍未 claim → failed(redelivery_lost)
        if retrying_age >= _RETRYING_LOST_SECONDS:
            await _fail_run_no_owner(db, run, "redelivery_lost", now)
            stats["retrying_lost"] += 1
            continue

        # 2 分钟 + published outbox → pending rescue
        if retrying_age >= _RETRYING_RESCUE_SECONDS and outbox.status == "published":
            await _requeue_stale(db, run.id)
            stats["retrying_rescued"] += 1
            continue

    await db.flush()
    return stats


# ── Internal Helpers ─────────────────────────────────────────────────────────


async def _get_runs_by_status(db: AsyncSession, status: str) -> list[EvaluationRun]:
    """获取指定状态的 runs"""
    stmt = select(EvaluationRun).where(EvaluationRun.status == status)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_outbox_for_run(db: AsyncSession, run_id: str) -> EvaluationDispatchOutbox | None:
    """获取 run 对应的 outbox"""
    stmt = select(EvaluationDispatchOutbox).where(
        EvaluationDispatchOutbox.run_id == run_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _release_outbox_lease(db: AsyncSession, outbox: EvaluationDispatchOutbox, now: datetime) -> None:
    """释放过期 outbox lease → pending"""
    outbox.status = "pending"
    outbox.lease_owner = None
    outbox.lease_expires_at = None
    outbox.next_attempt_at = now
    outbox.updated_at = now


async def _requeue_stale(db: AsyncSession, run_id: str) -> None:
    """调用 requeue_stale_dispatch 重置 published outbox → pending"""
    from app.services.evaluation_dispatch_service import requeue_stale_dispatch
    await requeue_stale_dispatch(db, run_id=run_id)


async def _dead_letter_outbox(db: AsyncSession, outbox: EvaluationDispatchOutbox, now: datetime) -> None:
    """outbox → dead_letter"""
    outbox.status = "dead_letter"
    outbox.updated_at = now


async def _fail_run_unclaimed(db: AsyncSession, run: EvaluationRun, now: datetime) -> None:
    """run → failed(dispatch_unclaimed)"""
    run.status = "failed"
    run.error_type = "dispatch_unclaimed"
    run.error_message = "Dispatch unclaimed after 24 hours"
    run.finished_at = now
    run.execution_owner = None
    run.execution_task_id = None
    run.heartbeat_at = None
    run.lease_expires_at = None


async def _clear_run_lease(db: AsyncSession, run: EvaluationRun) -> None:
    """清空 run 的执行租约"""
    run.execution_task_id = None
    run.execution_owner = None
    run.heartbeat_at = None
    run.lease_expires_at = None


async def _cancel_run(db: AsyncSession, run: EvaluationRun, now: datetime) -> None:
    """run → cancelled（有 owner 清理后）"""
    run.status = "cancelled"
    run.finished_at = now


async def _retry_run(db: AsyncSession, run: EvaluationRun, error_type: str) -> None:
    """run → retrying"""
    run.status = "retrying"
    run.error_type = error_type
    run.error_message = f"Worker lost, attempt {run.attempt}/{_MAX_BUSINESS_ATTEMPTS}"


async def _fail_run(db: AsyncSession, run: EvaluationRun, error_type: str, now: datetime) -> None:
    """run → failed"""
    run.status = "failed"
    run.error_type = error_type
    run.error_message = f"Worker lost, attempts exhausted ({run.attempt}/{_MAX_BUSINESS_ATTEMPTS})"
    run.finished_at = now


async def _cancel_run_no_owner(db: AsyncSession, run: EvaluationRun, now: datetime) -> None:
    """run → cancelled（无 owner）"""
    run.status = "cancelled"
    run.finished_at = now


async def _fail_run_no_owner(db: AsyncSession, run: EvaluationRun, error_type: str, now: datetime) -> None:
    """run → failed（无 owner）"""
    run.status = "failed"
    run.error_type = error_type
    run.error_message = f"Redelivery lost after {error_type}"
    run.finished_at = now


# ── Celery Task ──────────────────────────────────────────────────────────────


@celery_app.task(name="evaluation_reconciliation")
def evaluation_reconciliation_task() -> dict:
    """每 60 秒运行的 reconciliation Celery task"""
    logger.info("[Reconciliation] Starting reconciliation cycle")
    try:
        result = run_worker_coroutine(_do_reconcile())
        logger.info(f"[Reconciliation] Cycle complete: {result}")
        return result
    except Exception as e:
        logger.error(f"[Reconciliation] Cycle failed: {type(e).__name__}: {e}")
        return {"error": str(e)}


async def _do_reconcile() -> dict:
    """在 worker async runtime 中执行 reconciliation"""
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            stats = await reconcile_all(db)
            await db.commit()
            return stats
        except Exception:
            await db.rollback()
            raise
