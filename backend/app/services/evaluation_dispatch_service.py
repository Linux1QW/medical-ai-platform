# -*- coding: utf-8 -*-
"""EvaluationDispatch Outbox Service — Transactional Outbox 派发服务

提供 enqueue/claim/acknowledge/reject/cancel/requeue/purge 接口。
所有写操作只 flush、不 commit，由调用方控制事务边界。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation_dispatch_outbox import (
    EvaluationDispatchOutbox,
    validate_payload,
)
from app.models.evaluation_run import EvaluationRun
from app.services.observability.metrics import (
    EVALUATION_DISPATCH_DEAD_LETTER_TOTAL,
    EVALUATION_OUTBOX_EVENTS_TOTAL,
    EVALUATION_OUTBOX_PENDING,
)


# ── Exceptions ────────────────────────────────────────────────────────────────


class DispatchLeaseLost(Exception):
    """Outbox 租约不匹配（owner 或 status 已变更）"""
    pass


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DispatchLease:
    """claim_dispatch_batch 返回的租约信息"""
    event_id: str
    run_id: str
    task_name: str
    payload: dict
    attempt: int
    lease_owner: str


# ── Constants ─────────────────────────────────────────────────────────────────

# 可 claim 的状态
_CLAIMABLE_STATUSES = frozenset({"pending"})

# 终态
_TERMINAL_STATUSES = frozenset({"cancelled", "dead_letter"})

# 可取消的状态
_CANCELLABLE_STATUSES = frozenset({"pending", "leased", "published"})

# 默认最大重试次数
_DEFAULT_MAX_ATTEMPTS = 1500

# 默认最大 age（秒）
_DEFAULT_MAX_AGE_SECONDS = 86400  # 24h

# Celery task 名称
_EVAL_TASK_NAME = "app.tasks.evaluation_task.run_evaluation"


# ── Backoff calculation ───────────────────────────────────────────────────────


def compute_backoff(attempts: int, jitter: float = 0.0) -> float:
    """计算退避时间：min(2 ** attempts, 60) + jitter"""
    base = min(2 ** attempts, 60)
    return float(base) + jitter


# ── Service functions ─────────────────────────────────────────────────────────


async def enqueue_dispatch(
    db: AsyncSession,
    *,
    run_id: str,
    consultation_id: int,
    trace_context: dict,
) -> EvaluationDispatchOutbox:
    """将评估任务写入 outbox（只 flush，不 commit）

    payload 通过 allowlist validator 拒绝 PII 字段。
    与 run 在同一事务中写入，确保原子性。
    """
    payload = {
        "run_id": run_id,
        "consultation_id": consultation_id,
        "trace_context": trace_context,
    }
    # 校验 payload — 拒绝 PII
    validate_payload(payload)

    outbox = EvaluationDispatchOutbox(
        event_id=str(uuid.uuid4()),
        run_id=run_id,
        status="pending",
        task_name=_EVAL_TASK_NAME,
        payload=payload,
        attempt=0,
        next_attempt_at=datetime.utcnow(),
    )
    db.add(outbox)
    await db.flush()
    EVALUATION_OUTBOX_PENDING.inc()
    return outbox


async def claim_dispatch_batch(
    db: AsyncSession,
    *,
    worker_id: str,
    now: datetime,
    batch_size: int = 20,
    lease_seconds: int = 30,
) -> list[DispatchLease]:
    """批量 claim pending 或 lease 过期的 outbox 行

    SQLite 不支持 SELECT FOR UPDATE SKIP LOCKED，此处用普通 SELECT 模拟。
    生产环境（MySQL）应使用 FOR UPDATE SKIP LOCKED。
    """
    # 查找可 claim 的行：
    # 1. status=pending AND (next_attempt_at IS NULL OR next_attempt_at <= now)
    # 2. status=leased AND lease_expires_at <= now（过期租约回收）
    stmt = (
        select(EvaluationDispatchOutbox)
        .where(
            or_(
                # pending 且到达重试时间
                and_(
                    EvaluationDispatchOutbox.status == "pending",
                    or_(
                        EvaluationDispatchOutbox.next_attempt_at.is_(None),
                        EvaluationDispatchOutbox.next_attempt_at <= now,
                    ),
                ),
                # leased 但租约过期
                and_(
                    EvaluationDispatchOutbox.status == "leased",
                    EvaluationDispatchOutbox.lease_expires_at <= now,
                ),
            )
        )
        .limit(batch_size)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    leases: list[DispatchLease] = []
    lease_expires = now + timedelta(seconds=lease_seconds)

    for row in rows:
        row.status = "leased"
        row.lease_owner = worker_id
        row.lease_expires_at = lease_expires
        row.updated_at = now
        leases.append(
            DispatchLease(
                event_id=row.event_id,
                run_id=row.run_id,
                task_name=row.task_name,
                payload=row.payload,
                attempt=row.attempt,
                lease_owner=worker_id,
            )
        )

    if leases:
        await db.flush()
        EVALUATION_OUTBOX_PENDING.dec(len(leases))

    return leases


async def acknowledge_dispatch(
    db: AsyncSession,
    *,
    event_id: str,
    lease_owner: str,
    celery_task_id: str,
    published_at: datetime,
) -> None:
    """确认 outbox 已成功 publish

    匹配 event_id + lease_owner + status=leased，0 行时抛 DispatchLeaseLost。
    """
    stmt = (
        update(EvaluationDispatchOutbox)
        .where(
            and_(
                EvaluationDispatchOutbox.event_id == event_id,
                EvaluationDispatchOutbox.lease_owner == lease_owner,
                EvaluationDispatchOutbox.status == "leased",
            )
        )
        .values(
            status="published",
            last_task_id=celery_task_id,
            published_at=published_at,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=published_at,
        )
    )
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise DispatchLeaseLost(
            f"Dispatch {event_id}: lease lost (owner={lease_owner}, status != leased)"
        )
    await db.flush()
    EVALUATION_OUTBOX_EVENTS_TOTAL.labels(result="published").inc()


async def reject_dispatch(
    db: AsyncSession,
    *,
    event_id: str,
    lease_owner: str,
    error_code: str,
    now: datetime,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS,
) -> str:
    """拒绝 outbox（publish 失败）

    - 未达上限：重置为 pending，attempt+1，计算 next_attempt_at
    - 达到上限：设为 dead_letter，若 run 仍 queued 则标记 failed

    返回新状态。
    """
    # 获取当前行
    stmt = select(EvaluationDispatchOutbox).where(
        EvaluationDispatchOutbox.event_id == event_id
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        raise DispatchLeaseLost(f"Dispatch {event_id} not found")

    if row.lease_owner != lease_owner:
        raise DispatchLeaseLost(
            f"Dispatch {event_id}: owner mismatch (expected={lease_owner}, got={row.lease_owner})"
        )

    row.attempt += 1
    row.last_error_code = error_code
    row.lease_owner = None
    row.lease_expires_at = None
    row.updated_at = now

    # 判断是否达到上限
    age_seconds = (now - row.created_at).total_seconds()
    exhausted = row.attempt > max_attempts or age_seconds >= max_age_seconds

    if exhausted:
        row.status = "dead_letter"
        EVALUATION_DISPATCH_DEAD_LETTER_TOTAL.labels(reason=error_code).inc()
        EVALUATION_OUTBOX_EVENTS_TOTAL.labels(result="dead_letter").inc()
        # 若 run 仍 queued/retrying，标记为 failed
        await _maybe_fail_run(db, run_id=row.run_id, now=now)
    else:
        row.status = "pending"
        backoff = compute_backoff(row.attempt)
        row.next_attempt_at = now + timedelta(seconds=backoff)
        EVALUATION_OUTBOX_PENDING.inc()
        EVALUATION_OUTBOX_EVENTS_TOTAL.labels(result="rejected").inc()

    await db.flush()
    return row.status


async def cancel_dispatch(
    db: AsyncSession,
    *,
    run_id: str,
) -> bool:
    """取消 run 对应的 outbox（只 flush，不 commit）

    将 pending/leased/published 原子转为 cancelled。
    返回是否有行被更新。
    """
    stmt = (
        update(EvaluationDispatchOutbox)
        .where(
            and_(
                EvaluationDispatchOutbox.run_id == run_id,
                EvaluationDispatchOutbox.status.in_(_CANCELLABLE_STATUSES),
            )
        )
        .values(
            status="cancelled",
            lease_owner=None,
            lease_expires_at=None,
            updated_at=datetime.utcnow(),
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    if result.rowcount > 0:
        EVALUATION_OUTBOX_EVENTS_TOTAL.labels(result="cancelled").inc()
        EVALUATION_OUTBOX_PENDING.dec(result.rowcount)
    return result.rowcount > 0


async def requeue_stale_dispatch(
    db: AsyncSession,
    *,
    run_id: str,
) -> bool:
    """将 stale published outbox 重新排队

    前提条件：
    - run 状态为 queued 或 retrying
    - outbox 状态为 published
    - 无有效 Worker lease（lease_expires_at IS NULL 或已过期）

    不创建新 event，只重置同一行的状态。
    """
    # 检查 run 状态
    run_stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    run_result = await db.execute(run_stmt)
    run = run_result.scalar_one_or_none()

    if run is None:
        return False
    if run.status not in ("queued", "retrying"):
        return False

    now = datetime.utcnow()

    # 更新 outbox
    stmt = (
        update(EvaluationDispatchOutbox)
        .where(
            and_(
                EvaluationDispatchOutbox.run_id == run_id,
                EvaluationDispatchOutbox.status == "published",
                or_(
                    EvaluationDispatchOutbox.lease_expires_at.is_(None),
                    EvaluationDispatchOutbox.lease_expires_at <= now,
                ),
            )
        )
        .values(
            status="pending",
            next_attempt_at=now,
            attempt=0,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=now,
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    if result.rowcount > 0:
        EVALUATION_OUTBOX_EVENTS_TOTAL.labels(result="requeued").inc()
        EVALUATION_OUTBOX_PENDING.inc(result.rowcount)
    return result.rowcount > 0


async def purge_terminal_dispatches(
    db: AsyncSession,
    *,
    older_than: datetime,
) -> int:
    """清理终态 outbox 行（published/cancelled/dead_letter）

    只删除 created_at < older_than 的行。返回删除行数。
    """
    stmt = (
        delete(EvaluationDispatchOutbox)
        .where(
            and_(
                EvaluationDispatchOutbox.status.in_({"published", "cancelled", "dead_letter"}),
                EvaluationDispatchOutbox.created_at < older_than,
            )
        )
    )
    result = await db.execute(stmt)
    await db.flush()
    return result.rowcount


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _maybe_fail_run(
    db: AsyncSession,
    *,
    run_id: str,
    now: datetime,
) -> None:
    """若 run 仍为 queued/retrying 且无活跃租约，标记为 failed(dispatch_exhausted)"""
    run_stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    run_result = await db.execute(run_stmt)
    run = run_result.scalar_one_or_none()

    if run is None:
        return

    # 只在 queued/retrying 且无活跃租约时标记 failed
    if run.status in ("queued", "retrying") and run.execution_owner is None:
        run.status = "failed"
        run.error_type = "dispatch_exhausted"
        run.error_message = "Dispatch attempts exhausted (max attempts or max age reached)"
        run.finished_at = now
        await db.flush()
