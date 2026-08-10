# -*- coding: utf-8 -*-
"""EvaluationRun 生命周期状态机 — 持久化 run 管理

集中管理 run 的创建、claim（租约获取）、续期、重试、终态和取消。
所有状态转换在此模块集中定义，非法转换抛 InvalidRunTransition，
owner 不匹配抛 RunLeaseLost。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation_run import EvaluationRun
from app.services.observability.metrics import (
    EVALUATION_ACTIVE_RUNS,
    EVALUATION_CANCELLATIONS_TOTAL,
    EVALUATION_RETRIES_TOTAL,
    EVALUATION_RUN_DURATION,
    EVALUATION_RUNS_TOTAL,
)

# ── Exceptions ───────────────────────────────────────────────────────────────


class InvalidRunTransition(Exception):
    """非法状态转换"""
    pass


class RunLeaseLost(Exception):
    """execution_owner 不匹配，租约已丢失"""
    pass


# ── Enums ────────────────────────────────────────────────────────────────────


class RunClaimDisposition(str, Enum):
    STARTED = "started"
    RECLAIMED_STALE = "reclaimed_stale"
    ACTIVE_SAME_TASK = "active_same_task"
    ACTIVE_OTHER_TASK = "active_other_task"
    TERMINAL = "terminal"


class CancelDisposition(str, Enum):
    CANCELLED_BEFORE_START = "cancelled_before_start"
    REQUESTED_RUNNING = "requested_running"
    ALREADY_TERMINAL = "already_terminal"
    NOT_CANCELLABLE = "not_cancellable"


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunClaimResult:
    disposition: RunClaimDisposition
    run: EvaluationRun
    retry_after_seconds: int | None = None


# ── Constants ────────────────────────────────────────────────────────────────

_TERMINAL_STATUSES = frozenset({"completed", "needs_review", "failed", "cancelled", "reviewed"})

_CLAIMABLE_STATUSES = frozenset({"queued", "retrying"})


# ── Service functions ────────────────────────────────────────────────────────


async def get_run(
    db: AsyncSession, run_id: str, *, for_update: bool = False
) -> EvaluationRun | None:
    """获取单个 run，可选 FOR UPDATE 锁行"""
    stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def create_queued_run(
    db: AsyncSession, *, run_id: str, consultation_id: int
) -> EvaluationRun:
    """创建 queued 状态的 run（attempt=0, started_at=None）"""
    run = EvaluationRun(
        id=run_id,
        consultation_id=consultation_id,
        graph_version="evaluation-graph-v1",
        scoring_policy_version="v1",
        checkpoint_thread_id=f"evaluation:{run_id}",
        status="queued",
        attempt=0,
        started_at=None,
    )
    db.add(run)
    await db.flush()
    return run


async def claim_run(
    db: AsyncSession,
    *,
    run_id: str,
    celery_task_id: str,
    execution_owner: str,
    now: datetime,
    lease_seconds: int,
) -> RunClaimResult:
    """尝试 claim 一个 run 的执行租约

    状态转换：
    - queued/retrying → STARTED（attempt+1）
    - running + lease 过期 → RECLAIMED_STALE（attempt+1，新 owner 接管）
    - running + lease 有效 + 同 task_id → ACTIVE_SAME_TASK
    - running + lease 有效 + 不同 task_id → ACTIVE_OTHER_TASK
    - 终态 → TERMINAL
    """
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise InvalidRunTransition(f"Run {run_id} not found")

    # 终态 → TERMINAL
    if run.status in _TERMINAL_STATUSES:
        return RunClaimResult(
            disposition=RunClaimDisposition.TERMINAL,
            run=run,
        )

    # queued/retrying → STARTED
    if run.status in _CLAIMABLE_STATUSES:
        run.status = "running"
        run.attempt += 1
        run.execution_task_id = celery_task_id
        run.execution_owner = execution_owner
        run.heartbeat_at = now
        run.lease_expires_at = now + timedelta(seconds=lease_seconds)
        if run.started_at is None:
            run.started_at = now
        # 清除重试相关字段
        run.error_type = None
        run.error_message = None
        run.finished_at = None
        await db.flush()
        return RunClaimResult(
            disposition=RunClaimDisposition.STARTED,
            run=run,
        )

    # running — 检查 lease
    if run.status == "running":
        lease_valid = run.lease_expires_at is not None and run.lease_expires_at > now

        if lease_valid:
            # 同 task_id → ACTIVE_SAME_TASK
            if run.execution_task_id == celery_task_id:
                _lease_exp = run.lease_expires_at  # type: datetime, guaranteed by lease_valid
                remaining = math.ceil(
                    (_lease_exp - now).total_seconds()  # type: ignore[operator]
                ) + 1
                return RunClaimResult(
                    disposition=RunClaimDisposition.ACTIVE_SAME_TASK,
                    run=run,
                    retry_after_seconds=remaining,
                )
            # 不同 task_id → ACTIVE_OTHER_TASK
            return RunClaimResult(
                disposition=RunClaimDisposition.ACTIVE_OTHER_TASK,
                run=run,
            )

        # lease 过期 → RECLAIMED_STALE
        run.attempt += 1
        run.execution_task_id = celery_task_id
        run.execution_owner = execution_owner
        run.heartbeat_at = now
        run.lease_expires_at = now + timedelta(seconds=lease_seconds)
        run.error_type = None
        run.error_message = None
        run.finished_at = None
        await db.flush()
        return RunClaimResult(
            disposition=RunClaimDisposition.RECLAIMED_STALE,
            run=run,
        )

    raise InvalidRunTransition(
        f"Cannot claim run {run_id} in status '{run.status}'"
    )


async def renew_run_lease(
    db: AsyncSession,
    *,
    run_id: str,
    execution_owner: str,
    now: datetime,
    lease_seconds: int,
) -> bool:
    """续期 run 的执行租约

    仅当 execution_owner 匹配且 status=running 时续期成功。
    owner 不匹配抛 RunLeaseLost。
    """
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise RunLeaseLost(f"Run {run_id} not found")

    if run.execution_owner != execution_owner:
        raise RunLeaseLost(
            f"Run {run_id} owner mismatch: expected '{execution_owner}', "
            f"got '{run.execution_owner}'"
        )

    if run.status != "running":
        return False

    run.heartbeat_at = now
    run.lease_expires_at = now + timedelta(seconds=lease_seconds)
    await db.flush()
    return True


async def mark_run_retrying(
    db: AsyncSession,
    *,
    run_id: str,
    execution_owner: str,
    error: BaseException,
) -> None:
    """标记 run 为 retrying 状态（清除执行租约，不写 finished_at）"""
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise InvalidRunTransition(f"Run {run_id} not found")

    if run.execution_owner != execution_owner:
        raise RunLeaseLost(
            f"Run {run_id} owner mismatch: expected '{execution_owner}', "
            f"got '{run.execution_owner}'"
        )

    run.status = "retrying"
    run.execution_task_id = None
    run.execution_owner = None
    run.heartbeat_at = None
    run.lease_expires_at = None
    run.error_type = type(error).__name__
    run.error_message = f"{type(error).__name__}: {error}"[:500]
    # retrying 不写 finished_at
    await db.flush()

    # ── metrics ──
    error_code = type(error).__name__
    EVALUATION_RETRIES_TOTAL.labels(error_code=error_code).inc()
    EVALUATION_ACTIVE_RUNS.labels(status="running").dec()
    EVALUATION_ACTIVE_RUNS.labels(status="retrying").inc()


async def mark_run_terminal(
    db: AsyncSession,
    *,
    run_id: str,
    status: str,
    execution_owner: str,
    evaluation_id: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    now: datetime | None = None,
) -> None:
    """标记 run 为终态（completed/failed/cancelled/needs_review）

    终态幂等：若 run 已在终态，不重写 finished_at/error/evaluation_id。
    """
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise InvalidRunTransition(f"Run {run_id} not found")

    # 终态幂等
    if run.status in _TERMINAL_STATUSES:
        return

    if run.execution_owner != execution_owner:
        raise RunLeaseLost(
            f"Run {run_id} owner mismatch: expected '{execution_owner}', "
            f"got '{run.execution_owner}'"
        )

    _apply_terminal_state(
        run,
        status=status,
        evaluation_id=evaluation_id,
        error_code=error_code,
        error_message=error_message,
        now=now or datetime.utcnow(),
    )
    await db.flush()

    # ── metrics ──
    _ec = error_code or ""
    EVALUATION_RUNS_TOTAL.labels(status=status, error_code=_ec).inc()
    EVALUATION_ACTIVE_RUNS.labels(status="running").dec()
    if run.started_at is not None:
        _now = now or datetime.utcnow()
        dur = (_now - run.started_at).total_seconds()
        EVALUATION_RUN_DURATION.labels(status=status).observe(max(dur, 0))


async def mark_unowned_run_terminal(
    db: AsyncSession,
    *,
    run_id: str,
    expected_status: Literal["queued", "retrying"],
    status: Literal["failed", "cancelled"],
    error_code: str | None = None,
) -> bool:
    """在无 execution_owner 的情况下标记 run 为终态

    仅当 run 处于 expected_status 且无活跃租约时成功。
    用于 outbox dead-letter、排队期取消和 stale retrying 清理。
    """
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        return False

    if run.status != expected_status:
        return False

    # 确保无活跃租约
    if run.execution_owner is not None or run.lease_expires_at is not None:
        return False

    now = datetime.utcnow()
    run.status = status
    run.finished_at = now
    if error_code:
        run.error_type = error_code
    await db.flush()

    # ── metrics ──
    _ec = error_code or ""
    EVALUATION_RUNS_TOTAL.labels(status=status, error_code=_ec).inc()
    EVALUATION_ACTIVE_RUNS.labels(status=expected_status).dec()
    return True


async def request_run_cancel(
    db: AsyncSession,
    *,
    run_id: str,
    requested_by: int,
    now: datetime,
) -> tuple[CancelDisposition, EvaluationRun]:
    """请求取消 run

    - queued/retrying → 直接 cancelled
    - running → 只设置 cancel_requested_at/by（协作式停止）
    - 终态 → 幂等返回 ALREADY_TERMINAL
    - needs_review → NOT_CANCELLABLE
    """
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise InvalidRunTransition(f"Run {run_id} not found")

    # needs_review 不可取消（优先检查，因为它在终态集合中但语义不同）
    if run.status == "needs_review":
        return CancelDisposition.NOT_CANCELLABLE, run

    # 终态幂等
    if run.status in _TERMINAL_STATUSES:
        return CancelDisposition.ALREADY_TERMINAL, run

    # queued/retrying → 直接 cancelled
    if run.status in _CLAIMABLE_STATUSES:
        old_status = run.status
        run.status = "cancelled"
        run.cancel_requested_at = now
        run.cancel_requested_by = requested_by
        run.finished_at = now
        await db.flush()
        EVALUATION_CANCELLATIONS_TOTAL.labels(phase=old_status).inc()
        EVALUATION_RUNS_TOTAL.labels(status="cancelled", error_code="").inc()
        EVALUATION_ACTIVE_RUNS.labels(status=old_status).dec()
        return CancelDisposition.CANCELLED_BEFORE_START, run

    # running → 只设置取消标志
    if run.status == "running":
        run.cancel_requested_at = now
        run.cancel_requested_by = requested_by
        await db.flush()
        EVALUATION_CANCELLATIONS_TOTAL.labels(phase="running").inc()
        return CancelDisposition.REQUESTED_RUNNING, run

    raise InvalidRunTransition(
        f"Cannot cancel run {run_id} in status '{run.status}'"
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _apply_terminal_state(
    run: EvaluationRun,
    *,
    status: str,
    evaluation_id: int | None,
    error_code: str | None,
    error_message: str | None,
    now: datetime,
) -> None:
    """将 run 转为终态，清空执行租约"""
    run.status = status
    run.finished_at = now
    # 清空执行租约
    run.execution_task_id = None
    run.execution_owner = None
    run.heartbeat_at = None
    run.lease_expires_at = None
    # 设置终态相关字段
    if evaluation_id is not None:
        run.evaluation_id = evaluation_id
    if error_code is not None:
        run.error_type = error_code
    if error_message is not None:
        run.error_message = error_message[:500]
