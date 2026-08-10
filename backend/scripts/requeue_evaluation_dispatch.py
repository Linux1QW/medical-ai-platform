# -*- coding: utf-8 -*-
"""恢复 dead_letter dispatch — CLI 工具

用法:
    python scripts/requeue_evaluation_dispatch.py \
        --run-id <uuid> \
        --operator-id <int> \
        --reason-code <broker_recovered|configuration_fixed|state_recovered>

前提条件:
    - run 状态为 failed
    - run.error_type 为 dispatch_exhausted 或 dispatch_unclaimed
    - 无 Evaluation 关联
    - 无取消意图（cancel_requested_at IS NULL）
    - operator 为 active admin
    - 一个事务中恢复 run → queued, outbox → pending
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# 允许的 reason codes
_VALID_REASON_CODES = frozenset({
    "broker_recovered",
    "configuration_fixed",
    "state_recovered",
})

# 允许的 dispatch 错误类型
_VALID_DISPATCH_ERRORS = frozenset({
    "dispatch_exhausted",
    "dispatch_unclaimed",
})


async def requeue_dispatch(
    db: AsyncSession,
    *,
    run_id: str,
    operator_id: int,
    reason_code: str,
) -> bool:
    """恢复 dead_letter dispatch

    在一个事务中:
    1. 锁定 run 行
    2. 校验前提条件
    3. 恢复 run → queued, 清除 error/finished_at
    4. 恢复 outbox → pending, attempts=0, next_attempt_at=now
    5. 写 strict audit

    Returns:
        True 成功，False 前提不满足
    """
    from app.core.audit import record_audit_log
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.models.evaluation_run import EvaluationRun
    from app.models.user import User

    now = datetime.utcnow()

    # 1. 校验 operator 为 active admin
    user_stmt = select(User).where(User.id == operator_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    if user is None or user.role != "admin":
        print(f"Error: operator {operator_id} is not an active admin", file=sys.stderr)
        return False

    # 2. 锁定 run 行
    run_stmt = select(EvaluationRun).where(EvaluationRun.id == run_id).with_for_update()
    run_result = await db.execute(run_stmt)
    run = run_result.scalar_one_or_none()
    if run is None:
        print(f"Error: run {run_id} not found", file=sys.stderr)
        return False

    # 3. 校验 run 状态
    if run.status != "failed":
        print(f"Error: run {run_id} is not failed (status={run.status})", file=sys.stderr)
        return False

    if run.error_type not in _VALID_DISPATCH_ERRORS:
        print(
            f"Error: run error_type '{run.error_type}' is not a dispatch error. "
            f"Allowed: {_VALID_DISPATCH_ERRORS}",
            file=sys.stderr,
        )
        return False

    # 4. 校验无 Evaluation 关联
    if run.evaluation_id is not None:
        print(f"Error: run {run_id} already has Evaluation #{run.evaluation_id}", file=sys.stderr)
        return False

    # 5. 校验无取消意图
    if run.cancel_requested_at is not None:
        print(f"Error: run {run_id} has cancel_requested_at set", file=sys.stderr)
        return False

    # 6. 恢复 run → queued
    run.status = "queued"
    run.error_type = None
    run.error_message = None
    run.finished_at = None
    run.attempt = 0
    await db.flush()

    # 7. 恢复 outbox → pending
    outbox_stmt = (
        update(EvaluationDispatchOutbox)
        .where(EvaluationDispatchOutbox.run_id == run_id)
        .values(
            status="pending",
            attempt=0,
            next_attempt_at=now,
            last_error_code=None,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=now,
        )
    )
    await db.execute(outbox_stmt)
    await db.flush()

    # 8. 写 strict audit
    await record_audit_log(
        db,
        user_id=operator_id,
        action="admin_action",
        resource_id=run_id,
        detail=f"dispatch_recovered reason={reason_code}",
        strict=True,
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="恢复 dead_letter dispatch — 将 failed run 恢复为 queued"
    )
    parser.add_argument("--run-id", required=True, help="评估运行 UUID")
    parser.add_argument("--operator-id", required=True, type=int, help="操作管理员 ID")
    parser.add_argument(
        "--reason-code",
        required=True,
        choices=sorted(_VALID_REASON_CODES),
        help="恢复原因",
    )
    args = parser.parse_args()

    async def _run():
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            try:
                success = await requeue_dispatch(
                    db,
                    run_id=args.run_id,
                    operator_id=args.operator_id,
                    reason_code=args.reason_code,
                )
                await db.commit()
                if success:
                    print(f"OK: run {args.run_id} recovered to queued")
                    return 0
                else:
                    await db.rollback()
                    return 1
            except Exception as e:
                await db.rollback()
                print(f"Error: {e}", file=sys.stderr)
                return 1

    exit_code = asyncio.run(_run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
