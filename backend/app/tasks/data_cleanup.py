"""数据留存策略 — 定时清理过期数据

- Outbox: 7 天前 published/cancelled/dead_letter 清理
- Run: 180 天前无 Evaluation 的 failed/cancelled 清理
- Audit: 仅在 auto_delete=true 且 policy_id 非空时清理
"""

import logging
from datetime import datetime, timedelta

from app.celery_app import celery_app
from app.core.config import settings
from app.tasks.async_runtime import run_worker_coroutine

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup_expired_records")
def cleanup_expired_records() -> dict:
    """定时清理过期数据

    - 审计日志保留 AUDIT_LOG_RETENTION_DAYS 天（默认 90），仅 auto_delete=true 时
    - Outbox 终态行保留 DISPATCH_RETENTION_DAYS 天（默认 7）
    - 评估运行记录保留 UNREPORTED_RUN_RETENTION_DAYS 天（默认 180），
      且只处理无 Evaluation 关联的 failed/cancelled
    """
    logger.info("[Cleanup] 开始清理过期数据")
    try:
        result = run_worker_coroutine(_do_cleanup())
        logger.info(f"[Cleanup] 清理完成: {result}")
        return result
    except Exception as exc:
        logger.error(f"[Cleanup] 清理失败: {exc}")
        raise


async def _do_cleanup() -> dict:
    """执行清理逻辑"""
    from sqlalchemy import delete, select

    from app.db.session import AsyncSessionLocal
    from app.models.audit_log import AuditLog
    from app.models.evaluation_dispatch_outbox import EvaluationDispatchOutbox
    from app.models.evaluation_node_result import EvaluationNodeResult
    from app.models.evaluation_run import EvaluationRun

    async with AsyncSessionLocal() as db:
        now = datetime.utcnow()

        # 1. 清理过期审计日志（仅在 auto_delete=true 时）
        audit_result = await maybe_cleanup_audit_logs(
            db,
            now=now,
            auto_delete_enabled=settings.AUDIT_LOG_AUTO_DELETE_ENABLED,
            policy_id=settings.DATA_RETENTION_POLICY_ID,
            retention_days=settings.AUDIT_LOG_RETENTION_DAYS,
        )
        audit_deleted = audit_result["deleted"]

        # 2. 清理终态 outbox 行（7 天前 published/cancelled/dead_letter）
        outbox_cutoff = now - timedelta(days=settings.DISPATCH_RETENTION_DAYS)
        from app.services.evaluation_dispatch_service import purge_terminal_dispatches
        outbox_deleted = await purge_terminal_dispatches(db, older_than=outbox_cutoff)

        # 3. 清理无 Evaluation 的 failed/cancelled run（180 天前）
        run_deleted = await cleanup_unreported_runs(
            db,
            now=now,
            retention_days=settings.UNREPORTED_RUN_RETENTION_DAYS,
        )

        await db.commit()

        return {
            "audit_logs_deleted": audit_deleted,
            "outbox_deleted": outbox_deleted,
            "evaluation_runs_deleted": run_deleted,
            "audit_cutoff": (now - timedelta(days=settings.AUDIT_LOG_RETENTION_DAYS)).isoformat(),
            "outbox_cutoff": outbox_cutoff.isoformat(),
            "run_cutoff": (now - timedelta(days=settings.UNREPORTED_RUN_RETENTION_DAYS)).isoformat(),
        }


async def maybe_cleanup_audit_logs(
    db,
    *,
    now: datetime,
    auto_delete_enabled: bool,
    policy_id: str,
    retention_days: int,
) -> dict:
    """清理审计日志 — 仅在 auto_delete=true 且 policy_id 非空时执行

    返回 {"deleted": int, "skipped": bool}
    """
    if not auto_delete_enabled or not policy_id:
        return {"deleted": 0, "skipped": True}

    if retention_days < 90:
        logger.warning(
            f"[Cleanup] Audit retention {retention_days}d < 90d minimum, skipping"
        )
        return {"deleted": 0, "skipped": True}

    from sqlalchemy import delete
    from app.models.audit_log import AuditLog

    cutoff = now - timedelta(days=retention_days)
    result = await db.execute(
        delete(AuditLog).where(AuditLog.created_at < cutoff)
    )
    return {"deleted": result.rowcount, "skipped": False}


async def cleanup_unreported_runs(
    db,
    *,
    now: datetime,
    retention_days: int,
) -> int:
    """清理无 Evaluation 关联的 failed/cancelled run

    只处理：
    - status IN ('failed', 'cancelled')
    - created_at < (now - retention_days)
    - evaluation_id IS NULL（无 Evaluation 关联）

    返回删除行数。
    """
    from sqlalchemy import and_, delete, select

    from app.models.evaluation_node_result import EvaluationNodeResult
    from app.models.evaluation_run import EvaluationRun

    cutoff = now - timedelta(days=retention_days)

    # 查找符合条件的 run IDs
    stmt = select(EvaluationRun.id).where(
        and_(
            EvaluationRun.status.in_(("failed", "cancelled")),
            EvaluationRun.created_at < cutoff,
            EvaluationRun.evaluation_id.is_(None),
        )
    )
    result = await db.execute(stmt)
    run_ids = [row[0] for row in result.all()]

    if not run_ids:
        return 0

    # 先删除关联的 node results
    await db.execute(
        delete(EvaluationNodeResult).where(
            EvaluationNodeResult.run_id.in_(run_ids)
        )
    )

    # 删除 runs（outbox 通过 FK CASCADE 自动删除）
    run_result = await db.execute(
        delete(EvaluationRun).where(EvaluationRun.id.in_(run_ids))
    )
    return run_result.rowcount
