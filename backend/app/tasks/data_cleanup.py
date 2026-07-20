"""数据留存策略 — 定时清理过期数据"""

import asyncio
import logging
from datetime import datetime, timedelta

from app.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup_expired_records")
def cleanup_expired_records() -> dict:
    """定时清理过期数据

    - 审计日志保留 AUDIT_LOG_RETENTION_DAYS 天（默认 90）
    - 评估运行记录保留 EVALUATION_RUN_RETENTION_DAYS 天（默认 180）
    """
    logger.info("[Cleanup] 开始清理过期数据")
    try:
        result = asyncio.run(_do_cleanup())
        logger.info(f"[Cleanup] 清理完成: {result}")
        return result
    except Exception as exc:
        logger.error(f"[Cleanup] 清理失败: {exc}")
        raise


async def _do_cleanup() -> dict:
    """执行清理逻辑"""
    from app.db.session import AsyncSessionLocal
    from app.models.audit_log import AuditLog
    from app.models.evaluation_run import EvaluationRun
    from sqlalchemy import delete

    async with AsyncSessionLocal() as db:
        now = datetime.utcnow()

        # 清理过期审计日志
        audit_cutoff = now - timedelta(days=settings.AUDIT_LOG_RETENTION_DAYS)
        audit_result = await db.execute(
            delete(AuditLog).where(AuditLog.created_at < audit_cutoff)
        )
        audit_deleted = audit_result.rowcount

        # 清理过期评估运行记录
        run_cutoff = now - timedelta(days=settings.EVALUATION_RUN_RETENTION_DAYS)
        run_result = await db.execute(
            delete(EvaluationRun).where(EvaluationRun.created_at < run_cutoff)
        )
        run_deleted = run_result.rowcount

        await db.commit()

        return {
            "audit_logs_deleted": audit_deleted,
            "evaluation_runs_deleted": run_deleted,
            "audit_cutoff": audit_cutoff.isoformat(),
            "run_cutoff": run_cutoff.isoformat(),
        }
