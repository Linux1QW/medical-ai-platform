"""异步评估任务 — 在 Celery Worker 中执行"""

import asyncio
import logging

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="run_evaluation", max_retries=2)
def run_evaluation_task(self, consultation_id: int, run_id: str) -> dict:
    """异步评估任务

    在 Celery worker 进程中执行完整评估流程。
    使用 asyncio.run() 桥接同步 Celery worker 与异步评估逻辑。

    Args:
        consultation_id: 问诊记录 ID
        run_id: 评估锁 run_id（用于状态关联）

    Returns:
        dict: {"evaluation_id": ..., "status": ..., "consultation_id": ...}
    """
    logger.info(
        f"[Celery] 开始评估任务: consultation_id={consultation_id}, run_id={run_id}"
    )

    try:
        result = asyncio.run(_execute_evaluation(consultation_id, run_id))
        logger.info(
            f"[Celery] 评估完成: consultation_id={consultation_id}, "
            f"status={result.get('status')}"
        )
        return result

    except Exception as exc:
        logger.error(
            f"[Celery] 评估失败: consultation_id={consultation_id}, error={exc}"
        )
        # 更新锁状态为 failed
        try:
            asyncio.run(_mark_lock_failed(consultation_id, str(exc)))
        except Exception:
            logger.exception("更新锁失败状态时出错")

        # 可重试异常（网络/超时类）
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        raise


async def _execute_evaluation(consultation_id: int, run_id: str) -> dict:
    """在异步上下文中执行评估"""
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import update_lock_status
    from app.services.evaluation_service import run_evaluation

    async with AsyncSessionLocal() as db:
        try:
            await update_lock_status(db, consultation_id, "running")
            await db.commit()

            evaluation = await run_evaluation(db, consultation_id)

            final_status = evaluation.evaluation_status
            if final_status == "needs_review":
                await update_lock_status(db, consultation_id, "needs_review")
            else:
                await update_lock_status(db, consultation_id, "completed")
            await db.commit()

            return {
                "evaluation_id": evaluation.id,
                "status": final_status,
                "consultation_id": consultation_id,
            }

        except Exception as exc:
            await db.rollback()
            raise exc


async def _mark_lock_failed(consultation_id: int, error: str) -> None:
    """标记评估锁为失败状态"""
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import update_lock_status

    async with AsyncSessionLocal() as db:
        await update_lock_status(db, consultation_id, "failed", error_message=error[:500])
        await db.commit()
