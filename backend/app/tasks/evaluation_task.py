"""异步评估任务 — 在 Celery Worker 中执行"""

import asyncio
import logging
from contextlib import suppress

from app.celery_app import celery_app

logger = logging.getLogger(__name__)

RETRY_BACKOFF_BASE = 30  # 重试退避基础秒数（30s → 60s 指数增长）

# 可重试的异常类型（网络/超时类）；业务性错误（数据缺失、校验失败）重试无意义
_RETRYABLE_EXC_TYPES = (TimeoutError, ConnectionError, OSError)
_RETRYABLE_NAME_KEYWORDS = ("timeout", "connection", "network", "unavailable", "ratelimit", "temporar")


def _is_retryable_error(exc: BaseException) -> bool:
    """判断异常是否值得重试：类型匹配 或 异常类名含网络/超时类关键词"""
    if isinstance(exc, _RETRYABLE_EXC_TYPES):
        return True
    name = type(exc).__name__.lower()
    return any(keyword in name for keyword in _RETRYABLE_NAME_KEYWORDS)


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
        # 重试执行时尝试从上次失败 run 的 LangGraph checkpoint 断点续跑
        resume = self.request.retries > 0
        result = asyncio.run(_execute_evaluation(consultation_id, run_id, resume=resume))
        logger.info(
            f"[Celery] 评估完成: consultation_id={consultation_id}, "
            f"status={result.get('status')}"
        )
        return result

    except Exception as exc:
        logger.error(
            f"[Celery] 评估失败: consultation_id={consultation_id}, error={exc}"
        )
        # 更新锁状态为 failed（running → failed 合法转移）
        try:
            asyncio.run(_mark_lock_failed(consultation_id, str(exc)))
        except Exception:
            logger.exception("更新锁失败状态时出错")

        # 仅网络/超时类异常重试；重试前将锁重置为 pending（failed → pending 合法转移），
        # 使重试执行时的 pending → running 符合状态机，避免锁停留在 failed 与实际运行状态打架
        will_retry = (
            self.request.retries < self.max_retries and _is_retryable_error(exc)
        )
        if will_retry:
            try:
                asyncio.run(_mark_lock_pending_for_retry(consultation_id))
            except Exception:
                logger.exception("重试前重置锁状态时出错")
            countdown = RETRY_BACKOFF_BASE * (2 ** self.request.retries)
            logger.info(
                f"[Celery] 将在 {countdown}s 后重试: consultation_id={consultation_id}, "
                f"retries={self.request.retries + 1}/{self.max_retries}"
            )
            raise self.retry(exc=exc, countdown=countdown) from exc
        raise


async def _ensure_worker_checkpointer() -> None:
    """Celery worker 侧按任务重建 LangGraph Checkpointer

    checkpointer 原本只在 FastAPI lifespan 中初始化，worker 进程内为 None，
    图会以“无 checkpointer”方式编译，断点续跑无从谈起。
    又因 asyncio.run 每个任务新建事件循环，异步 Redis 连接跨循环不可复用，
    故每次任务先关旧建新并重置图编译缓存（索引创建幂等，开销相对评估时长可忽略）。
    """
    from app.core.config import settings

    if not settings.LANGGRAPH_ENABLED:
        return

    from app.orchestration.checkpointer import close_checkpointer, init_checkpointer
    from app.orchestration.graph import close_graph

    await close_checkpointer()
    await close_graph()
    await init_checkpointer(
        redis_url=settings.REDIS_CHECKPOINT_URL,
        ttl=settings.REDIS_CHECKPOINT_TTL,
    )


async def _execute_evaluation(
    consultation_id: int, run_id: str, resume: bool = False
) -> dict:
    """在异步上下文中执行评估（附带锁心跳续期后台任务）"""
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import update_lock_status
    from app.services.evaluation_service import run_evaluation

    await _ensure_worker_checkpointer()

    async with AsyncSessionLocal() as db:
        heartbeat_task: asyncio.Task | None = None
        try:
            await update_lock_status(db, consultation_id, "running")
            await db.commit()

            # 启动心跳续期：防止长评估期间锁 TTL 过期被并发请求清理
            heartbeat_task = asyncio.create_task(
                _heartbeat_loop(consultation_id, run_id)
            )

            evaluation = await run_evaluation(db, consultation_id, resume=resume)

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

        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task


async def _heartbeat_loop(consultation_id: int, run_id: str) -> None:
    """锁心跳续期循环：每 HEARTBEAT_INTERVAL 秒刷新 heartbeat_at 并延长 TTL

    使用独立 DB 会话，避免与评估主流程的事务互相干扰。
    续期失败（锁丢失/已进入终态）时停止心跳，续期异常则下轮重试。
    """
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import HEARTBEAT_INTERVAL, renew_lock

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            async with AsyncSessionLocal() as db:
                renewed = await renew_lock(db, consultation_id, run_id)
                await db.commit()
            if not renewed:
                logger.warning(
                    f"[Heartbeat] 锁续期失败（锁丢失或非 running），停止心跳: "
                    f"consultation_id={consultation_id}"
                )
                return
            logger.debug(
                f"[Heartbeat] 锁续期成功: consultation_id={consultation_id}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Heartbeat] 续期异常（下轮重试）: {e}")


async def _mark_lock_failed(consultation_id: int, error: str) -> None:
    """标记评估锁为失败状态"""
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import update_lock_status

    async with AsyncSessionLocal() as db:
        await update_lock_status(db, consultation_id, "failed", error_message=error[:500])
        await db.commit()


async def _mark_lock_pending_for_retry(consultation_id: int) -> None:
    """重试前将锁重置为 pending（failed → pending 合法转移，并恢复短 TTL）"""
    from app.db.session import AsyncSessionLocal
    from app.services.evaluation_lock_service import update_lock_status

    async with AsyncSessionLocal() as db:
        await update_lock_status(db, consultation_id, "pending")
        await db.commit()
