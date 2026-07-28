# -*- coding: utf-8 -*-
"""评估任务取消服务 — Redis 取消标志 + Celery task_id 映射（Harness）

取消采用协作式双通道：
- 排队未执行：按 consultation_id 存储的 Celery task_id 执行 revoke，任务不再启动
- 执行中：Redis 取消标志（db=2，复用 llm_cache 客户端）由
  evaluation_service 的取消看守轮询，命中后 cancel 图执行任务，
  抛 EvaluationCancelled 走既有失败路径（error_type="cancelled"，不重试）

标志与映射均带 TTL，且新评估提交时主动清除残留标志，
避免陈旧取消标志误杀后续 run。读取路径 best-effort：Redis 不可用时
评估照常执行（无法取消但不阻断主流程）；写入路径（用户请求取消）
失败则向上抛出，由 API 返回明确错误。
"""

import logging

from app.services.llm_cache import _get_redis

logger = logging.getLogger(__name__)

CANCEL_FLAG_PREFIX = "eval_cancel"
TASK_ID_PREFIX = "eval_task_id"
CANCEL_FLAG_TTL = 600  # 秒，覆盖评估 deadline（240s）+ Celery 重试窗口


def _flag_key(consultation_id: int) -> str:
    return f"{CANCEL_FLAG_PREFIX}:{consultation_id}"


def _task_key(consultation_id: int) -> str:
    return f"{TASK_ID_PREFIX}:{consultation_id}"


async def request_cancel(consultation_id: int) -> None:
    """置取消标志（写路径：失败向上抛出，API 层转成明确错误响应）"""
    redis = await _get_redis()
    if redis is None:
        raise RuntimeError("Redis 不可用，无法提交取消请求")
    await redis.set(_flag_key(consultation_id), "1", ex=CANCEL_FLAG_TTL)


async def is_cancel_requested(consultation_id: int) -> bool:
    """查询取消标志（读路径 best-effort：Redis 异常视为未请求取消）"""
    try:
        redis = await _get_redis()
        if redis is None:
            return False
        return await redis.get(_flag_key(consultation_id)) is not None
    except Exception as e:
        logger.debug(f"取消标志查询异常: {e}")
        return False


async def clear_cancel_flag(consultation_id: int) -> None:
    """清除取消标志（新评估提交前调用，防止陈旧标志误杀；best-effort）"""
    try:
        redis = await _get_redis()
        if redis is not None:
            await redis.delete(_flag_key(consultation_id))
    except Exception as e:
        logger.debug(f"取消标志清除异常: {e}")


async def store_task_id(consultation_id: int, task_id: str) -> None:
    """记录评估的 Celery task_id（供取消时 revoke 排队任务；best-effort）"""
    try:
        redis = await _get_redis()
        if redis is not None:
            await redis.set(_task_key(consultation_id), task_id, ex=CANCEL_FLAG_TTL)
    except Exception as e:
        logger.debug(f"task_id 记录异常: {e}")


async def get_task_id(consultation_id: int) -> str | None:
    """查询评估的 Celery task_id（best-effort）"""
    try:
        redis = await _get_redis()
        if redis is None:
            return None
        return await redis.get(_task_key(consultation_id))
    except Exception as e:
        logger.debug(f"task_id 查询异常: {e}")
        return None
