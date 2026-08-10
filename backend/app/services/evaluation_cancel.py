# -*- coding: utf-8 -*-
"""评估任务取消服务 — Redis 取消标志 + Celery task_id 映射（Harness）

取消采用协作式双通道：
- 排队未执行：按 run_id 存储的 Celery task_id 执行 revoke，任务不再启动
- 执行中：Redis 取消标志（独立 Redis db=8）由
  evaluation_service 的取消看守轮询，命中后 cancel 图执行任务，
  抛 EvaluationCancelled 走既有失败路径（error_type="cancelled"，不重试）

标志与映射均带 TTL，且新评估提交时主动清除残留标志，
避免陈旧取消标志误杀后续 run。读取路径 best-effort：Redis 异常时
评估照常执行（无法取消但不阻断主流程）；写入路径（用户请求取消）
失败则向上抛出，由 API 返回明确错误。

Task 2: 所有 key 改为 run-scoped，不再使用 consultation_id。
"""

import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

CANCEL_FLAG_PREFIX = "evaluation:cancel"
TASK_ID_PREFIX = "evaluation:task"
CANCEL_FLAG_TTL = 600  # 秒，覆盖评估 deadline（240s）+ Celery 重试窗口

# 独立控制 Redis 连接（db=8）
_control_redis: Optional[aioredis.Redis] = None


def _cancel_key(run_id: str) -> str:
    return f"{CANCEL_FLAG_PREFIX}:{run_id}"


def _task_key(run_id: str) -> str:
    return f"{TASK_ID_PREFIX}:{run_id}"


async def _get_control_redis() -> Optional[aioredis.Redis]:
    """获取控制 Redis 客户端（lazy init）"""
    global _control_redis
    if _control_redis is not None:
        return _control_redis

    try:
        from app.core.config import settings
        url = getattr(settings, "EVALUATION_CONTROL_REDIS_URL", "redis://localhost:6379/8")
        _control_redis = aioredis.from_url(url, decode_responses=True)
        await _control_redis.ping()
        return _control_redis
    except Exception as e:
        logger.debug(f"控制 Redis 连接失败: {e}")
        return None


async def close_control_redis() -> None:
    """幂等关闭控制 Redis 连接"""
    global _control_redis
    if _control_redis is not None:
        try:
            await _control_redis.close()
        except Exception:
            pass
        _control_redis = None


# ── 向后兼容：保留 consultation_id 接口（内部转为 run_id key）──────────────
# 旧接口使用 consultation_id，新接口使用 run_id。
# 为保持向后兼容，旧接口将 consultation_id 转为 str 作为 key。


async def request_cancel(consultation_id: int) -> None:
    """置取消标志（写路径：失败向上抛出，API 层转成明确错误响应）

    注意：此函数保留 consultation_id 参数以兼容旧 API，
    新代码应使用 publish_cancel_nudge(run_id)。
    """
    await publish_cancel_nudge(str(consultation_id))


async def is_cancel_requested(consultation_id: int) -> bool:
    """查询取消标志（读路径 best-effort：Redis 异常视为未请求取消）

    注意：此函数保留 consultation_id 参数以兼容旧代码，
    新代码应使用 run_id 版本。
    """
    return await _is_cancel_requested_for_key(str(consultation_id))


async def clear_cancel_flag(consultation_id: int) -> None:
    """清除取消标志（新评估提交前调用，防止陈旧标志误杀；best-effort）"""
    await _clear_cancel_for_key(str(consultation_id))


async def store_task_id(consultation_id: int, task_id: str) -> None:
    """记录评估的 Celery task_id（供取消时 revoke 排队任务；best-effort）"""
    await _store_task_id_for_key(str(consultation_id), task_id)


async def get_task_id(consultation_id: int) -> str | None:
    """查询评估的 Celery task_id（best-effort）"""
    return await _get_task_id_for_key(str(consultation_id))


# ── 新 run-scoped 接口 ─────────────────────────────────────────────────────


async def publish_cancel_nudge(run_id: str) -> None:
    """发布取消标志（按 run_id）"""
    try:
        r = await _get_control_redis()
        if r is None:
            # 回退到 llm_cache Redis
            from app.services.llm_cache import _get_redis
            r = await _get_redis()
        if r is None:
            raise RuntimeError("Redis 不可用，无法提交取消请求")
        await r.set(_cancel_key(run_id), "1", ex=CANCEL_FLAG_TTL)
    except Exception as e:
        if "Redis 不可用" in str(e):
            raise
        logger.debug(f"publish_cancel_nudge 异常: {e}")
        raise RuntimeError(f"取消请求失败: {e}") from e


async def _is_cancel_requested_for_key(key: str) -> bool:
    """查询取消标志（best-effort）"""
    try:
        r = await _get_control_redis()
        if r is None:
            from app.services.llm_cache import _get_redis
            r = await _get_redis()
        if r is None:
            return False
        return await r.get(_cancel_key(key)) is not None
    except Exception as e:
        logger.debug(f"取消标志查询异常: {e}")
        return False


async def _clear_cancel_for_key(key: str) -> None:
    """清除取消标志（best-effort）"""
    try:
        r = await _get_control_redis()
        if r is None:
            from app.services.llm_cache import _get_redis
            r = await _get_redis()
        if r is not None:
            await r.delete(_cancel_key(key))
    except Exception as e:
        logger.debug(f"取消标志清除异常: {e}")


async def _store_task_id_for_key(key: str, task_id: str) -> None:
    """记录 Celery task_id（best-effort）"""
    try:
        r = await _get_control_redis()
        if r is None:
            from app.services.llm_cache import _get_redis
            r = await _get_redis()
        if r is not None:
            await r.set(_task_key(key), task_id, ex=CANCEL_FLAG_TTL)
    except Exception as e:
        logger.debug(f"task_id 记录异常: {e}")


async def _get_task_id_for_key(key: str) -> str | None:
    """查询 Celery task_id（best-effort）"""
    try:
        r = await _get_control_redis()
        if r is None:
            from app.services.llm_cache import _get_redis
            r = await _get_redis()
        if r is None:
            return None
        return await r.get(_task_key(key))
    except Exception as e:
        logger.debug(f"task_id 查询异常: {e}")
        return None
