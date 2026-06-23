"""LangGraph Checkpointer 工厂和生命周期管理 — Redis 版本"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_checkpointer = None


async def init_checkpointer(redis_url: str = None, ttl: int = None):
    """初始化 Redis Checkpointer（在 FastAPI lifespan 中调用）"""
    global _checkpointer

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    if redis_url is None:
        from app.core.config import settings
        redis_url = settings.REDIS_CHECKPOINT_URL
    if ttl is None:
        from app.core.config import settings
        ttl = settings.REDIS_CHECKPOINT_TTL

    _checkpointer = AsyncRedisSaver.from_conn_string(redis_url)
    # 创建 Redis 索引（幂等操作，首次连接时执行一次）
    await _checkpointer.setup()
    logger.info(f"LangGraph Redis Checkpointer 已初始化: {redis_url}")
    return _checkpointer


async def close_checkpointer():
    """关闭 Checkpointer（在 FastAPI shutdown 中调用）"""
    global _checkpointer
    if _checkpointer is not None:
        await _checkpointer.close()
        _checkpointer = None
        logger.info("LangGraph Redis Checkpointer 已关闭")


def get_checkpointer():
    """获取当前 Checkpointer 实例"""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer 未初始化，请先调用 init_checkpointer()")
    return _checkpointer
