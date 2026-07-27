"""LangGraph Checkpointer 工厂和生命周期管理 — Redis 版本"""

import logging
from contextlib import AsyncExitStack
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_checkpointer = None
_exit_stack: Optional[AsyncExitStack] = None


async def init_checkpointer(redis_url: str = None, ttl: int = None):
    """初始化 Redis Checkpointer（在 FastAPI lifespan 中调用）

    Returns:
        checkpointer 实例，如果 LANGGRAPH_ENABLED=false 则返回 None

    Raises:
        RuntimeError: LANGGRAPH_ENABLED=true 但 Redis 初始化失败
    """
    global _checkpointer, _exit_stack

    # LANGGRAPH_ENABLED=false 时跳过 checkpointer 初始化
    if not settings.LANGGRAPH_ENABLED:
        logger.info("LangGraph disabled, skipping checkpointer initialization")
        return None

    if redis_url is None:
        redis_url = settings.REDIS_CHECKPOINT_URL
    if ttl is None:
        ttl = settings.REDIS_CHECKPOINT_TTL

    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    except ImportError as e:
        # LANGGRAPH_ENABLED=true 但依赖未安装，启动失败
        raise RuntimeError(
            f"langgraph-checkpoint-redis 未安装，但 LANGGRAPH_ENABLED=true。"
            f"请安装依赖或设置 LANGGRAPH_ENABLED=false。错误: {e}"
        ) from e

    try:
        # from_conn_string 是 async context manager，长驻服务用 AsyncExitStack 持有；
        # 进入上下文时内部已执行 asetup() 创建 Redis 索引（幂等操作）
        # REDIS_CHECKPOINT_TTL 单位秒，AsyncRedisSaver 的 ttl 配置单位分钟
        ttl_config = {"default_ttl": max(1, ttl // 60)} if ttl and ttl > 0 else None
        _exit_stack = AsyncExitStack()
        _checkpointer = await _exit_stack.enter_async_context(
            AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_config)
        )
        logger.info(f"LangGraph Redis Checkpointer 已初始化: {redis_url} (ttl={ttl_config})")
        return _checkpointer
    except Exception as e:
        _exit_stack = None
        _checkpointer = None
        # Redis 连接失败，LANGGRAPH_ENABLED=true 时不允许降级
        raise RuntimeError(
            f"Redis Checkpointer 初始化失败: {e}。"
            f"请确保 Redis 服务正在运行，或设置 LANGGRAPH_ENABLED=false。"
        ) from e


async def close_checkpointer():
    """关闭 Checkpointer（在 FastAPI shutdown 或 Celery worker 任务重建前调用）

    关闭异常仅告警不上抛：Celery 任务间事件循环已切换时，
    旧连接绑定的循环已关闭，aclose 可能报错，但全局引用必须被清理。
    """
    global _checkpointer, _exit_stack
    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
        except Exception as e:
            logger.warning(f"关闭 Checkpointer 时出错（已忽略，可能因事件循环已切换）: {e}")
        _exit_stack = None
        _checkpointer = None
        logger.info("LangGraph Redis Checkpointer 已关闭")


def get_checkpointer():
    """获取当前 Checkpointer 实例

    Returns:
        checkpointer 实例，如果未初始化则返回 None
    """
    return _checkpointer
