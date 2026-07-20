"""JWT 黑名单服务 — 基于 Redis 的 Token 吊销机制

用户登出时，将 access_token 的 JTI 加入黑名单，
后续请求在 get_current_user 中检查 JTI 是否在黑名单中。
"""
import logging
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None

_BLACKLIST_PREFIX = "jwt_blacklist:"


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 客户端（复用 db=2，与 token_tracker 同实例）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if settings.TESTING:
        return None
    try:
        redis_url = settings.REDIS_CHECKPOINT_URL
        # 使用 db=2 避免与 checkpoint (db=1) 冲突
        if "/1" in redis_url:
            redis_url = redis_url.replace("/1", "/2")
        elif redis_url.endswith("redis://localhost:6379"):
            redis_url = redis_url + "/2"
        _redis_client = aioredis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=3
        )
        await _redis_client.ping()
        logger.info(f"JWT 黑名单 Redis 已连接: {redis_url}")
        return _redis_client
    except Exception as e:
        logger.warning(f"JWT 黑名单 Redis 连接失败，黑名单功能禁用: {e}")
        return None


async def blacklist_token(token: str) -> bool:
    """将 token 加入黑名单

    Args:
        token: access_token 字符串

    Returns:
        True 表示成功加入，False 表示失败（Redis 不可用时）
    """
    if not settings.JWT_TOKEN_BLACKLIST_ENABLED:
        return True

    from app.core.security import get_token_jti, get_token_remaining_ttl

    jti = get_token_jti(token)
    if jti is None:
        logger.warning("无法从 token 中提取 JTI，跳过黑名单")
        return False

    ttl = get_token_remaining_ttl(token)
    if ttl is None:
        ttl = 86400  # 默认 24 小时

    r = await _get_redis()
    if r is None:
        return False

    try:
        key = f"{_BLACKLIST_PREFIX}{jti}"
        await r.setex(key, ttl, "1")
        logger.info(f"Token JTI {jti} 已加入黑名单，TTL={ttl}s")
        return True
    except Exception as e:
        logger.error(f"将 token 加入黑名单失败: {e}")
        return False


async def is_token_blacklisted(token: str) -> bool:
    """检查 token 是否在黑名单中

    Args:
        token: access_token 字符串

    Returns:
        True 表示 token 已吊销，False 表示正常
    """
    if not settings.JWT_TOKEN_BLACKLIST_ENABLED:
        return False

    from app.core.security import get_token_jti

    jti = get_token_jti(token)
    if jti is None:
        return False

    r = await _get_redis()
    if r is None:
        return False

    try:
        key = f"{_BLACKLIST_PREFIX}{jti}"
        return bool(await r.exists(key))
    except Exception as e:
        logger.error(f"检查 token 黑名单失败: {e}")
        return False


async def close_blacklist_redis() -> None:
    """关闭黑名单 Redis 连接（lifespan 关闭时调用）"""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
