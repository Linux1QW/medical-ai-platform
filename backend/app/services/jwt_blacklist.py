"""JWT 黑名单服务 — 基于 Redis 的 Token 吊销机制

用户登出时，将 access_token 的 JTI 加入黑名单，
后续请求在 authenticate_access_token 中检查 JTI 是否在黑名单中。

生产环境（fail-closed）：Redis 不可用时抛 TokenRevocationStoreUnavailable，拒绝放行。
开发环境（fail-open）：Redis 不可用时返回 False（未吊销）+ 结构化 warning。
"""
import logging
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None

_BLACKLIST_PREFIX = "jwt_blacklist:"


class TokenRevocationStoreUnavailable(RuntimeError):
    """吊销存储（Redis）不可用，fail-closed 模式下抛出"""
    pass


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 客户端（使用独立 JWT_BLACKLIST_REDIS_URL，db=7）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if settings.TESTING:
        return None
    try:
        redis_url = settings.JWT_BLACKLIST_REDIS_URL
        _redis_client = aioredis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=3
        )
        await _redis_client.ping()
        logger.info(f"JWT 黑名单 Redis 已连接: {redis_url}")
        return _redis_client
    except Exception as e:
        logger.warning(f"JWT 黑名单 Redis 连接失败: {e}")
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

    fail-closed 模式（staging/production）：Redis 不可用或读取异常时
    抛 TokenRevocationStoreUnavailable，由上层转为 503。

    fail-open 模式（development/test）：Redis 不可用时返回 False + warning。

    Args:
        token: access_token 字符串

    Returns:
        True 表示 token 已吊销，False 表示正常

    Raises:
        TokenRevocationStoreUnavailable: fail-closed 模式下 Redis 不可用
    """
    if not settings.JWT_TOKEN_BLACKLIST_ENABLED:
        return False

    from app.core.security import get_token_jti

    jti = get_token_jti(token)
    if jti is None:
        # 没有 jti 的 token 在 authenticate_access_token 层处理
        return False

    r = await _get_redis()
    if r is None:
        if settings.JWT_BLACKLIST_FAIL_CLOSED:
            raise TokenRevocationStoreUnavailable(
                "JWT 吊销存储（Redis）连接不可用，拒绝放行"
            )
        logger.warning(
            "[fail-open] JWT 黑名单 Redis 连接不可用，token 视为未吊销"
        )
        return False

    try:
        key = f"{_BLACKLIST_PREFIX}{jti}"
        return bool(await r.exists(key))
    except Exception as e:
        logger.error(f"检查 token 黑名单失败: {e}")
        if settings.JWT_BLACKLIST_FAIL_CLOSED:
            raise TokenRevocationStoreUnavailable(
                f"JWT 吊销存储（Redis）读取异常: {e}"
            ) from e
        logger.warning("[fail-open] Redis 读取异常，token 视为未吊销")
        return False


async def close_blacklist_redis() -> None:
    """关闭黑名单 Redis 连接（lifespan 关闭时调用）"""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
