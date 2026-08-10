"""统一认证入口

提供 authenticate_access_token()，供 HTTP dependency（get_current_user）
和 WebSocket 路由共用，避免认证逻辑分散。

校验顺序（不可改变）：
1. 一次 decode/签名与过期校验
2. blacklist 启用时要求 jti 存在（无 jti → AUTH_INVALID_TOKEN）
3. 查询吊销状态（is_token_blacklisted）
4. 解析 sub
5. 查用户（get_user_by_id）
"""
import logging
from typing import Optional

from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.services.jwt_blacklist import TokenRevocationStoreUnavailable, is_token_blacklisted
from app.services.user_service import get_user_by_id

# Re-export for convenience (tests and other modules can import from here)
__all__ = [
    "AuthenticationError",
    "TokenRevocationStoreUnavailable",
    "authenticate_access_token",
]

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """统一认证异常，携带 HTTP 状态码与错误码"""

    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def _decode_access_token(token: str) -> Optional[dict]:
    """解码并验证 JWT access token（签名 + 过期），返回 payload 或 None"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


async def authenticate_access_token(db: AsyncSession, token: str) -> User:
    """统一认证流程

    顺序固定：
    1. 一次 decode/签名与过期校验
    2. blacklist 启用时要求 jti 存在
    3. 查询吊销状态
    4. 解析 sub
    5. 查用户

    Args:
        db: 数据库 session
        token: JWT access token 字符串

    Returns:
        User 对象

    Raises:
        AuthenticationError: 认证失败（401/503）
    """
    # 1. 一次 decode/签名与过期校验
    payload = _decode_access_token(token)
    if payload is None:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_INVALID_TOKEN",
            message="无效的认证凭据",
        )

    # 2. blacklist 启用时要求 jti 存在
    jti = payload.get("jti")
    if settings.JWT_TOKEN_BLACKLIST_ENABLED and not jti:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_INVALID_TOKEN",
            message="无效的认证凭据：缺少 token 标识",
        )

    # 3. 查询吊销状态
    try:
        blacklisted = await is_token_blacklisted(token)
    except TokenRevocationStoreUnavailable as e:
        raise AuthenticationError(
            status_code=503,
            error_code="AUTH_REVOCATION_UNAVAILABLE",
            message="吊销服务暂不可用，请稍后重试",
        ) from e

    if blacklisted:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_TOKEN_REVOKED",
            message="凭据已失效，请重新登录",
        )

    # 4. 解析 sub
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_INVALID_TOKEN",
            message="无效的认证凭据",
        )
    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError) as e:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_INVALID_TOKEN",
            message="无效的认证凭据",
        ) from e

    # 5. 查用户
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise AuthenticationError(
            status_code=401,
            error_code="AUTH_USER_NOT_FOUND",
            message="用户不存在",
        )
    return user
