from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.authentication import AuthenticationError, authenticate_access_token
from app.db.session import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """HTTP 认证依赖：调用统一 authenticate_access_token 并将 AuthenticationError 转为 HTTPException"""
    try:
        user = await authenticate_access_token(db, token)
    except AuthenticationError as e:
        headers = {"WWW-Authenticate": "Bearer"} if e.status_code == 401 else {}
        raise HTTPException(
            status_code=e.status_code,
            detail={"error_code": e.error_code, "message": e.message},
            headers=headers,
        ) from e
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail={"error_code": "AUTH_FORBIDDEN", "message": "权限不足，需要管理员角色"},
        )
    return current_user
