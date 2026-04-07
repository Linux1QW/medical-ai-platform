from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "AUTH_INVALID_TOKEN", "message": "无效的认证凭据"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_INVALID_TOKEN", "message": "无效的认证凭据"},
        )
    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_INVALID_TOKEN", "message": "无效的认证凭据"},
        )

    from app.services.user_service import get_user_by_id

    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "AUTH_USER_NOT_FOUND", "message": "用户不存在"},
        )
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail={"error_code": "AUTH_FORBIDDEN", "message": "权限不足，需要管理员角色"},
        )
    return current_user
