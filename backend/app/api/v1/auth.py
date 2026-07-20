from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limiter import limiter
from app.core.config import settings

from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from app.core.deps import get_current_user
from app.core.audit import record_audit_log
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import (
    UserRegister, UserLogin, UserUpdate, UserOut, Token, RefreshTokenRequest,
)
from app.services.user_service import (
    create_user,
    authenticate_user,
    get_user_by_username,
    get_user_by_email,
)
from app.services.jwt_blacklist import blacklist_token

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@router.post("/register", response_model=UserOut)
@limiter.limit("5/minute")
async def register(request: Request, data: UserRegister, db: AsyncSession = Depends(get_db)):
    if await get_user_by_username(db, data.username):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "AUTH_USERNAME_EXISTS", "message": "用户名已存在"},
        )
    if data.email and await get_user_by_email(db, data.email):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "AUTH_EMAIL_EXISTS", "message": "邮箱已注册"},
        )
    user = await create_user(db, data)
    return user


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, data: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, data.username, data.password)
    if not user:
        # 记录失败登录尝试（user_id 为 None，因为认证失败）
        await record_audit_log(
            db, user_id=None, action="login", request=request,
            detail=f"登录失败: username={data.username}",
        )
        if db is not None:
            await db.commit()
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_INVALID_CREDENTIALS", "message": "用户名或密码错误"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    # 记录成功登录
    await record_audit_log(
        db, user_id=user.id, action="login", request=request,
        detail=f"登录成功: username={user.username}",
    )
    if db is not None:
        await db.commit()
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserOut.model_validate(user),
    )


@router.post("/refresh")
async def refresh_token(data: RefreshTokenRequest):
    """使用 refresh_token 获取新的 access_token"""
    payload = verify_refresh_token(data.refresh_token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "AUTH_INVALID_REFRESH_TOKEN", "message": "无效的 refresh_token，请重新登录"},
        )
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "AUTH_INVALID_REFRESH_TOKEN", "message": "无效的 refresh_token"},
        )
    new_access_token = create_access_token(data={"sub": user_id_str})
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
    }


@router.post("/logout")
async def logout(
    request: Request,
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
):
    """登出：将当前 access_token 加入黑名单"""
    success = await blacklist_token(token)
    await record_audit_log(
        None, user_id=current_user.id, action="logout", request=request,
        detail=f"登出: username={current_user.username}, blacklist={'ok' if success else 'skipped'}",
    )
    return {"message": "已成功登出"}


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/profile", response_model=UserOut)
async def update_profile(
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    update_data = data.model_dump(exclude_unset=True)
    if "email" in update_data:
        existing = await get_user_by_email(db, update_data["email"])
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "AUTH_EMAIL_IN_USE", "message": "邮箱已被使用"},
            )
    for k, v in update_data.items():
        setattr(current_user, k, v)
    await db.commit()
    await db.refresh(current_user)
    return current_user
