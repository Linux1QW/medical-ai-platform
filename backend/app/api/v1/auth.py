from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token
from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserRegister, UserLogin, UserUpdate, UserOut, Token
from app.services.user_service import (
    create_user,
    authenticate_user,
    get_user_by_username,
    get_user_by_email,
)

router = APIRouter()


@router.post("/register", response_model=UserOut)
async def register(data: UserRegister, db: AsyncSession = Depends(get_db)):
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
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "AUTH_INVALID_CREDENTIALS", "message": "用户名或密码错误"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=access_token, user=UserOut.model_validate(user))


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
