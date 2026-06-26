from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str | None = Field(default="", max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    real_name: str | None = Field(default="", max_length=50)
    department: str | None = Field(default="", max_length=100)


class UserLogin(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., max_length=128)


class UserUpdate(BaseModel):
    real_name: str | None = Field(default=None, max_length=50)
    department: str | None = Field(default=None, max_length=100)
    email: EmailStr | None = None


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    real_name: str
    role: str
    department: str
    avatar: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
