from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserRegister(BaseModel):
    username: str
    email: Optional[str] = ""
    password: str
    real_name: Optional[str] = ""
    department: Optional[str] = ""


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    real_name: Optional[str] = None
    department: Optional[str] = None
    email: Optional[EmailStr] = None


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
