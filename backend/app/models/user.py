from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    real_name: Mapped[str] = mapped_column(String(50), default="", nullable=True)
    role: Mapped[str] = mapped_column(
        Enum("doctor", "admin", name="user_role"), default="doctor", nullable=False
    )
    department: Mapped[str] = mapped_column(String(100), default="", nullable=True)
    avatar: Mapped[str] = mapped_column(String(255), default="", nullable=True)
    permissions: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, comment="细粒度权限列表")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True
    )
