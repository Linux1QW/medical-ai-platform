from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Enum, Text, JSON

from app.models.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    real_name = Column(String(50), default="")
    role = Column(Enum("doctor", "admin", name="user_role"), default="doctor", nullable=False)
    department = Column(String(100), default="")
    avatar = Column(String(255), default="")
    permissions = Column(JSON, nullable=True, comment="细粒度权限列表")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
