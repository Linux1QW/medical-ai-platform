from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, index=True, comment="操作用户ID"
    )
    action: Mapped[str] = mapped_column(
        Enum(
            "login",
            "create_consultation",
            "submit_diagnosis",
            "trigger_evaluation",
            "admin_action",
            name="audit_action_type",
        ),
        nullable=False,
        comment="操作类型",
    )
    resource_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="关联资源ID"
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45), nullable=True, comment="客户端IP"
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, comment="客户端UA"
    )
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="操作详情（脱敏）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
