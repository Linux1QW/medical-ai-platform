from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Enum, Text

from app.models.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True, comment="操作用户ID")
    action = Column(
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
    resource_id = Column(String(50), nullable=True, comment="关联资源ID")
    ip_address = Column(String(45), nullable=True, comment="客户端IP")
    user_agent = Column(String(500), nullable=True, comment="客户端UA")
    detail = Column(Text, nullable=True, comment="操作详情（脱敏）")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
