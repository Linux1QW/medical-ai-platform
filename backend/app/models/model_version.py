"""模型版本注册表"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, JSON, DateTime, Text, Enum

from app.models.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True, comment="模型名称")
    version = Column(String(50), nullable=False, comment="版本号")
    config_json = Column(JSON, nullable=True, comment="模型配置")
    status = Column(
        Enum("active", "inactive", "deprecated", name="model_version_status"),
        nullable=False,
        default="active",
        comment="状态",
    )
    description = Column(Text, nullable=True, comment="版本描述")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
