"""模型版本注册表"""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True, comment="模型名称")
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment="版本号")
    config_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, comment="模型配置")
    status: Mapped[str] = mapped_column(
        Enum("active", "inactive", "deprecated", name="model_version_status"),
        nullable=False,
        default="active",
        comment="状态",
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="版本描述")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
