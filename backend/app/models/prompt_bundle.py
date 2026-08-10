"""Prompt bundle model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PromptBundle(Base):
    __tablename__ = "prompt_bundles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft",
        comment="draft / active / archived",
    )
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    node_prompts: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    output_schemas: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    model_config: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    skill_manifest_checksum: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    source_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    author: Mapped[str] = mapped_column(String(120), nullable=False)
    approval_metadata: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
