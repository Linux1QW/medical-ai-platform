"""Trainee memory model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TraineeMemory(Base):
    __tablename__ = "trainee_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="candidate",
        comment="candidate / approved / rejected / expired",
    )
    skill_dimension: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    evidence_refs: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    reviewer_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    review_comment: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
