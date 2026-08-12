"""Trainee memory model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TraineeMemory(Base):
    __tablename__ = "trainee_memories"
    __table_args__ = (
        CheckConstraint(
            "status IN ('candidate', 'approved', 'rejected', 'expired')",
            name="ck_trainee_memory_status",
        ),
        Index(
            "ix_trainee_memory_doctor_status_expires",
            "doctor_id", "status", "expires_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="candidate",
        comment="candidate / approved / rejected / expired",
    )
    skill_dimension: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    evidence_refs: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    reviewer_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True,
    )
    review_comment: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TraineeMemoryConsent(Base):
    """Persist memory consent separately from the memory record."""

    __tablename__ = "trainee_memory_consents"
    __table_args__ = (
        UniqueConstraint("doctor_id", name="uq_trainee_memory_consent_doctor_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False,
    )
    granted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="boolean: 1=granted, 0=not granted",
    )
    granted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
