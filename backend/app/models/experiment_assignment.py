"""Experiment assignment model for V1.2 agent intelligence."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ExperimentAssignment(Base):
    __tablename__ = "experiment_assignments"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "subject_id", name="uq_experiment_subject"
        ),
        CheckConstraint(
            "stage IN ('planned', 'running', 'completed', 'cancelled')",
            name="ck_experiment_stage",
        ),
        CheckConstraint(
            "rollout_pct >= 0 AND rollout_pct <= 100",
            name="ck_experiment_rollout_pct",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(80), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(120), nullable=False)
    variant: Mapped[str] = mapped_column(String(50), nullable=False)
    stage: Mapped[str] = mapped_column(
        String(20), nullable=False, default="planned",
        comment="planned / running / completed / cancelled",
    )
    weight: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rollout_pct: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="0-100 percentage for gradual rollout",
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
