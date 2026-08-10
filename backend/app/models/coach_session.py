"""Coach session model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CoachSession(Base):
    __tablename__ = "coach_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("consultations.id"), nullable=False, unique=True
    )
    doctor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    mode: Mapped[str] = mapped_column(
        String(20), nullable=False, default="off",
        comment="off / on_demand / shadow",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active",
        comment="active / ended / error",
    )
    thread_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_bundle_version: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_turn_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
