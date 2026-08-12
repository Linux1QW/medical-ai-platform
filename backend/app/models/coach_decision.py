"""Coach decision model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CoachDecision(Base):
    __tablename__ = "coach_decisions"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "turn_no", name="uq_coach_decision_session_turn"
        ),
        UniqueConstraint(
            "session_id", "idempotency_key",
            name="uq_coach_decision_session_idempotency",
        ),
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="ck_coach_decision_risk_level",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("coach_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    suggestion_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, default=lambda: str(uuid4()),
        comment="durable public identifier for the suggestion",
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(64), nullable=False,
        comment="client-provided idempotency key",
    )
    intent: Mapped[str] = mapped_column(String(120), nullable=False)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    suggestion_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    visible_context_hmac: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feedback_value: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    feedback_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    feedback_comment: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    feedback_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    locked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="when the decision was locked for processing"
    )
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="lease expiry for the lock"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
