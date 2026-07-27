from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EvaluationLock(Base):
    """评估任务锁 — 防止同一问诊重复提交评估"""

    __tablename__ = "evaluation_locks"

    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("consultations.id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    locked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    VALID_TRANSITIONS = {
        "pending": {"running", "failed"},
        "running": {"completed", "needs_review", "failed"},
        "failed": {"pending"},
        "completed": set(),
        "needs_review": {"reviewed"},
        "reviewed": set(),
    }

    def can_transition_to(self, new_status: str) -> bool:
        return new_status in self.VALID_TRANSITIONS.get(self.status, set())
