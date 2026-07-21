from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.models.base import Base


class EvaluationLock(Base):
    """评估任务锁 — 防止同一问诊重复提交评估"""

    __tablename__ = "evaluation_locks"

    consultation_id = Column(
        Integer, ForeignKey("consultations.id"), primary_key=True
    )
    status = Column(String(20), nullable=False, default="pending")
    run_id = Column(String(36), nullable=True)
    locked_at = Column(DateTime, default=datetime.utcnow)
    heartbeat_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    error_message = Column(Text, nullable=True)

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
