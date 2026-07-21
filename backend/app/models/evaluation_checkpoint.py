from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, String

from app.models.base import Base


class EvaluationCheckpoint(Base):
    """评估检查点 — LangGraph 状态持久化"""

    __tablename__ = "evaluation_checkpoints"

    id = Column(String(36), primary_key=True)
    evaluation_id = Column(String(36), nullable=False, index=True)
    state_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
