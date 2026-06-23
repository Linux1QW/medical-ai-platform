"""评估运行记录"""

from datetime import datetime

from sqlalchemy import Column, String, Integer, JSON, DateTime, Text, ForeignKey

from app.models.base import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True, comment="run_id (UUID)")
    consultation_id = Column(Integer, ForeignKey("consultations.id"), nullable=False)
    evaluation_id = Column(Integer, nullable=True)
    graph_version = Column(String(50), nullable=False, default="evaluation-graph-v1")
    scoring_policy_version = Column(String(50), nullable=False, default="v1")
    checkpoint_thread_id = Column(String(100), nullable=False)
    status = Column(String(30), nullable=False, default="running")
    selected_agents = Column(JSON, nullable=True)
    attempt = Column(Integer, nullable=False, default=1)
    error_type = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
