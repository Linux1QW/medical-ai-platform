"""评估节点执行结果（审计）"""


from sqlalchemy import JSON, BigInteger, Column, DateTime, ForeignKey, Integer, String

from app.models.base import Base


class EvaluationNodeResult(Base):
    __tablename__ = "evaluation_node_results"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("evaluation_runs.id"), nullable=False)
    node_name = Column(String(50), nullable=False)
    attempt = Column(Integer, nullable=False, default=1)
    status = Column(String(20), nullable=False, comment="success/skipped/error/insufficient")
    duration_ms = Column(Integer, nullable=True)
    result_summary = Column(JSON, nullable=True)
    error_type = Column(String(100), nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
