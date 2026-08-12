"""评估运行记录"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_consultation_created", "consultation_id", "created_at"),
        Index("ix_evaluation_runs_status_updated", "status", "updated_at"),
        Index("ix_evaluation_runs_evaluation_id", "evaluation_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, comment="run_id (UUID)")
    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("consultations.id"), nullable=False
    )
    evaluation_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    graph_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="evaluation-graph-v1"
    )
    scoring_policy_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    checkpoint_thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued")
    selected_agents: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    evaluation_plan: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True, comment="评估计划（Plan-Execute 模式）"
    )
    execution_results: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True, comment="计划步骤执行结果"
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # ── 执行租约字段（Task 2: 状态机 fencing）──
    execution_owner: Mapped[Optional[str]] = mapped_column(
        String(160), nullable=True, comment="执行者标识 (hostname:pid:invocation_uuid)"
    )
    execution_task_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, comment="Celery task_id"
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="最近心跳时间"
    )
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="租约过期时间"
    )

    # ── 取消请求字段 ──
    cancel_requested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="取消请求时间"
    )
    cancel_requested_by: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="请求取消的用户 ID"
    )
