"""Agent trace event model for V1.2 agent intelligence."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AgentTraceEvent(Base):
    __tablename__ = "agent_trace_events"
    __table_args__ = (
        UniqueConstraint(
            "trace_id", "sequence", name="uq_trace_event_trace_sequence"
        ),
        Index("ix_trace_event_session_created", "session_id", "created_at"),
        Index("ix_trace_event_type_created", "event_type", "created_at"),
        Index("ix_trace_event_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    agent_name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    node_name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="started / finished / error / blocked",
    )
    input_payload: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True, default=None
    )
    output_payload: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True, default=None
    )
    input_hmac: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_hmac: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input_char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
