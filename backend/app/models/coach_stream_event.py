"""Coach stream event model for V1.2 agent intelligence.

Stores SSE (Server-Sent Events) stream events for coach sessions,
enabling replay and durability guarantees.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CoachStreamEvent(Base):
    __tablename__ = "coach_stream_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_coach_stream_event_id"),
        UniqueConstraint(
            "session_id", "sequence", name="uq_coach_stream_session_sequence"
        ),
        Index(
            "ix_coach_stream_session_expires",
            "session_id", "expires_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        String(36), nullable=False, comment="UUID for the event"
    )
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("coach_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="monotonically increasing per session"
    )
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="e.g. suggestion, heartbeat, error"
    )
    data_json: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True, comment="event payload"
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="TTL for cleanup"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
