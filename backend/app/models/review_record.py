from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ReviewRecord(Base):
    """人工复核记录"""

    __tablename__ = "review_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    reviewer_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    feedback: Mapped[str] = mapped_column(Text, nullable=False)
    review_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    score_adjustments: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
