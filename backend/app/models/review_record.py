from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, String, Text

from app.models.base import Base


class ReviewRecord(Base):
    """人工复核记录"""

    __tablename__ = "review_records"

    id = Column(String(36), primary_key=True)
    evaluation_id = Column(String(36), nullable=False, index=True)
    reviewer_id = Column(String(50), nullable=False, index=True)
    feedback = Column(Text, nullable=False)
    review_reason = Column(String(255), nullable=True)
    score_adjustments = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
