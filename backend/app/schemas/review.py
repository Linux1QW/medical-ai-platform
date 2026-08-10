# -*- coding: utf-8 -*-
"""Task 8: 复核 API schema contracts"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ScoreAdjustments(BaseModel):
    """五维评分调整 — 仅非 None 字段生效"""

    inquiry_score: Optional[float] = Field(default=None, ge=0, le=100)
    knowledge_score: Optional[float] = Field(default=None, ge=0, le=100)
    humanistic_score: Optional[float] = Field(default=None, ge=0, le=100)
    diagnosis_score: Optional[float] = Field(default=None, ge=0, le=100)
    treatment_score: Optional[float] = Field(default=None, ge=0, le=100)


class ReviewSubmission(BaseModel):
    """教师提交的复核意见 — reviewer_id 由认证 token 解析，请求体不传"""

    feedback: str = Field(min_length=2, max_length=5000)
    score_adjustments: Optional[ScoreAdjustments] = None


class ReviewSubmitOut(BaseModel):
    """复核提交响应"""

    review_id: UUID
    evaluation_id: int
    run_id: UUID
    status: Literal["reviewed"]
    reviewed_at: datetime


class PendingReviewItemOut(BaseModel):
    """待复核列表项"""

    evaluation_id: int
    consultation_id: int
    run_id: Optional[UUID] = None
    review_reason: Optional[str] = None
    total_score: Optional[float] = None
    retrieval_status: str
    evidence_stance: str
    created_at: datetime
