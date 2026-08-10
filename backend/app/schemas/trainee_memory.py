"""Trainee memory API schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateMemoryRequest(BaseModel):
    """Doctor identity comes from token, not request body."""
    skill_dimension: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(default_factory=list)


class ReviewMemoryRequest(BaseModel):
    """Reviewer identity comes from token, not request body."""
    action: Literal["approve", "reject"]
    review_comment: str | None = Field(default=None, max_length=500)
    ttl_days: int = Field(default=90, ge=1, le=365)


class MemoryResponse(BaseModel):
    memory_id: int
    doctor_id: int
    status: str
    skill_dimension: str
    summary: str
    evidence_refs: list[str] | None = None
    reviewer_id: int | None = None
    review_comment: str | None = None
    reviewed_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime


class ConsentRequest(BaseModel):
    """Doctor identity comes from token."""
    consent: bool


class ConsentResponse(BaseModel):
    doctor_id: int
    consent: bool
