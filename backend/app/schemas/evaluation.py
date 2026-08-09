from datetime import datetime, timezone
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CitationOut(BaseModel):
    citation_id: str
    claim: str
    source: str
    page: Optional[int] = None
    heading_path: str = ""
    text_snippet: str = ""
    rerank_score: Optional[float] = None


class EvaluationOut(BaseModel):
    id: int
    consultation_id: int
    inquiry_score: float
    inquiry_analysis: str
    knowledge_score: Optional[float] = None
    knowledge_analysis: str
    humanistic_score: float
    humanistic_analysis: str
    diagnosis_score: float
    diagnosis_analysis: str
    treatment_score: float
    treatment_analysis: str
    total_score: Optional[float] = None
    overall_summary: str
    improvement_suggestions: str
    created_at: datetime

    # RAG 审计字段
    citation_data: Optional[List[CitationOut]] = None
    retrieval_status: str = "not_run"
    evidence_stance: str = "undetermined"
    human_review_needed: bool = False
    review_reason: Optional[str] = None
    rag_trace_data: Optional[dict] = None
    evaluation_status: str = "completed"

    model_config = ConfigDict(from_attributes=True)


class EvaluationRequest(BaseModel):
    consultation_id: int


class UserStatItem(BaseModel):
    user_id: int
    username: str
    real_name: str
    department: str
    total_consultations: int
    total_evaluations: int
    avg_inquiry_score: float = 0
    avg_knowledge_score: float = 0
    avg_humanistic_score: float = 0
    avg_diagnosis_score: float = 0
    avg_treatment_score: float = 0
    avg_total_score: float = 0


class StatsSummary(BaseModel):
    total_consultations: int
    total_evaluations: int
    avg_inquiry_score: Optional[float] = 0
    avg_knowledge_score: Optional[float] = 0
    avg_humanistic_score: Optional[float] = 0
    avg_diagnosis_score: Optional[float] = 0
    avg_treatment_score: Optional[float] = 0
    avg_total_score: Optional[float] = 0
    score_distribution: List[dict] = []
    user_stats: Optional[List[UserStatItem]] = None


# === V1.1 Evaluation Job Contracts ===

EvaluationJobStatus = Literal[
    "queued", "running", "retrying", "completed",
    "needs_review", "reviewed", "failed", "cancelled",
]


class EvaluationSubmitOut(BaseModel):
    """202 response for evaluation submission."""
    run_id: UUID
    consultation_id: int
    status: Literal["queued"]
    status_url: str
    websocket_url: str


class EvaluationRunStatusOut(BaseModel):
    """200 response for run status polling."""
    run_id: UUID
    consultation_id: int
    status: EvaluationJobStatus
    progress: int | None = Field(default=None, ge=0, le=100)
    message: str | None = None
    evaluation_id: int | None = None
    error_code: str | None = None
    attempt: int = Field(ge=0)
    cancel_requested: bool = False
    cancel_requested_at: datetime | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("submitted_at", "started_at", "finished_at", "cancel_requested_at", mode="before")
    @classmethod
    def _normalize_datetime_to_utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class EvaluationCancelOut(BaseModel):
    """Response for cancel request."""
    run_id: UUID
    status: EvaluationJobStatus
    cancel_requested: bool
    requested_at: datetime | None = None
