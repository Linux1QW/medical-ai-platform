from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


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

    class Config:
        from_attributes = True


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
