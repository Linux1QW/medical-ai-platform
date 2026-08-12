"""Typed contracts for V1.2 agent runtime."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Intent labels for clinical interview
CoachIntent = Literal[
    "rapport",
    "chief_complaint",
    "hpi_onset",
    "hpi_progression",
    "hpi_severity",
    "hpi_timing",
    "hpi_associated",
    "past_history",
    "medication",
    "allergy",
    "family_social",
    "examination",
    "assessment_communication",
    "treatment_communication",
    "closing",
    "off_topic",
    "unsafe",
]

# Interview stages
InterviewStage = Literal[
    "rapport",
    "chief_complaint",
    "history_present_illness",
    "past_medical_history",
    "medication_allergy",
    "family_social_history",
    "physical_examination",
    "assessment_communication",
    "closing",
    "off_topic",
]


class VisiblePatientProfile(BaseModel):
    """Patient information visible to the doctor."""
    age: int = Field(ge=0, le=150)
    gender: str = Field(min_length=1, max_length=20)
    chief_complaint: str = Field(min_length=1, max_length=500)


class VisibleMessage(BaseModel):
    """A message visible in the consultation."""
    sequence: int = Field(ge=1)
    role: Literal["doctor", "patient", "system"]
    content: str = Field(min_length=0, max_length=5000)


class ApprovedMemory(BaseModel):
    """An approved trainee profile memory."""
    memory_id: UUID
    skill_dimension: str = Field(min_length=1, max_length=50)
    summary: str = Field(min_length=1, max_length=500)


class CoachContextView(BaseModel):
    """Context view available to the Coach Agent.

    STRICT: extra fields are forbidden to prevent hidden fact leakage.
    """
    model_config = ConfigDict(extra="forbid")

    consultation_id: int
    doctor_id: int
    visible_patient: VisiblePatientProfile
    messages: list[VisibleMessage] = Field(default_factory=list)
    approved_profile_memories: list[ApprovedMemory] = Field(default_factory=list)


class CoachSuggestion(BaseModel):
    """A coach suggestion emitted to the doctor.

    MUST NOT contain chain_of_thought or reasoning fields.
    Only rationale_summary is allowed.
    """
    suggestion_id: UUID
    session_id: UUID
    turn_no: int = Field(ge=1)
    intent: CoachIntent
    stage: InterviewStage
    suggested_question: str = Field(min_length=2, max_length=240)
    rationale_summary: str = Field(min_length=2, max_length=300)
    targeted_slots: list[str] = Field(default_factory=list, max_length=5)
    citation_ids: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]


class AgentRequest(BaseModel):
    """A request to an agent node."""
    request_id: UUID
    session_id: UUID
    turn_no: int = Field(ge=1)
    agent_name: str = Field(min_length=1, max_length=50)
    input_data: dict = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """A decision produced by an agent node."""
    decision_id: UUID
    request_id: UUID
    agent_name: str = Field(min_length=1, max_length=50)
    status: Literal["success", "degraded", "blocked", "error"]
    output_data: dict = Field(default_factory=dict)
    token_count: int = Field(ge=0, default=0)
    latency_ms: int = Field(ge=0, default=0)
