"""LangGraph state definition for the Coach Agent.

All fields are typed. No hidden patient facts, no raw chain-of-thought.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.agent_runtime.contracts import (
    CoachContextView,
    CoachIntent,
    CoachSuggestion,
    InterviewStage,
)
from app.agent_runtime.context import ContextBudget
from app.agent_runtime.critic import CriticResult
from app.services.agents.coach.planner import FollowupPlan

# ── Value types flowing through the graph ────────────────────────────────────


class IntentDecision(BaseModel):
    """Structured intent classification result."""

    intent: CoachIntent
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    rationale: str = Field(default="", max_length=200)


class EvidenceItem(BaseModel):
    """A single piece of retrieved evidence."""

    source: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=0, max_length=2000)
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    doc_id: str = Field(default="", max_length=100)


class DraftSuggestion(BaseModel):
    """Intermediate draft before critic review."""

    intent: CoachIntent
    stage: InterviewStage
    suggested_question: str = Field(min_length=2, max_length=240)
    rationale_summary: str = Field(min_length=2, max_length=300)
    targeted_slots: list[str] = Field(default_factory=list, max_length=5)
    citation_ids: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    risk_level: Literal["low", "medium", "high"] = "low"


GraphStatus = Literal["pending", "running", "degraded", "done"]


@dataclass
class CoachGraphState:
    """Typed state flowing through the LangGraph coach pipeline.

    STRICT: no hidden patient facts, no raw chain-of-thought.
    """

    # Input
    context: CoachContextView
    turn: int = 1
    session_id: UUID = field(default_factory=UUID)  # placeholder, set at invoke
    latest_message: str = ""

    # Working memory (pre-turn snapshot for repetition check)
    asked_dimensions: dict[str, int] = field(default_factory=dict)

    # Intermediate results
    intent_result: IntentDecision | None = None
    plan: FollowupPlan | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    draft: DraftSuggestion | None = None
    critic_result: CriticResult | None = None

    # Output
    final_suggestion: CoachSuggestion | None = None

    # Budget & control
    budget: ContextBudget = field(default_factory=ContextBudget)
    status: str = "pending"
    trace_refs: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
