"""Deterministic follow-up plan builder."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agent_runtime.contracts import CoachIntent, InterviewStage
from app.services.memory.working import WorkingMemoryState

# Intent → stage mapping
INTENT_TO_STAGE: dict[str, InterviewStage] = {
    "rapport": "rapport",
    "chief_complaint": "chief_complaint",
    "hpi_onset": "history_present_illness",
    "hpi_progression": "history_present_illness",
    "hpi_severity": "history_present_illness",
    "hpi_timing": "history_present_illness",
    "hpi_associated": "history_present_illness",
    "past_history": "past_medical_history",
    "medication": "medication_allergy",
    "allergy": "medication_allergy",
    "family_social": "family_social_history",
    "examination": "physical_examination",
    "assessment_communication": "assessment_communication",
    "treatment_communication": "assessment_communication",  # reuse stage
    "closing": "closing",
    "off_topic": "off_topic",
    "unsafe": "off_topic",
}

# Stage priority order (lower = higher priority)
STAGE_PRIORITY: list[InterviewStage] = [
    "rapport",
    "chief_complaint",
    "history_present_illness",
    "past_medical_history",
    "medication_allergy",
    "family_social_history",
    "physical_examination",
    "assessment_communication",
    "closing",
]


@dataclass
class FollowupCandidate:
    """A single follow-up candidate in the plan."""

    intent: CoachIntent
    stage: InterviewStage
    priority_score: float  # lower = higher priority
    rationale: str


@dataclass
class FollowupPlan:
    """A deterministic follow-up plan with 1-3 candidates."""

    candidates: list[FollowupCandidate] = field(default_factory=list)
    current_stage: InterviewStage | None = None
    recommended_intent: CoachIntent | None = None


def build_followup_plan(
    *,
    current_intent: CoachIntent,
    working_memory: WorkingMemoryState,
    current_turn: int,
) -> FollowupPlan:
    """Build a deterministic follow-up plan.

    Priority logic:
    1. Current stage's uncovered sub-intents (HPI has 5 sub-intents)
    2. Next unvisited stage by priority order
    3. Red-flag check if red_flags exist in working memory

    Returns 1-3 candidates sorted by priority.
    """
    candidates: list[FollowupCandidate] = []
    current_stage = INTENT_TO_STAGE.get(current_intent, "off_topic")

    # Check which stages have been visited
    visited_stages = set(working_memory.stage_history)

    # Priority 1: Uncovered HPI sub-intents
    hpi_intents: list[CoachIntent] = [
        "hpi_onset",
        "hpi_progression",
        "hpi_severity",
        "hpi_timing",
        "hpi_associated",
    ]
    if current_stage == "history_present_illness":
        for intent in hpi_intents:
            dim = f"hpi_{intent}"
            if not working_memory.is_repeated(
                dim, within_turns=2, current_turn=current_turn
            ):
                candidates.append(
                    FollowupCandidate(
                        intent=intent,
                        stage="history_present_illness",
                        priority_score=1.0 + len(candidates) * 0.1,
                        rationale=f"Uncovered HPI dimension: {intent}",
                    )
                )

    # Priority 2: Next unvisited stage
    for stage in STAGE_PRIORITY:
        if stage not in visited_stages and stage != current_stage:
            # Find a representative intent for this stage
            stage_intent = next(
                (intent for intent, stg in INTENT_TO_STAGE.items() if stg == stage),
                "off_topic",
            )
            candidates.append(
                FollowupCandidate(
                    intent=stage_intent,  # type: ignore[arg-type]
                    stage=stage,
                    priority_score=2.0 + len(candidates) * 0.1,
                    rationale=f"Unvisited stage: {stage}",
                )
            )
            break  # Only add one next-stage candidate

    # Priority 3: Red flag follow-up
    if working_memory.red_flags:
        candidates.append(
            FollowupCandidate(
                intent="hpi_severity",
                stage="history_present_illness",
                priority_score=0.5,  # High priority
                rationale=f"Red flags detected: {', '.join(working_memory.red_flags)}",
            )
        )

    # Sort by priority and limit to 3
    candidates.sort(key=lambda c: c.priority_score)
    candidates = candidates[:3]

    recommended = candidates[0] if candidates else None

    return FollowupPlan(
        candidates=candidates,
        current_stage=current_stage,
        recommended_intent=recommended.intent if recommended else None,
    )
