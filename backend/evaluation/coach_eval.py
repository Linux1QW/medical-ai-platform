"""72-case coach benchmark evaluator."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal, cast

from app.agent_runtime.contracts import (
    CoachContextView,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.agent_runtime.graph import CoachGraph, CoachGraphState
from evaluation.coach_cases.coach_dataset import CoachCase, load_cases


@dataclass
class CaseResult:
    """Result of evaluating a single case."""
    case_id: str
    specialty: str
    difficulty: str
    personality: str
    expected_intent: str
    actual_intent: str | None = None
    intent_correct: bool = False
    hidden_fact_leaked: bool = False
    unsafe_suggestion: bool = False
    suggestion_produced: bool = False
    blocked: bool = False
    block_reason: str = ""
    failure_labels: list[str] = field(default_factory=list)


FAILURE_LABELS = [
    "intent_mismatch",
    "hidden_fact_leak",
    "unsafe_suggestion",
    "no_suggestion",
    "blocked_by_safety",
    "diagnostic_phrasing",
    "repeated_question",
    "unsupported_citation",
    "context_overflow",
    "stage_mismatch",
    "timeout",
    "graph_error",
    "schema_violation",
]


def evaluate_case(case: CoachCase) -> CaseResult:
    """Evaluate a single benchmark case through the coach graph."""
    result = CaseResult(
        case_id=case.case_id,
        specialty=case.specialty,
        difficulty=case.difficulty,
        personality=case.personality,
        expected_intent=case.expected_intent,
    )

    try:
        # Build context view from case
        messages = []
        for i, msg in enumerate(case.dialogue_prefix):
            messages.append(VisibleMessage(
                sequence=i + 1,
                role=cast(Literal["doctor", "patient", "system"], msg["role"]),
                content=msg["content"],
            ))

        vc = case.visible_context
        context_view = CoachContextView(
            consultation_id=hash(case.case_id) % 10000,
            doctor_id=1,
            visible_patient=VisiblePatientProfile(
                age=vc.get("patient_age", 45),
                gender=vc.get("patient_gender", "未知"),
                chief_complaint=vc.get("chief_complaint", "问诊"),
            ),
            messages=messages,
        )

        # The last doctor message is what we classify intent for
        last_doctor_msg = ""
        for msg in reversed(case.dialogue_prefix):
            if msg["role"] == "doctor":
                last_doctor_msg = msg["content"]
                break

        # Run coach graph
        graph = CoachGraph()
        state = CoachGraphState(
            context_view=context_view,
            latest_message=last_doctor_msg,
            turn_no=len(messages),
        )

        final_state = asyncio.get_event_loop().run_until_complete(graph.run(state))

        result.actual_intent = final_state.intent
        result.intent_correct = (final_state.intent == case.expected_intent)
        result.blocked = final_state.blocked
        result.block_reason = final_state.block_reason
        result.suggestion_produced = final_state.final_suggestion is not None

        # Check hidden-fact leakage
        if final_state.final_suggestion:
            suggestion_text = (
                final_state.final_suggestion.suggested_question + " " +
                final_state.final_suggestion.rationale_summary
            ).lower()
            for fact in case.forbidden_hidden_facts:
                # Extract the value part after ":"
                fact_value = fact.split(":", 1)[-1].lower() if ":" in fact else fact.lower()
                if fact_value and fact_value in suggestion_text:
                    result.hidden_fact_leaked = True
                    break

        # Build failure labels
        if not result.intent_correct:
            result.failure_labels.append("intent_mismatch")
        if result.hidden_fact_leaked:
            result.failure_labels.append("hidden_fact_leak")
        if result.blocked:
            result.failure_labels.append("blocked_by_safety")
        if not result.suggestion_produced and not result.blocked:
            result.failure_labels.append("no_suggestion")

    except Exception as e:
        result.failure_labels.append("graph_error")
        result.block_reason = str(e)

    return result


def evaluate_all(cases: list[CoachCase] | None = None) -> list[CaseResult]:
    """Run all benchmark cases."""
    if cases is None:
        cases = load_cases()
    return [evaluate_case(case) for case in cases]
