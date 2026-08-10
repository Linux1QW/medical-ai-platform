"""Tests for context compiler."""
from app.agent_runtime.context import (
    DEFAULT_COACH_BUDGET,
    CompiledContext,
    ContextBudget,
    compile_coach_context,
)
from app.agent_runtime.contracts import (
    CoachContextView,
    VisibleMessage,
    VisiblePatientProfile,
)


def test_default_budget_totals():
    """Default budget has correct token allocation."""
    assert DEFAULT_COACH_BUDGET.total_tokens == 16_000
    assert DEFAULT_COACH_BUDGET.output_reserve == 2_000
    assert DEFAULT_COACH_BUDGET.system_tokens == 1_800
    assert DEFAULT_COACH_BUDGET.policy_tokens == 1_200
    assert DEFAULT_COACH_BUDGET.memory_tokens == 2_000
    assert DEFAULT_COACH_BUDGET.dialogue_tokens == 6_000
    assert DEFAULT_COACH_BUDGET.evidence_tokens == 3_000


def test_compile_context_returns_compiled():
    """compile_coach_context returns a CompiledContext."""
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[
            VisibleMessage(sequence=1, role="patient", content="我胸闷"),
            VisibleMessage(sequence=2, role="doctor", content="什么时候开始的？"),
        ],
    )
    result = compile_coach_context(view, budget=DEFAULT_COACH_BUDGET)
    assert isinstance(result, CompiledContext)
    assert result.system_section is not None
    assert result.policy_section is not None


def test_system_and_policy_never_truncated():
    """System and policy sections are never truncated regardless of budget."""
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[],
    )
    tiny_budget = ContextBudget(
        total_tokens=100,
        output_reserve=10,
        system_tokens=50,
        policy_tokens=30,
        memory_tokens=5,
        dialogue_tokens=5,
    )
    result = compile_coach_context(view, budget=tiny_budget)
    # System and policy are preserved in full
    assert result.system_section is not None
    assert result.policy_section is not None
