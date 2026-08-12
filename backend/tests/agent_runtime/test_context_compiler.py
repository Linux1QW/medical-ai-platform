"""Tests for context compiler."""
from uuid import uuid4

import pytest

from app.agent_runtime.context import (
    DEFAULT_COACH_BUDGET,
    CompiledContext,
    ContextBudget,
    ContextBudgetExceeded,
    _count_tokens,
    compile_coach_context,
)
from app.agent_runtime.contracts import (
    ApprovedMemory,
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
    # Budget large enough for system+policy but tiny for everything else
    adequate_budget = ContextBudget(
        total_tokens=1000,
        output_reserve=100,
        system_tokens=500,
        policy_tokens=300,
        memory_tokens=5,
        dialogue_tokens=5,
    )
    result = compile_coach_context(view, budget=adequate_budget)
    # System and policy are preserved in full
    assert result.system_section is not None
    assert result.policy_section is not None
    # With tiny memory/dialogue budget, they should be None (no content to fit)
    assert result.memory_section is None
    assert result.dialogue_section is None


def test_200_messages_deterministic_trimming_retains_latest():
    """200 messages with Chinese text: trimming retains latest turns deterministically."""
    messages = [
        VisibleMessage(
            sequence=i + 1,
            role="doctor" if i % 2 == 0 else "patient",
            content=f"这是第{i + 1}条对话消息，包含一些中文内容来模拟真实的问诊场景",
        )
        for i in range(200)
    ]
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=55, gender="male", chief_complaint="头痛"
        ),
        messages=messages,
    )
    result = compile_coach_context(view, budget=DEFAULT_COACH_BUDGET)

    # Dialogue section should exist but not contain all 200 messages
    assert result.dialogue_section is not None
    dialogue_lines = result.dialogue_section.split("\n")
    assert len(dialogue_lines) < 200

    # Latest message (sequence=200) should be retained
    assert "第200条" in result.dialogue_section
    # Earliest message (sequence=1) should be trimmed
    assert "第1条" not in result.dialogue_section


def test_100_token_budget_fits_or_raises():
    """100-token budget either fits or raises ContextBudgetExceeded."""
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[
            VisibleMessage(sequence=1, role="patient", content="我胸闷"),
        ],
    )
    tiny_budget = ContextBudget(
        total_tokens=100,
        output_reserve=10,
        system_tokens=50,
        policy_tokens=30,
        memory_tokens=5,
        dialogue_tokens=5,
    )
    # Either it fits (total_tokens <= 90) or raises ContextBudgetExceeded
    try:
        result = compile_coach_context(view, budget=tiny_budget)
        assert result.total_tokens <= 90
    except ContextBudgetExceeded as e:
        assert e.mandatory_tokens > e.budget_limit


def test_mandatory_sections_exceed_budget_raises():
    """When system+policy alone exceed input_limit, raise ContextBudgetExceeded."""
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[],
    )
    # Budget so tiny that system+policy can't fit
    impossible_budget = ContextBudget(
        total_tokens=20,
        output_reserve=5,
        system_tokens=5,
        policy_tokens=5,
        memory_tokens=0,
        dialogue_tokens=0,
    )
    with pytest.raises(ContextBudgetExceeded):
        compile_coach_context(view, budget=impossible_budget)


def test_token_count_is_accurate_and_non_constant():
    """Token count reflects actual content, not a constant."""
    view_empty = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[],
    )
    view_many = CoachContextView(
        consultation_id=2,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[
            VisibleMessage(sequence=1, role="patient", content="我最近三天总是觉得胸闷，活动后加重"),
            VisibleMessage(sequence=2, role="doctor", content="胸闷是什么情况下会出现？休息后能缓解吗？"),
            VisibleMessage(sequence=3, role="patient", content="走路快了或者爬楼梯的时候比较明显，休息几分钟就好"),
        ],
    )
    result_empty = compile_coach_context(view_empty, budget=DEFAULT_COACH_BUDGET)
    result_many = compile_coach_context(view_many, budget=DEFAULT_COACH_BUDGET)

    # Token counts should differ
    assert result_empty.total_tokens != result_many.total_tokens
    # More content = more tokens
    assert result_many.total_tokens > result_empty.total_tokens
    # Total tokens should be actual count, not the budget limit
    assert result_many.total_tokens < DEFAULT_COACH_BUDGET.total_tokens


def test_count_tokens_chinese_vs_english():
    """Token counting differentiates Chinese and English text."""
    chinese_text = "这是一个中文测试句子，包含多个中文字符"
    english_text = "This is an English test sentence"

    chinese_tokens = _count_tokens(chinese_text)
    english_tokens = _count_tokens(english_text)

    # Both should be > 0
    assert chinese_tokens > 0
    assert english_tokens > 0
    # Chinese chars should produce more tokens per character
    assert chinese_tokens > english_tokens


def test_memory_trimming_respects_budget():
    """Memories that exceed memory_tokens budget are trimmed."""
    memories = [
        ApprovedMemory(
            memory_id=uuid4(),
            skill_dimension=f"维度{i}",
            summary="A" * 200,  # Long summary to consume tokens
        )
        for i in range(10)
    ]
    view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[],
        approved_profile_memories=memories,
    )
    # Small memory budget
    small_memory_budget = ContextBudget(
        total_tokens=16_000,
        output_reserve=2_000,
        system_tokens=1_800,
        policy_tokens=1_200,
        memory_tokens=200,  # Very small
        dialogue_tokens=6_000,
    )
    result = compile_coach_context(view, budget=small_memory_budget)

    # Memory section should be limited
    if result.memory_section:
        mem_lines = result.memory_section.split("\n")
        assert len(mem_lines) < 10  # Not all memories fit
