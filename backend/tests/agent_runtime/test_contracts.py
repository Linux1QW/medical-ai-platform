"""Tests for agent runtime contracts."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agent_runtime.contracts import (
    CoachContextView,
    CoachIntent,
    CoachSuggestion,
    InterviewStage,
    VisiblePatientProfile,
)


def test_coach_context_rejects_hidden_case_fields():
    """CoachContextView must reject expected_diagnosis and other hidden fields."""
    with pytest.raises(ValidationError):
        CoachContextView.model_validate({
            "consultation_id": 1,
            "doctor_id": 1,
            "visible_patient": {
                "age": 44,
                "gender": "female",
                "chief_complaint": "胸闷",
            },
            "messages": [],
            "expected_diagnosis": "acute coronary syndrome",
        })


def test_coach_context_accepts_valid_input():
    """CoachContextView accepts properly structured input."""
    ctx = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(
            age=44, gender="female", chief_complaint="胸闷"
        ),
        messages=[],
    )
    assert ctx.consultation_id == 1
    assert ctx.approved_profile_memories == []


def test_suggestion_never_contains_chain_of_thought():
    """CoachSuggestion must have rationale_summary but NOT chain_of_thought or reasoning."""
    fields = set(CoachSuggestion.model_fields)
    assert "chain_of_thought" not in fields
    assert "reasoning" not in fields
    assert "rationale_summary" in fields


def test_suggestion_requires_valid_risk_level():
    """CoachSuggestion risk_level must be low/medium/high."""
    with pytest.raises(ValidationError):
        CoachSuggestion(
            suggestion_id=uuid4(),
            session_id=uuid4(),
            turn_no=1,
            intent="hpi_onset",
            stage="history_present_illness",
            suggested_question="症状什么时候开始的？",
            rationale_summary="需要明确发病时间以判断急性/慢性",
            confidence=0.8,
            risk_level="critical",  # invalid
        )


def test_suggestion_valid():
    """CoachSuggestion accepts valid input."""
    s = CoachSuggestion(
        suggestion_id=uuid4(),
        session_id=uuid4(),
        turn_no=1,
        intent="hpi_onset",
        stage="history_present_illness",
        suggested_question="症状什么时候开始的？",
        rationale_summary="需要明确发病时间",
        confidence=0.8,
        risk_level="medium",
    )
    assert s.confidence == 0.8
    assert s.risk_level == "medium"


def test_coach_intent_has_required_labels():
    """CoachIntent must include key clinical interview intents."""
    valid_intents = CoachIntent.__args__
    assert "hpi_onset" in valid_intents
    assert "allergy" in valid_intents
    assert "family_social" in valid_intents
    assert "treatment_communication" in valid_intents
    assert "off_topic" in valid_intents
    assert "unsafe" in valid_intents


def test_interview_stage_has_required_values():
    """InterviewStage must include key interview stages."""
    valid_stages = InterviewStage.__args__
    assert "rapport" in valid_stages
    assert "chief_complaint" in valid_stages
    assert "history_present_illness" in valid_stages
    assert "assessment_communication" in valid_stages
    assert "closing" in valid_stages
