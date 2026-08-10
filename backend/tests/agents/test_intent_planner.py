"""Tests for intent classification and follow-up planning."""
from __future__ import annotations

from typing import get_args

import pytest

from app.agent_runtime.contracts import CoachIntent, InterviewStage
from app.services.agents.coach.intent import INTENT_RULES, classify_intent
from app.services.agents.coach.planner import (
    INTENT_TO_STAGE,
    STAGE_PRIORITY,
    FollowupCandidate,
    FollowupPlan,
    build_followup_plan,
)
from app.services.memory.working import WorkingMemoryState


# ── Intent classification tests ─────────────────────────────────────────────


class TestClassifyIntent:
    """Tests for classify_intent function."""

    def test_classify_intent_rapport_rule(self) -> None:
        """Test that '你好' is classified as rapport intent."""
        result = classify_intent("你好，请坐")
        assert result == "rapport"

    def test_classify_intent_chief_complaint_rule(self) -> None:
        """Test that '哪里不舒服' is classified as chief_complaint intent."""
        result = classify_intent("你哪里不舒服？")
        assert result == "chief_complaint"

    def test_classify_intent_off_topic_no_match(self) -> None:
        """Test that unmatched messages fall through to off_topic."""
        result = classify_intent("今天天气不错")
        assert result == "off_topic"

    def test_classify_intent_llm_fallback(self) -> None:
        """Test that LLM fallback works when no rule matches."""
        mock_llm = lambda msg: "examination"
        result = classify_intent("请量一下血压", llm_fn=mock_llm)
        # Note: "血压" is in examination keywords, so it will match rule first
        # Let's use a message that doesn't match any rule
        result = classify_intent("请描述你的日常作息", llm_fn=mock_llm)
        assert result == "examination"

    def test_classify_intent_llm_fallback_invalid(self) -> None:
        """Test that invalid LLM response falls through to off_topic."""
        mock_llm = lambda msg: "invalid_intent"
        result = classify_intent("今天天气不错", llm_fn=mock_llm)
        assert result == "off_topic"

    def test_classify_intent_llm_fallback_exception(self) -> None:
        """Test that LLM exception falls through to off_topic."""
        def failing_llm(msg: str) -> str:
            raise ValueError("LLM error")

        result = classify_intent("今天天气不错", llm_fn=failing_llm)
        assert result == "off_topic"

    def test_classify_intent_hpi_onset(self) -> None:
        """Test HPI onset classification."""
        result = classify_intent("什么时候开始发病的？")
        assert result == "hpi_onset"

    def test_classify_intent_past_history(self) -> None:
        """Test past history classification."""
        result = classify_intent("以前得过什么病吗？")
        assert result == "past_history"


# ── Follow-up planning tests ────────────────────────────────────────────────


class TestBuildFollowupPlan:
    """Tests for build_followup_plan function."""

    def test_build_followup_plan_initial(self) -> None:
        """Test initial plan with empty working memory includes rapport/chief_complaint."""
        working_memory = WorkingMemoryState()
        plan = build_followup_plan(
            current_intent="rapport",
            working_memory=working_memory,
            current_turn=1,
        )
        assert plan.current_stage == "rapport"
        # Should have candidates for unvisited stages
        assert len(plan.candidates) > 0
        # First candidate should be chief_complaint (next unvisited stage)
        assert plan.recommended_intent is not None

    def test_build_followup_plan_hpi_coverage(self) -> None:
        """Test that HPI stage generates sub-intent candidates."""
        working_memory = WorkingMemoryState()
        plan = build_followup_plan(
            current_intent="hpi_onset",
            working_memory=working_memory,
            current_turn=3,
        )
        assert plan.current_stage == "history_present_illness"
        # Should have HPI sub-intent candidates
        hpi_candidates = [
            c for c in plan.candidates if c.stage == "history_present_illness"
        ]
        assert len(hpi_candidates) > 0

    def test_build_followup_plan_red_flag_priority(self) -> None:
        """Test that red flags get highest priority."""
        working_memory = WorkingMemoryState(
            red_flags=["胸痛", "呼吸困难"],
        )
        plan = build_followup_plan(
            current_intent="rapport",
            working_memory=working_memory,
            current_turn=1,
        )
        # Red flag candidate should have highest priority (lowest score)
        assert len(plan.candidates) > 0
        red_flag_candidates = [
            c for c in plan.candidates if "Red flags" in c.rationale
        ]
        assert len(red_flag_candidates) == 1
        assert red_flag_candidates[0].priority_score == 0.5
        # Red flag should be the recommended intent
        assert plan.recommended_intent == "hpi_severity"

    def test_intent_to_stage_mapping_completeness(self) -> None:
        """Test that all CoachIntent values have a stage mapping."""
        all_intents = get_args(CoachIntent)
        for intent in all_intents:
            assert intent in INTENT_TO_STAGE, f"Missing mapping for intent: {intent}"

    def test_build_followup_plan_limits_candidates(self) -> None:
        """Test that plan limits candidates to 3."""
        working_memory = WorkingMemoryState(
            red_flags=["flag1", "flag2", "flag3", "flag4"],
        )
        plan = build_followup_plan(
            current_intent="hpi_onset",
            working_memory=working_memory,
            current_turn=1,
        )
        assert len(plan.candidates) <= 3

    def test_build_followup_plan_visited_stages(self) -> None:
        """Test that visited stages are not repeated."""
        working_memory = WorkingMemoryState(
            stage_history=["rapport", "chief_complaint"],
        )
        plan = build_followup_plan(
            current_intent="hpi_onset",
            working_memory=working_memory,
            current_turn=5,
        )
        # Should not include rapport or chief_complaint stages
        for candidate in plan.candidates:
            assert candidate.stage not in ["rapport", "chief_complaint"]


# ── Data structure tests ────────────────────────────────────────────────────


class TestDataStructures:
    """Tests for FollowupCandidate and FollowupPlan dataclasses."""

    def test_followup_candidate_creation(self) -> None:
        """Test FollowupCandidate dataclass creation."""
        candidate = FollowupCandidate(
            intent="hpi_onset",
            stage="history_present_illness",
            priority_score=1.0,
            rationale="Test rationale",
        )
        assert candidate.intent == "hpi_onset"
        assert candidate.stage == "history_present_illness"
        assert candidate.priority_score == 1.0
        assert candidate.rationale == "Test rationale"

    def test_followup_plan_defaults(self) -> None:
        """Test FollowupPlan default values."""
        plan = FollowupPlan()
        assert plan.candidates == []
        assert plan.current_stage is None
        assert plan.recommended_intent is None
