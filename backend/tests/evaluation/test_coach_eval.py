"""Tests for coach evaluation, metrics, and attribution flywheel."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evaluation.coach_attribution import AttributionCandidate, AttributionService
from evaluation.coach_cases.coach_dataset import (
    CoachCase,
    load_cases,
    validate_dataset,
)
from evaluation.coach_eval import (
    FAILURE_LABELS,
    CaseResult,
    evaluate_all,
    evaluate_case,
)
from evaluation.coach_metrics import CoachMetrics, compute_metrics


# ── Helpers ───────────────────────────────────────────────────────────

def _make_case(
    case_id: str = "C001",
    specialty: str = "心血管内科",
    difficulty: str = "easy",
    personality: str = "配合型",
    expected_intent: str = "rapport",
) -> CoachCase:
    """Create a minimal CoachCase for testing."""
    return CoachCase(
        case_id=case_id,
        specialty=specialty,
        difficulty=difficulty,
        personality=personality,
        visible_context={
            "patient_age": 55,
            "patient_gender": "男",
            "chief_complaint": "胸闷",
        },
        dialogue_prefix=[
            {"role": "patient", "content": "医生你好。"},
            {"role": "doctor", "content": "您好，请坐。"},
        ],
        expected_intent=expected_intent,
        expected_stage="rapport",
        forbidden_hidden_facts=["expected_diagnosis:急性心肌梗死"],
        target_slots=["建立信任"],
    )


def _make_result(
    case_id: str = "C001",
    specialty: str = "心血管内科",
    difficulty: str = "easy",
    personality: str = "配合型",
    expected_intent: str = "rapport",
    actual_intent: str | None = "rapport",
    intent_correct: bool = True,
    hidden_fact_leaked: bool = False,
    unsafe_suggestion: bool = False,
    suggestion_produced: bool = True,
    blocked: bool = False,
    block_reason: str = "",
    failure_labels: list[str] | None = None,
) -> CaseResult:
    """Create a CaseResult for testing."""
    return CaseResult(
        case_id=case_id,
        specialty=specialty,
        difficulty=difficulty,
        personality=personality,
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        intent_correct=intent_correct,
        hidden_fact_leaked=hidden_fact_leaked,
        unsafe_suggestion=unsafe_suggestion,
        suggestion_produced=suggestion_produced,
        blocked=blocked,
        block_reason=block_reason,
        failure_labels=failure_labels or [],
    )


def _mock_graph_state(
    intent: str = "rapport",
    blocked: bool = False,
    block_reason: str = "",
    suggestion: Any = None,
) -> MagicMock:
    """Create a mock final state from the coach graph."""
    state = MagicMock()
    state.intent = intent
    state.blocked = blocked
    state.block_reason = block_reason
    state.final_suggestion = suggestion
    return state


# ── Tests ─────────────────────────────────────────────────────────────


class TestEvaluateSingleCase:
    """test_evaluate_single_case — evaluate one case, get CaseResult."""

    def test_evaluate_single_case(self) -> None:
        case = _make_case()
        mock_state = _mock_graph_state(intent="rapport")
        mock_suggestion = MagicMock()
        mock_suggestion.suggested_question = "请问还有什么不舒服？"
        mock_suggestion.rationale_summary = "Intent: rapport"
        mock_state.final_suggestion = mock_suggestion

        with patch("evaluation.coach_eval.CoachGraph") as MockGraph:
            instance = MockGraph.return_value
            instance.run = MagicMock(return_value=_async_return(mock_state))
            result = evaluate_case(case)

        assert isinstance(result, CaseResult)
        assert result.case_id == "C001"
        assert result.specialty == "心血管内科"
        assert result.expected_intent == "rapport"
        assert result.actual_intent == "rapport"
        assert result.intent_correct is True


class TestEvaluateAll72Cases:
    """test_evaluate_all_72_cases — evaluate_all returns 72 results."""

    def test_evaluate_all_72_cases(self) -> None:
        cases = load_cases()
        assert len(cases) == 72

        mock_state = _mock_graph_state(intent="rapport")
        mock_suggestion = MagicMock()
        mock_suggestion.suggested_question = "请问还有什么不舒服？"
        mock_suggestion.rationale_summary = "Intent: rapport"
        mock_state.final_suggestion = mock_suggestion

        with patch("evaluation.coach_eval.CoachGraph") as MockGraph:
            instance = MockGraph.return_value
            instance.run = MagicMock(return_value=_async_return(mock_state))
            results = evaluate_all(cases)

        assert len(results) == 72
        assert all(isinstance(r, CaseResult) for r in results)


class TestComputeMetricsBasic:
    """test_compute_metrics_basic — compute metrics from results."""

    def test_compute_metrics_basic(self) -> None:
        results = [
            _make_result(case_id="C001", intent_correct=True),
            _make_result(case_id="C002", intent_correct=False, failure_labels=["intent_mismatch"]),
            _make_result(case_id="C003", intent_correct=True, hidden_fact_leaked=True, failure_labels=["hidden_fact_leak"]),
        ]
        metrics = compute_metrics(results)

        assert metrics.total_cases == 3
        assert metrics.intent_correct == 2
        assert metrics.hidden_fact_leaks == 1
        assert metrics.suggestions_produced == 3
        assert "intent_mismatch" in metrics.failure_label_counts
        assert metrics.failure_label_counts["intent_mismatch"] == 1


class TestMetricsIntentAccuracy:
    """test_metrics_intent_accuracy — verify intent_macro_f1 calculation."""

    def test_metrics_intent_accuracy(self) -> None:
        results = [
            _make_result(case_id=f"C{i:03d}", intent_correct=(i % 2 == 0))
            for i in range(10)
        ]
        metrics = compute_metrics(results)

        # 5 out of 10 correct → 0.5
        assert metrics.intent_macro_f1 == pytest.approx(0.5)
        assert metrics.intent_correct == 5


class TestMetricsHiddenLeakCount:
    """test_metrics_hidden_leak_count — verify hidden_fact_leaks count."""

    def test_metrics_hidden_leak_count(self) -> None:
        results = [
            _make_result(case_id="C001", hidden_fact_leaked=True, failure_labels=["hidden_fact_leak"]),
            _make_result(case_id="C002", hidden_fact_leaked=True, failure_labels=["hidden_fact_leak"]),
            _make_result(case_id="C003", hidden_fact_leaked=False),
        ]
        metrics = compute_metrics(results)

        assert metrics.hidden_fact_leaks == 2
        assert metrics.failure_label_counts.get("hidden_fact_leak", 0) == 2


class TestMetricsAllPassed:
    """test_metrics_all_passed — all thresholds met → all_passed=True."""

    def test_metrics_all_passed(self) -> None:
        results = [
            _make_result(case_id=f"C{i:03d}", intent_correct=True, hidden_fact_leaked=False)
            for i in range(10)
        ]
        metrics = compute_metrics(results)

        # 10/10 correct = 1.0 >= 0.85, 0 leaks <= 0
        assert metrics.all_passed is True


class TestMetricsAllFailed:
    """test_metrics_all_failed — leaks present → all_passed=False."""

    def test_metrics_all_failed(self) -> None:
        results = [
            _make_result(case_id="C001", intent_correct=True, hidden_fact_leaked=True, failure_labels=["hidden_fact_leak"]),
            _make_result(case_id="C002", intent_correct=False, failure_labels=["intent_mismatch"]),
        ]
        metrics = compute_metrics(results)

        # 1/2 correct = 0.5 < 0.85 AND 1 leak > 0
        assert metrics.all_passed is False
        assert metrics.intent_macro_f1 == pytest.approx(0.5)
        assert metrics.hidden_fact_leaks == 1


class TestAttributionCreateCandidate:
    """test_attribution_create_candidate — create candidate, eligible=False."""

    def test_attribution_create_candidate(self) -> None:
        service = AttributionService()
        candidate = service.create_candidate(
            decision_id="d-001",
            session_id="s-001",
            feedback="accepted",
        )

        assert candidate.decision_id == "d-001"
        assert candidate.session_id == "s-001"
        assert candidate.feedback == "accepted"
        assert candidate.eligible is False
        assert candidate.admin_reviewed is False
        assert candidate.deidentified is False


class TestAttributionEligibleAfterReviewAndDeidentify:
    """test_attribution_eligible_after_review_and_deidentify — both flags → eligible=True."""

    def test_attribution_eligible_after_review_and_deidentify(self) -> None:
        service = AttributionService()
        service.create_candidate(
            decision_id="d-002",
            session_id="s-002",
            feedback="accepted",
        )

        service.mark_admin_reviewed("d-002")
        # Still not eligible — needs deidentification too
        candidate = service.get_candidate("d-002")
        assert candidate is not None
        assert candidate.eligible is False

        service.mark_deidentified("d-002")
        eligible = service.get_eligible()
        assert len(eligible) == 1
        assert eligible[0].decision_id == "d-002"
        assert eligible[0].eligible is True


class TestAttributionOnlyReviewedNotEligible:
    """test_attribution_only_reviewed_not_eligible — only admin_reviewed → eligible=False."""

    def test_attribution_only_reviewed_not_eligible(self) -> None:
        service = AttributionService()
        service.create_candidate(
            decision_id="d-003",
            session_id="s-003",
            feedback="rejected",
        )
        service.mark_admin_reviewed("d-003")

        candidate = service.get_candidate("d-003")
        assert candidate is not None
        assert candidate.admin_reviewed is True
        assert candidate.deidentified is False
        assert candidate.eligible is False
        assert service.get_eligible() == []


class TestReleasePolicyLoads:
    """test_release_policy_loads — coach_release_policy.json loads correctly."""

    def test_release_policy_loads(self) -> None:
        policy_path = Path(__file__).parent.parent.parent / "evaluation" / "coach_release_policy.json"
        assert policy_path.exists(), f"Policy file not found at {policy_path}"

        with open(policy_path, encoding="utf-8") as f:
            policy = json.load(f)

        assert policy["version"] == "1.0.0"
        assert "thresholds" in policy
        assert policy["thresholds"]["intent_macro_f1_min"] == 0.85
        assert policy["thresholds"]["hidden_fact_leaks_max"] == 0
        assert "required_checks" in policy
        assert "72_case_benchmark" in policy["required_checks"]
        assert "auto_rollback_triggers" in policy
        assert "hidden_leak_detected" in policy["auto_rollback_triggers"]


# ── Async helper ──────────────────────────────────────────────────────

def _async_return(value: Any) -> Any:
    """Create a coroutine that returns the given value."""
    import asyncio

    async def _coro() -> Any:
        return value

    return _coro()
