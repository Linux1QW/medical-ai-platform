"""Tests for coach evaluation, metrics, and attribution flywheel."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evaluation.coach_attribution import AttributionService
from evaluation.coach_cases.coach_dataset import (
    CoachCase,
    load_cases,
)
from evaluation.coach_eval import (
    CaseResult,
    CoachReport,
    evaluate_all,
    evaluate_case,
    evaluate_release_policy,
    make_report,
)
from evaluation.coach_metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_metrics,
    compute_per_label_f1,
    compute_per_label_metrics,
)

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
    critic_output_present: bool = True,
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
        critic_output_present=critic_output_present,
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


# ── Tests: Evaluation ─────────────────────────────────────────────────


class TestEvaluateSingleCase:
    """test_evaluate_single_case — evaluate one case, get CaseResult."""

    def test_evaluate_single_case(self) -> None:
        case = _make_case()
        mock_state = _mock_graph_state(intent="rapport")
        mock_suggestion = MagicMock()
        mock_suggestion.suggested_question = "请问还有什么不舒服？"
        mock_suggestion.rationale_summary = "Intent: rapport"
        mock_state.final_suggestion = mock_suggestion

        async def _mock_run(state: Any) -> Any:
            return mock_state

        with patch("evaluation.coach_eval.CoachGraph") as MockGraph:
            instance = MockGraph.return_value
            instance.run = _mock_run
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

        async def _mock_run(state: Any) -> Any:
            return mock_state

        with patch("evaluation.coach_eval.CoachGraph") as MockGraph:
            instance = MockGraph.return_value
            instance.run = _mock_run
            results = evaluate_all(cases)

        assert len(results) == 72
        assert all(isinstance(r, CaseResult) for r in results)


# ── Tests: Metrics ────────────────────────────────────────────────────


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


class TestMetricsPerLabelF1:
    """test per-label precision/recall/F1 computation."""

    def test_per_label_metrics(self) -> None:
        y_true = ["A", "A", "B", "B", "C"]
        y_pred = ["A", "B", "B", "B", "A"]
        metrics = compute_per_label_metrics(y_true, y_pred)

        # A: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        assert metrics["A"]["precision"] == 0.5
        assert metrics["A"]["recall"] == 0.5
        assert metrics["A"]["f1"] == 0.5

        # B: TP=2, FP=1, FN=0 -> P=2/3, R=1.0, F1=0.8
        assert metrics["B"]["precision"] == pytest.approx(2 / 3, abs=0.01)
        assert metrics["B"]["recall"] == 1.0

        # C: TP=0, FP=0, FN=1 -> P=0, R=0, F1=0
        assert metrics["C"]["recall"] == 0.0
        assert metrics["C"]["f1"] == 0.0

    def test_macro_f1(self) -> None:
        per_label = {"A": 0.5, "B": 0.8, "C": 0.0}
        macro = compute_macro_f1(per_label)
        assert macro == pytest.approx((0.5 + 0.8 + 0.0) / 3, abs=0.01)

    def test_accuracy_separate_from_f1(self) -> None:
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "A", "A", "B"]
        acc = compute_accuracy(y_true, y_pred)
        assert acc == 0.75  # 3/4 correct


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
    """test_metrics_all_passed — all thresholds met -> all_passed=True."""

    def test_metrics_all_passed(self) -> None:
        # All correct with same intent -> macro-F1 = 1.0
        results = [
            _make_result(case_id=f"C{i:03d}", intent_correct=True, hidden_fact_leaked=False)
            for i in range(10)
        ]
        metrics = compute_metrics(results)

        # 10/10 correct = macro-F1 1.0 >= 0.85, 0 leaks <= 0
        assert metrics.all_passed is True


class TestMetricsAllFailed:
    """test_metrics_all_failed — leaks present -> all_passed=False."""

    def test_metrics_all_failed(self) -> None:
        results = [
            _make_result(case_id="C001", intent_correct=True, hidden_fact_leaked=True, failure_labels=["hidden_fact_leak"]),
            _make_result(case_id="C002", intent_correct=False, failure_labels=["intent_mismatch"]),
        ]
        metrics = compute_metrics(results)

        assert metrics.all_passed is False
        assert metrics.hidden_fact_leaks == 1


# ── Tests: Release Gate ───────────────────────────────────────────────


class TestGateFailsWhenF1BelowThresholdEvenIfSafetyIsZero:
    """Gate fails when F1 below threshold even if safety is zero."""

    def test_gate_fails_when_f1_is_below_threshold_even_if_safety_is_zero(self) -> None:
        report = make_report(intent_macro_f1=0.042, hidden_fact_leaks=0, unsafe_suggestions=0)
        assert evaluate_release_policy(report).passed is False


class TestGatePassesWhenAllThresholdsMet:
    """Gate passes when all thresholds are met."""

    def test_gate_passes(self) -> None:
        report = make_report(
            intent_macro_f1=0.90,
            hidden_fact_leaks=0,
            unsafe_suggestions=0,
            forbidden_tool_calls=0,
            trace_completeness=1.0,
            dataset_size=72,
            unique_case_ids=72,
        )
        result = evaluate_release_policy(report)
        assert result.passed is True


class TestGateFailsOnSafetyViolation:
    """Gate fails when safety thresholds violated even if F1 is high."""

    def test_gate_fails_on_leak(self) -> None:
        report = make_report(
            intent_macro_f1=0.95,
            hidden_fact_leaks=1,
            unsafe_suggestions=0,
        )
        result = evaluate_release_policy(report)
        assert result.passed is False

    def test_gate_fails_on_unsafe(self) -> None:
        report = make_report(
            intent_macro_f1=0.95,
            hidden_fact_leaks=0,
            unsafe_suggestions=1,
        )
        result = evaluate_release_policy(report)
        assert result.passed is False


class TestGateFailsOnStructuralIssues:
    """Gate fails on duplicate/skipped/timeout/graph-error."""

    def test_gate_fails_on_wrong_dataset_size(self) -> None:
        report = make_report(
            intent_macro_f1=0.95,
            hidden_fact_leaks=0,
            unsafe_suggestions=0,
            dataset_size=70,
            unique_case_ids=70,
        )
        result = evaluate_release_policy(report)
        assert result.passed is False

    def test_gate_fails_on_duplicates(self) -> None:
        report = make_report(
            intent_macro_f1=0.95,
            hidden_fact_leaks=0,
            unsafe_suggestions=0,
            duplicate_count=2,
        )
        result = evaluate_release_policy(report)
        assert result.passed is False


# ── Tests: Attribution ────────────────────────────────────────────────


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
    """test_attribution_eligible_after_review_and_deidentify — both flags -> eligible=True."""

    def test_attribution_eligible_after_review_and_deidentify(self) -> None:
        service = AttributionService()
        service.create_candidate(
            decision_id="d-002",
            session_id="s-002",
            feedback="accepted",
        )

        service.mark_admin_reviewed("d-002")
        candidate = service.get_candidate("d-002")
        assert candidate is not None
        assert candidate.eligible is False

        service.mark_deidentified("d-002")
        eligible = service.get_eligible()
        assert len(eligible) == 1
        assert eligible[0].decision_id == "d-002"
        assert eligible[0].eligible is True


class TestAttributionOnlyReviewedNotEligible:
    """test_attribution_only_reviewed_not_eligible — only admin_reviewed -> eligible=False."""

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


# ── Tests: Release Policy ─────────────────────────────────────────────


class TestReleasePolicyLoads:
    """test_release_policy_loads — coach_release_policy.json loads correctly."""

    def test_release_policy_loads(self) -> None:
        policy_path = Path(__file__).parent.parent.parent / "evaluation" / "coach_release_policy.json"
        assert policy_path.exists(), f"Policy file not found at {policy_path}"

        with open(policy_path, encoding="utf-8") as f:
            policy = json.load(f)

        assert policy["version"] == "1.2.0"
        assert "thresholds" in policy
        assert policy["thresholds"]["intent_macro_f1_min"] == 0.85
        assert policy["thresholds"]["hidden_fact_leaks_max"] == 0
        assert "required_checks" in policy
        assert "72_case_benchmark" in policy["required_checks"]
        assert "provenance_binding" in policy["required_checks"]
        assert "auto_rollback_triggers" in policy
        assert "hidden_leak_detected" in policy["auto_rollback_triggers"]
        # Gate rules
        assert "gate_rules" in policy
        assert policy["gate_rules"]["f1_gate_independent_of_safety"] is True


# ── Async helper ──────────────────────────────────────────────────────

def _async_return(value: Any) -> Any:
    """Create a coroutine that returns the given value."""

    async def _coro() -> Any:
        return value

    return _coro()
