# -*- coding: utf-8 -*-
"""安全红旗回归集测试 — Task 5

验证：
1. 高危红旗必须进入 needs_review
2. 规则和 LLM 结果冲突时规则优先
3. LLM 超时时 fail closed
4. 普通低风险病例不会全部误报
5. RiskFinding 结构完整
"""
import pytest
from pydantic import ValidationError

from evaluation.safety_cases import (
    RiskFinding,
    RiskType,
    SafetyDecision,
    calculate_safety_metrics,
    evaluate_safety_case,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _case(text: str = "", red_flags: list[str] | None = None, **kwargs) -> dict:
    return {
        "case_id": "patient_001",
        "conversation_text": text,
        "red_flags": red_flags or [],
        **kwargs,
    }


# ── 1. RiskType 枚举 ────────────────────────────────────────────────────────


class TestRiskType:
    def test_all_types_defined(self):
        assert RiskType.EMERGENCY == "emergency"
        assert RiskType.MEDICATION == "medication"
        assert RiskType.POPULATION == "population"
        assert RiskType.PRIVACY == "privacy"
        assert RiskType.EVIDENCE_CONFLICT == "evidence_conflict"


# ── 2. RiskFinding 结构 ─────────────────────────────────────────────────────


class TestRiskFinding:
    def test_valid_finding(self):
        f = RiskFinding(
            risk_type=RiskType.EMERGENCY,
            severity="high",
            description="胸痛伴大汗",
            evidence_span="患者诉持续性胸痛伴大汗2小时",
            source="rule:acute_mi",
            policy_action="immediate_review",
        )
        assert f.severity == "high"
        assert f.policy_action == "immediate_review"

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            RiskFinding(
                risk_type=RiskType.EMERGENCY,
                severity="bogus",
                description="x",
            )


# ── 3. evaluate_safety_case ─────────────────────────────────────────────────


class TestEvaluateSafetyCase:
    def test_high_risk_flag_enters_review(self):
        """高危红旗 → needs_review = True"""
        case = _case(text="患者突发胸痛伴大汗", red_flags=["acute_mi"])
        decision = evaluate_safety_case(case)
        assert decision.needs_review is True
        assert decision.severity in ("high", "critical")

    def test_rule_priority_over_llm(self):
        """规则命中 high 时 LLM 不得降级"""
        case = _case(
            text="胸痛",
            red_flags=["cardiac_arrest"],
        )
        decision = evaluate_safety_case(case, llm_risk="low")
        assert decision.severity == "high"  # 规则优先

    def test_llm_failure_fail_closed(self):
        """LLM 失败 + 无规则命中 → undetermined + needs_review"""
        case = _case(text="头痛")
        decision = evaluate_safety_case(case, llm_failed=True)
        assert decision.needs_review is True
        assert decision.severity == "undetermined"

    def test_low_risk_no_false_positive(self):
        """普通低风险病例不误报为高危"""
        case = _case(text="感冒流涕三天", red_flags=[])
        decision = evaluate_safety_case(case)
        assert decision.severity in ("low", "none")
        assert decision.needs_review is False


# ── 4. calculate_safety_metrics ─────────────────────────────────────────────


class TestSafetyMetrics:
    def test_basic_metrics(self):
        results = [
            evaluate_safety_case(_case(text="胸痛", red_flags=["acute_mi"])),
            evaluate_safety_case(_case(text="感冒", red_flags=[])),
        ]
        metrics = calculate_safety_metrics(results)
        assert "total_cases" in metrics
        assert metrics["total_cases"] == 2
        assert "high_risk_count" in metrics
        assert metrics["high_risk_count"] >= 1

    def test_red_flag_recall_rate(self):
        """关键红旗召回率计算"""
        results = [
            evaluate_safety_case(_case(text="x", red_flags=["acute_mi"])),
            evaluate_safety_case(_case(text="y", red_flags=["cardiac_arrest"])),
        ]
        metrics = calculate_safety_metrics(results)
        assert metrics["red_flag_recall_rate"] == 1.0


# ── 5. SafetyDecision 结构 ──────────────────────────────────────────────────


class TestSafetyDecision:
    def test_decision_has_findings(self):
        case = _case(text="胸痛", red_flags=["acute_mi"])
        decision = evaluate_safety_case(case)
        assert isinstance(decision.findings, list)
        assert len(decision.findings) > 0

    def test_finding_has_evidence_span(self):
        """高危 finding 有 evidence_span"""
        case = _case(text="胸痛伴大汗", red_flags=["acute_mi"])
        decision = evaluate_safety_case(case)
        for f in decision.findings:
            if f.severity == "high":
                assert f.evidence_span
