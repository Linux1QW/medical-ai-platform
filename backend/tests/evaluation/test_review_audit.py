# -*- coding: utf-8 -*-
"""人工复核状态机测试 — Task 6

验证：
1. 状态迁移合法（pending → in_review → approved/rejected/returned）
2. 非法迁移拒绝
3. 高风险 approve 必须填写 reason
4. 原始评估不可变（snapshot 隔离）
5. 复核调整可还原
"""
import pytest

from evaluation.review_audit import (
    ReviewDecision,
    ReviewStatus,
    apply_review_decision,
    create_review_snapshot,
    validate_review_transition,
)

# ── 辅助 ────────────────────────────────────────────────────────────────────


def _evaluation(risk_level: str = "low", scores: dict | None = None) -> dict:
    return {
        "id": 1,
        "consultation_id": 100,
        "risk_level": risk_level,
        "inquiry_score": 80.0,
        "knowledge_score": 75.0,
        "humanistic_score": 85.0,
        "diagnosis_score": 70.0,
        "treatment_score": 78.0,
        "total_score": 77.6,
        **(scores or {}),
    }


def _decision(
    status: str = "approved",
    reason: str = "确认无误",
    adjusted_scores: dict | None = None,
    **kwargs,
) -> ReviewDecision:
    return ReviewDecision(
        reviewer_id="admin_001",
        status=status,
        reason_code="confirmed",
        reason=reason,
        adjusted_scores=adjusted_scores or {},
        **kwargs,
    )


# ── 1. ReviewStatus 枚举 ────────────────────────────────────────────────────


class TestReviewStatus:
    def test_all_statuses(self):
        assert ReviewStatus.PENDING == "pending_review"
        assert ReviewStatus.IN_REVIEW == "in_review"
        assert ReviewStatus.APPROVED == "approved"
        assert ReviewStatus.REJECTED == "rejected"
        assert ReviewStatus.RETURNED == "returned"


# ── 2. 状态迁移 ─────────────────────────────────────────────────────────────


class TestReviewTransition:
    def test_valid_transitions(self):
        """合法迁移"""
        assert validate_review_transition("pending_review", "in_review") is True
        assert validate_review_transition("in_review", "approved") is True
        assert validate_review_transition("in_review", "rejected") is True
        assert validate_review_transition("in_review", "returned") is True

    def test_invalid_transition_rejected(self):
        """非法迁移"""
        assert validate_review_transition("pending_review", "approved") is False
        assert validate_review_transition("approved", "pending_review") is False
        assert validate_review_transition("rejected", "in_review") is False


# ── 3. 高风险 approve 必须 reason ───────────────────────────────────────────


class TestHighRiskApproval:
    def test_high_risk_approve_requires_reason(self):
        """高风险 approve 无 reason → 拒绝"""
        decision = _decision(status="approved", reason="")
        with pytest.raises(ValueError, match="reason"):
            apply_review_decision(
                evaluation=_evaluation(risk_level="high"),
                decision=decision,
            )

    def test_high_risk_approve_with_reason_ok(self):
        """高风险 approve 有 reason → 通过"""
        decision = _decision(status="approved", reason="确认安全")
        result = apply_review_decision(
            evaluation=_evaluation(risk_level="high"),
            decision=decision,
        )
        assert result["final_status"] == "approved"

    def test_low_risk_approve_no_reason_ok(self):
        """低风险 approve 无 reason → 通过"""
        decision = _decision(status="approved", reason="")
        result = apply_review_decision(
            evaluation=_evaluation(risk_level="low"),
            decision=decision,
        )
        assert result["final_status"] == "approved"


# ── 4. 原始评估不可变 ───────────────────────────────────────────────────────


class TestSnapshotIsolation:
    def test_snapshot_preserves_original(self):
        """snapshot 保留原始分数"""
        evaluation = _evaluation()
        snapshot = create_review_snapshot(evaluation)
        assert snapshot["original_scores"]["inquiry_score"] == 80.0
        # 修改原评估不影响 snapshot
        evaluation["inquiry_score"] = 99.0
        assert snapshot["original_scores"]["inquiry_score"] == 80.0

    def test_snapshot_has_all_dimensions(self):
        """snapshot 包含全部五维"""
        snapshot = create_review_snapshot(_evaluation())
        expected = {"inquiry_score", "knowledge_score", "humanistic_score",
                    "diagnosis_score", "treatment_score"}
        assert set(snapshot["original_scores"].keys()) >= expected


# ── 5. 复核调整可还原 ───────────────────────────────────────────────────────


class TestReviewReversibility:
    def test_adjusted_scores_stored(self):
        """调整后分数被保存"""
        decision = _decision(
            status="approved",
            reason="调整",
            adjusted_scores={"inquiry_score": 85.0},
        )
        result = apply_review_decision(
            evaluation=_evaluation(),
            decision=decision,
        )
        assert result["adjusted_scores"]["inquiry_score"] == 85.0

    def test_original_preserved_after_adjustment(self):
        """调整后原始仍保留"""
        decision = _decision(
            status="approved",
            reason="调整",
            adjusted_scores={"inquiry_score": 85.0},
        )
        result = apply_review_decision(
            evaluation=_evaluation(),
            decision=decision,
        )
        assert result["original_scores"]["inquiry_score"] == 80.0
