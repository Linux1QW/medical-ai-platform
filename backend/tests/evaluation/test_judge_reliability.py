# -*- coding: utf-8 -*-
"""Judge 稳定性和人工校准测试 — Task 4

验证：
1. 重复评分一致性（repeat_agreement）
2. 位置一致性（position_consistency）
3. judge 降级调用被记录
4. 人工标签缺失时不伪造 human_agreement
5. needs_review 触发条件
"""
import pytest
from pydantic import ValidationError

from evaluation.judge_reliability import (
    JudgeReliability,
    JudgeRun,
    evaluate_judge_reliability,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _run(score: float = 80.0, position: str = "original", **kwargs) -> JudgeRun:
    return JudgeRun(
        case_id="patient_001",
        dimension="inquiry",
        score=score,
        position=position,
        judge_version="judge_v1",
        model_family="qwen",
        seed=42,
        **kwargs,
    )


# ── 1. JudgeRun 模型 ────────────────────────────────────────────────────────


class TestJudgeRun:
    def test_valid_run(self):
        run = _run()
        assert run.score == 80.0
        assert run.position == "original"

    def test_degraded_flag(self):
        run = _run(degraded=True)
        assert run.degraded is True


# ── 2. 重复一致性 ───────────────────────────────────────────────────────────


class TestRepeatAgreement:
    def test_identical_runs_high_agreement(self):
        """两次完全相同的评分 → 一致率 1.0"""
        runs = [_run(score=80.0), _run(score=80.0)]
        rel = evaluate_judge_reliability(runs)
        assert rel.repeat_agreement == 1.0

    def test_different_runs_lower_agreement(self):
        """两次不同评分 → 一致率 < 1.0"""
        runs = [_run(score=80.0), _run(score=60.0)]
        rel = evaluate_judge_reliability(runs)
        assert rel.repeat_agreement < 1.0

    def test_score_std_computed(self):
        """score_std 正确计算（样本标准差）"""
        runs = [_run(score=80.0), _run(score=60.0)]
        rel = evaluate_judge_reliability(runs)
        # stdev([80, 60]) = sqrt(200) ≈ 14.14
        assert rel.score_std == pytest.approx(14.14, abs=0.1)


# ── 3. 位置一致性 ───────────────────────────────────────────────────────────


class TestPositionConsistency:
    def test_same_score_different_position(self):
        """同分不同位 → 位置一致率 1.0"""
        runs = [_run(score=80.0, position="original"), _run(score=80.0, position="swapped")]
        rel = evaluate_judge_reliability(runs)
        assert rel.position_consistency == 1.0

    def test_different_score_different_position(self):
        """不同分不同位 → 位置一致率 < 1.0"""
        runs = [_run(score=80.0, position="original"), _run(score=60.0, position="swapped")]
        rel = evaluate_judge_reliability(runs)
        assert rel.position_consistency < 1.0


# ── 4. 人工标签缺失 ─────────────────────────────────────────────────────────


class TestHumanAgreement:
    def test_no_human_labels_no_agreement(self):
        """无人工标签 → human_agreement 为 None，不伪造"""
        runs = [_run(), _run()]
        rel = evaluate_judge_reliability(runs, human_labels=None)
        assert rel.human_agreement is None

    def test_with_human_labels_computed(self):
        """有人工标签 → human_agreement 有值"""
        runs = [_run(score=80.0), _run(score=75.0)]
        rel = evaluate_judge_reliability(runs, human_labels=[82.0, 78.0])
        assert rel.human_agreement is not None
        assert 0 <= rel.human_agreement <= 1.0


# ── 5. needs_review 触发 ────────────────────────────────────────────────────


class TestNeedsReview:
    def test_low_repeat_agreement_triggers_review(self):
        """一致率过低 → needs_review"""
        runs = [_run(score=90.0), _run(score=30.0)]
        rel = evaluate_judge_reliability(runs, threshold=0.5)
        assert rel.needs_review is True

    def test_high_agreement_no_review(self):
        """一致率高 → 不需要 review"""
        runs = [_run(score=80.0), _run(score=80.0)]
        rel = evaluate_judge_reliability(runs, threshold=0.5)
        assert rel.needs_review is False

    def test_degraded_call_recorded(self):
        """降级调用被记录"""
        runs = [_run(degraded=True), _run(degraded=False)]
        rel = evaluate_judge_reliability(runs)
        assert rel.degraded_count == 1
