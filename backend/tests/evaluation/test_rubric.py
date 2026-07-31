# -*- coding: utf-8 -*-
"""五维原子 Rubric 测试 — Task 3

验证：
1. RubricItem verdict 枚举完整
2. unassessed 不变成 0 分
3. high severity item 自动触发 review_required
4. partial 分数计算可重复
5. 缺少必需 item 时维度为 insufficient
6. rubric v1 JSON 加载合法
7. AgentResultEnvelope 兼容 rubric_items 字段
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.rubric import (
    RubricItem,
    RubricSet,
    RubricVerdict,
    aggregate_rubric,
    load_rubric_v1,
    validate_rubric_items,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _item(
    item_id: str = "inq_01",
    dimension: str = "inquiry",
    verdict: str = "pass",
    score: float = 10.0,
    severity: str = "normal",
    **kwargs,
) -> RubricItem:
    return RubricItem(
        item_id=item_id,
        dimension=dimension,
        verdict=verdict,
        score=score,
        severity=severity,
        description="测试项",
        **kwargs,
    )


# ── 1. RubricVerdict 枚举 ───────────────────────────────────────────────────


class TestRubricVerdict:
    def test_all_verdicts_defined(self):
        assert RubricVerdict.PASS == "pass"
        assert RubricVerdict.PARTIAL == "partial"
        assert RubricVerdict.FAIL == "fail"
        assert RubricVerdict.NOT_APPLICABLE == "not_applicable"
        assert RubricVerdict.UNASSESSED == "unassessed"

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            RubricItem(
                item_id="x", dimension="inquiry", verdict="bogus",
                score=0, description="bad",
            )


# ── 2. unassessed 不变成 0 ──────────────────────────────────────────────────


class TestUnassessedSemantics:
    def test_unassessed_score_is_none(self):
        """unassessed 的 score 必须为 None，不得为 0"""
        item = _item(verdict="unassessed", score=None)
        assert item.score is None

    def test_unassessed_excluded_from_aggregation(self):
        """聚合时 unassessed 不参与计算，维度不降为 0"""
        items = [
            _item(item_id="inq_01", verdict="pass", score=10.0),
            _item(item_id="inq_02", verdict="unassessed", score=None),
        ]
        result = aggregate_rubric(items, dimension="inquiry")
        # 只有 1 项参与，均分 = 10.0
        assert result.score == 10.0
        assert result.status == "scored"

    def test_all_unassessed_is_insufficient(self):
        """全部 unassessed → 维度 insufficient"""
        items = [
            _item(item_id="inq_01", verdict="unassessed", score=None),
            _item(item_id="inq_02", verdict="unassessed", score=None),
        ]
        result = aggregate_rubric(items, dimension="inquiry")
        assert result.status == "insufficient"
        assert result.score is None


# ── 3. high severity 自动触发 review ────────────────────────────────────────


class TestHighSeverityReview:
    def test_high_severity_fail_triggers_review(self):
        """high severity + fail → review_required = True"""
        items = [_item(item_id="inq_01", verdict="fail", score=0, severity="high")]
        result = aggregate_rubric(items, dimension="inquiry")
        assert result.review_required is True

    def test_normal_severity_fail_no_auto_review(self):
        """normal severity + fail 不自动触发 review"""
        items = [_item(item_id="inq_01", verdict="fail", score=0, severity="normal")]
        result = aggregate_rubric(items, dimension="inquiry")
        assert result.review_required is False

    def test_high_severity_pass_no_review(self):
        """high severity + pass 不触发 review"""
        items = [_item(item_id="inq_01", verdict="pass", score=10, severity="high")]
        result = aggregate_rubric(items, dimension="inquiry")
        assert result.review_required is False


# ── 4. partial 分数可重复 ───────────────────────────────────────────────────


class TestPartialScoring:
    def test_partial_score_used(self):
        """partial verdict 使用指定 score"""
        items = [_item(item_id="inq_01", verdict="partial", score=5.0)]
        result = aggregate_rubric(items, dimension="inquiry")
        assert result.score == 5.0

    def test_mixed_pass_partial_fail(self):
        """pass + partial + fail 聚合"""
        items = [
            _item(item_id="a", verdict="pass", score=10.0),
            _item(item_id="b", verdict="partial", score=5.0),
            _item(item_id="c", verdict="fail", score=0.0),
        ]
        result = aggregate_rubric(items, dimension="inquiry")
        # 均分 = (10 + 5 + 0) / 3
        assert abs(result.score - 5.0) < 0.01

    def test_aggregation_deterministic(self):
        """相同输入多次聚合结果一致"""
        items = [
            _item(item_id="a", verdict="pass", score=10.0),
            _item(item_id="b", verdict="partial", score=5.0),
        ]
        r1 = aggregate_rubric(items, dimension="inquiry")
        r2 = aggregate_rubric(items, dimension="inquiry")
        assert r1.score == r2.score


# ── 5. 缺少必需 item → insufficient ────────────────────────────────────────


class TestMissingItems:
    def test_empty_items_insufficient(self):
        """空 item 列表 → insufficient"""
        result = aggregate_rubric([], dimension="inquiry")
        assert result.status == "insufficient"
        assert result.score is None


# ── 6. rubric v1 JSON 加载 ──────────────────────────────────────────────────


class TestRubricV1:
    def test_load_rubric_v1(self):
        """v1 JSON 可加载且包含五维"""
        rubric = load_rubric_v1()
        assert isinstance(rubric, RubricSet)
        assert len(rubric.dimensions) == 5

    def test_all_five_dimensions_present(self):
        rubric = load_rubric_v1()
        expected = {"inquiry", "knowledge", "humanistic", "diagnosis", "treatment"}
        actual = set(rubric.dimensions.keys())
        assert actual == expected

    def test_each_dimension_has_items(self):
        rubric = load_rubric_v1()
        for dim, items in rubric.dimensions.items():
            assert len(items) >= 5, f"维度 {dim} 至少需要 5 个 rubric item"

    def test_all_items_have_required_fields(self):
        rubric = load_rubric_v1()
        for dim, items in rubric.dimensions.items():
            for item in items:
                assert item.item_id
                assert item.dimension == dim
                assert item.description


# ── 7. validate_rubric_items ────────────────────────────────────────────────


class TestValidateRubricItems:
    def test_valid_items_no_errors(self):
        items = [_item(item_id="inq_01"), _item(item_id="inq_02")]
        errors = validate_rubric_items(items)
        assert errors == []

    def test_duplicate_item_id_flagged(self):
        items = [_item(item_id="inq_01"), _item(item_id="inq_01")]
        errors = validate_rubric_items(items)
        assert any("duplicate" in e.lower() or "重复" in e for e in errors)

    def test_unassessed_with_score_flagged(self):
        """unassessed 但带了 score → 警告"""
        items = [_item(item_id="inq_01", verdict="unassessed", score=5.0)]
        errors = validate_rubric_items(items)
        assert any("unassessed" in e.lower() or "score" in e.lower() for e in errors)


# ── 8. AgentResultEnvelope 兼容 ─────────────────────────────────────────────


class TestEnvelopeCompat:
    def test_envelope_accepts_rubric_items(self):
        """AgentResultEnvelope 可携带 rubric_items"""
        from app.orchestration.state import AgentResultEnvelope
        items = [_item(item_id="inq_01").model_dump()]
        envelope = AgentResultEnvelope(
            agent_name="inquiry",
            trace={"rubric_items": items},
        )
        assert "rubric_items" in envelope.trace

    def test_envelope_default_trace_empty(self):
        from app.orchestration.state import AgentResultEnvelope
        envelope = AgentResultEnvelope(agent_name="inquiry")
        assert envelope.trace == {}
