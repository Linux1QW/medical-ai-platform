"""场景分类与动态路由测试"""

import pytest
from unittest.mock import MagicMock

from app.orchestration.routes import (
    build_submission_flags,
    build_route_plan,
    get_consultation_type,
)
from app.orchestration.state import SubmissionFlags, RoutePlan


def _make_consultation(diagnosis="", treatment_plan="", consultation_type=None):
    """创建模拟 Consultation 对象"""
    c = MagicMock()
    c.diagnosis = diagnosis
    c.treatment_plan = treatment_plan
    if consultation_type is not None:
        c.consultation_type = consultation_type
    return c


# ── build_submission_flags 测试 ─────────────────────────────────────────────

class TestBuildSubmissionFlags:
    def test_both_submitted(self):
        """诊断和治疗都已提交"""
        c = _make_consultation(
            diagnosis="急性支气管炎",
            treatment_plan="阿莫西林 0.5g tid"
        )
        flags = build_submission_flags(c)
        assert flags.has_diagnosis is True
        assert flags.has_treatment is True

    def test_only_diagnosis_submitted(self):
        """只提交诊断"""
        c = _make_consultation(
            diagnosis="急性支气管炎",
            treatment_plan=""
        )
        flags = build_submission_flags(c)
        assert flags.has_diagnosis is True
        assert flags.has_treatment is False

    def test_neither_submitted(self):
        """都不提交"""
        c = _make_consultation(diagnosis="", treatment_plan="")
        flags = build_submission_flags(c)
        assert flags.has_diagnosis is False
        assert flags.has_treatment is False

    def test_empty_string_vs_placeholder(self):
        """空字符串 vs 占位字符串：空字符串视为未提交，占位字符串视为已提交"""
        # 纯空格
        c1 = _make_consultation(diagnosis="   ", treatment_plan="   ")
        flags1 = build_submission_flags(c1)
        assert flags1.has_diagnosis is False
        assert flags1.has_treatment is False

        # 占位字符串（如 "暂无"）
        c2 = _make_consultation(diagnosis="暂无", treatment_plan="待补充")
        flags2 = build_submission_flags(c2)
        assert flags2.has_diagnosis is True
        assert flags2.has_treatment is True


# ── build_route_plan 测试 ───────────────────────────────────────────────────

class TestBuildRoutePlan:
    def test_initial_all_submitted(self):
        """initial + 全部提交 → selected 包含全部5个"""
        flags = SubmissionFlags(has_diagnosis=True, has_treatment=True)
        plan = build_route_plan("initial", flags)
        assert plan.consultation_type == "initial"
        assert set(plan.selected_agents) == {"inquiry", "humanistic", "diagnosis", "treatment", "knowledge"}
        assert len(plan.skipped_agents) == 0

    def test_initial_no_diagnosis(self):
        """initial + 未提交诊断 → diagnosis 在 skipped 中"""
        flags = SubmissionFlags(has_diagnosis=False, has_treatment=True)
        plan = build_route_plan("initial", flags)
        assert "diagnosis" in plan.skipped_agents
        assert "diagnosis" not in plan.selected_agents
        assert plan.skip_reasons["diagnosis"] == "未提交诊断结果"
        assert "treatment" in plan.selected_agents

    def test_initial_none_submitted(self):
        """initial + 都不提交 → diagnosis/treatment 跳过，knowledge 仍选中"""
        flags = SubmissionFlags(has_diagnosis=False, has_treatment=False)
        plan = build_route_plan("initial", flags)
        assert set(plan.selected_agents) == {"inquiry", "humanistic", "knowledge"}
        assert set(plan.skipped_agents) == {"diagnosis", "treatment"}

    def test_communication_no_knowledge(self):
        """communication → knowledge 不在 conditional 中"""
        flags = SubmissionFlags(has_diagnosis=True, has_treatment=True)
        plan = build_route_plan("communication", flags)
        assert plan.consultation_type == "communication"
        assert "knowledge" not in plan.selected_agents
        assert "knowledge" not in plan.skipped_agents
        assert set(plan.selected_agents) == {"inquiry", "humanistic"}

    def test_unknown_type_fallback(self):
        """未知类型 → 回退到 initial"""
        flags = SubmissionFlags(has_diagnosis=True, has_treatment=True)
        plan = build_route_plan("unknown_type", flags)
        assert plan.consultation_type == "initial"
        assert set(plan.selected_agents) == {"inquiry", "humanistic", "diagnosis", "treatment", "knowledge"}


# ── get_consultation_type 测试 ──────────────────────────────────────────────

class TestGetConsultationType:
    def test_has_value_returns_value(self):
        """有值返回值"""
        c = _make_consultation(consultation_type="follow_up")
        ct = get_consultation_type(c)
        assert ct == "follow_up"

    def test_no_attribute_returns_initial(self):
        """无属性返回 initial"""
        c = _make_consultation()
        # 删除 consultation_type 属性模拟旧数据
        delattr(c, "consultation_type")
        ct = get_consultation_type(c)
        assert ct == "initial"

    def test_empty_string_returns_initial(self):
        """空字符串返回 initial"""
        c = _make_consultation(consultation_type="")
        ct = get_consultation_type(c)
        assert ct == "initial"

    def test_whitespace_returns_initial(self):
        """纯空白返回 initial"""
        c = _make_consultation(consultation_type="   ")
        ct = get_consultation_type(c)
        assert ct == "initial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
