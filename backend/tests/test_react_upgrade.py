# -*- coding: utf-8 -*-
"""ReAct 技术升级测试 — Knowledge Agent ReAct + Reflection Agent"""

from unittest.mock import MagicMock, patch

import pytest

from app.orchestration.state import (
    ReflectionIssue,
    ReflectionResult,
)
from app.services.tools.base import ToolContext
from app.services.tools.consistency import (
    CheckEvidenceSufficiency,
    CheckScoreConsistency,
    DetectScoreContradictions,
    SummarizeEvaluation,
    register_consistency_tools,
)
from app.services.tools.registry import ToolRegistry

# ── 一致性检查工具测试 ─────────────────────────────────────────────────────────


@pytest.fixture
def tool_context():
    return ToolContext(
        run_id="test-run",
        agent_name="test",
        budgets={},
        allowed_citation_ids=set(),
        evidence_cache={},
    )


@pytest.fixture
def sample_dimension_scores():
    return [
        {"dimension": "inquiry", "score": 85, "status": "scored", "analysis": "病史采集良好"},
        {"dimension": "diagnosis", "score": 80, "status": "scored", "analysis": "诊断基本正确"},
        {"dimension": "treatment", "score": 75, "status": "scored", "analysis": "治疗方案合理"},
        {"dimension": "knowledge", "score": 70, "status": "scored", "analysis": "知识核对通过"},
        {"dimension": "humanistic", "score": 90, "status": "scored", "analysis": "沟通良好"},
    ]


class TestCheckScoreConsistency:
    @pytest.mark.asyncio
    async def test_consistent_scores(self, tool_context, sample_dimension_scores):
        tool = CheckScoreConsistency()
        result = await tool.execute(
            MagicMock(dimension_scores=sample_dimension_scores, threshold=0.3),
            tool_context,
        )
        # 分数范围 70-90，差异 20，相对差异 0.2 < 0.3 阈值
        assert result["consistent"] is True
        assert len(result["inconsistencies"]) == 0

    @pytest.mark.asyncio
    async def test_inconsistent_scores(self, tool_context):
        scores = [
            {"dimension": "inquiry", "score": 95, "status": "scored", "analysis": ""},
            {"dimension": "knowledge", "score": 30, "status": "scored", "analysis": ""},
        ]
        tool = CheckScoreConsistency()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, threshold=0.3),
            tool_context,
        )
        assert result["consistent"] is False
        assert len(result["inconsistencies"]) > 0

    @pytest.mark.asyncio
    async def test_insufficient_dimensions(self, tool_context):
        scores = [{"dimension": "inquiry", "score": 80, "status": "scored", "analysis": ""}]
        tool = CheckScoreConsistency()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, threshold=0.3),
            tool_context,
        )
        assert result["consistent"] is True


class TestCheckEvidenceSufficiency:
    @pytest.mark.asyncio
    async def test_all_sufficient(self, tool_context, sample_dimension_scores):
        tool = CheckEvidenceSufficiency()
        result = await tool.execute(
            MagicMock(dimension_scores=sample_dimension_scores, min_score_threshold=60.0),
            tool_context,
        )
        assert result["overall_sufficient"] is True
        assert len(result["insufficient_dimensions"]) == 0

    @pytest.mark.asyncio
    async def test_low_score_detected(self, tool_context):
        scores = [
            {"dimension": "knowledge", "score": 45, "status": "scored", "analysis": "证据不足"},
            {"dimension": "inquiry", "score": 85, "status": "scored", "analysis": "良好"},
        ]
        tool = CheckEvidenceSufficiency()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, min_score_threshold=60.0),
            tool_context,
        )
        assert result["overall_sufficient"] is False
        assert len(result["insufficient_dimensions"]) == 1

    @pytest.mark.asyncio
    async def test_error_dimension(self, tool_context):
        scores = [
            {"dimension": "diagnosis", "score": None, "status": "error", "analysis": "评估失败"},
        ]
        tool = CheckEvidenceSufficiency()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, min_score_threshold=60.0),
            tool_context,
        )
        assert result["overall_sufficient"] is False
        assert len(result["error_dimensions"]) == 1


class TestDetectScoreContradictions:
    @pytest.mark.asyncio
    async def test_no_contradiction(self, tool_context, sample_dimension_scores):
        tool = DetectScoreContradictions()
        result = await tool.execute(
            MagicMock(dimension_scores=sample_dimension_scores, contradiction_rules=[]),
            tool_context,
        )
        # 分数差异在合理范围内
        assert result["has_contradictions"] is False

    @pytest.mark.asyncio
    async def test_diagnosis_knowledge_contradiction(self, tool_context):
        scores = [
            {"dimension": "diagnosis", "score": 95, "status": "scored", "analysis": "诊断正确"},
            {"dimension": "knowledge", "score": 30, "status": "scored", "analysis": "缺乏循证依据"},
        ]
        tool = DetectScoreContradictions()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, contradiction_rules=[]),
            tool_context,
        )
        assert result["has_contradictions"] is True
        # 应检测到 diagnosis 高分 vs knowledge 低分的矛盾
        contradictions = result["contradictions"]
        assert any(c["dim_a"] == "diagnosis" and c["dim_b"] == "knowledge" for c in contradictions)


class TestSummarizeEvaluation:
    @pytest.mark.asyncio
    async def test_summary_generation(self, tool_context, sample_dimension_scores):
        tool = SummarizeEvaluation()
        result = await tool.execute(
            MagicMock(
                dimension_scores=sample_dimension_scores,
                total_score=78.0,
                include_recommendations=True,
            ),
            tool_context,
        )
        assert result["provided_total"] == 78.0
        assert len(result["dimensions"]) == 5
        assert result["computed_total"] is not None

    @pytest.mark.asyncio
    async def test_recommendations_for_low_scores(self, tool_context):
        scores = [
            {"dimension": "knowledge", "score": 55, "status": "scored", "analysis": ""},
            {"dimension": "inquiry", "score": 90, "status": "scored", "analysis": ""},
        ]
        tool = SummarizeEvaluation()
        result = await tool.execute(
            MagicMock(dimension_scores=scores, total_score=70.0, include_recommendations=True),
            tool_context,
        )
        assert len(result["recommendations"]) > 0


# ── 工具注册测试 ─────────────────────────────────────────────────────────────


class TestConsistencyToolRegistration:
    def test_register_all(self):
        registry = ToolRegistry()
        registry.reset()
        register_consistency_tools(registry)
        tools = registry.list_tools()
        assert "check_score_consistency" in tools
        assert "check_evidence_sufficiency" in tools
        assert "detect_score_contradictions" in tools
        assert "summarize_evaluation" in tools
        registry.reset()

    def test_idempotent_registration(self):
        registry = ToolRegistry()
        registry.reset()
        register_consistency_tools(registry)
        register_consistency_tools(registry)  # 重复注册不应报错
        assert len(registry.list_tools()) >= 4
        registry.reset()


# ── ReflectionResult 状态模型测试 ────────────────────────────────────────────


class TestReflectionResultModel:
    def test_default_construction(self):
        result = ReflectionResult()
        assert result.overall_quality == "acceptable"
        assert result.confidence == 0.5
        assert result.disabled is False

    def test_disabled_construction(self):
        result = ReflectionResult(disabled=True)
        assert result.disabled is True

    def test_full_construction(self):
        issue = ReflectionIssue(
            issue_type="score_contradiction",
            severity="high",
            description="诊断高分但知识核对低分",
            affected_dimensions=["diagnosis", "knowledge"],
            recommendation="建议复核",
        )
        result = ReflectionResult(
            overall_quality="needs_attention",
            confidence=0.8,
            issues_found=[issue],
            consistency_score=0.4,
            evidence_adequacy_score=0.6,
            summary="发现评分矛盾",
            needs_review=True,
            review_reasons=["诊断与知识核对不一致"],
            react_steps_count=4,
            dimension_count=5,
        )
        assert result.overall_quality == "needs_attention"
        assert len(result.issues_found) == 1
        assert result.issues_found[0].severity == "high"


# ── Knowledge Agent ReAct 解析测试 ──────────────────────────────────────────


class TestReActParsing:
    def test_parse_thought_action(self):
        from app.services.agents.knowledge_agent import _parse_react_step
        text = """Thought: 我需要先检索相关医学证据。
Action: search_medical_kb
Action Input: {"query": "高血压诊断标准", "query_type": "guideline", "top_k": 5}"""
        result = _parse_react_step(text)
        assert result["thought"] != ""
        assert result["action"] == "search_medical_kb"
        assert result["action_input"]["query"] == "高血压诊断标准"
        assert result["is_final"] is False

    def test_parse_final_answer(self):
        from app.services.agents.knowledge_agent import _parse_react_step
        text = """Thought: 基于检索到的证据，我可以做出判断。
Final Answer: {"consistency": "supports", "confidence": 0.85, "analysis": "诊断正确"}"""
        result = _parse_react_step(text)
        assert result["is_final"] is True
        assert "supports" in result["final_answer"]

    def test_parse_invalid_json_action_input(self):
        from app.services.agents.knowledge_agent import _parse_react_step
        text = """Thought: 检索证据
Action: search_medical_kb
Action Input: {"query": "test", "top_k": 5,}"""
        result = _parse_react_step(text)
        # 应能修复尾部逗号
        assert result["action"] == "search_medical_kb"
        assert result["action_input"].get("query") == "test"


# ── Reflection Agent 解析测试 ────────────────────────────────────────────────


class TestReflectionParsing:
    def test_parse_react_step(self):
        from app.services.agents.reflection_agent import _parse_react_step
        text = """Thought: 检查各维度评分一致性
Action: check_score_consistency
Action Input: {"dimension_scores": []}"""
        result = _parse_react_step(text)
        assert result["action"] == "check_score_consistency"
        assert result["is_final"] is False

    def test_parse_final_answer(self):
        from app.services.agents.reflection_agent import _parse_react_step
        text = """Thought: 检查完成
Final Answer: {"overall_quality": "good", "confidence": 0.9}"""
        result = _parse_react_step(text)
        assert result["is_final"] is True


# ── Reflection Agent 降级测试 ────────────────────────────────────────────────


class TestReflectionFallback:
    def test_build_no_data_result(self):
        from app.services.agents.reflection_agent import _build_no_data_result
        result = _build_no_data_result()
        assert result["overall_quality"] == "acceptable"
        assert result["dimension_count"] == 0

    def test_build_fallback_result(self):
        from app.services.agents.reflection_agent import _build_fallback_result
        dim_scores = [
            {"dimension": "knowledge", "score": 45, "status": "scored", "analysis": ""},
            {"dimension": "diagnosis", "score": None, "status": "error", "analysis": "失败"},
        ]
        result = _build_fallback_result(dim_scores, [])
        assert result["overall_quality"] == "needs_attention"
        assert len(result["issues_found"]) == 2

    def test_build_error_result(self):
        from app.services.agents.reflection_agent import _build_error_result
        result = _build_error_result("test error")
        assert result["overall_quality"] == "acceptable"
        assert "test error" in result["summary"]


# ── Graph 集成测试 ──────────────────────────────────────────────────────────


class TestGraphIntegration:
    def test_graph_builds_successfully(self):
        """验证图可以正常构建和编译"""
        from app.orchestration.graph import build_evaluation_graph
        graph = build_evaluation_graph()
        assert graph is not None

    def test_reflection_node_exists(self):
        """验证 reflection_check 节点已添加到图中"""
        from app.orchestration.graph import build_evaluation_graph
        graph = build_evaluation_graph()
        # 检查节点是否存在
        nodes = graph.nodes
        assert "reflection_check" in nodes

    @pytest.mark.asyncio
    async def test_reflection_check_disabled(self):
        """验证反思智能体禁用时正常跳过"""
        from app.orchestration.graph import reflection_check

        state = {
            "dimension_results": {},
            "total_score": None,
        }

        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ENABLE_REACT_REFLECTION = False
            result = await reflection_check(state)
            assert result["reflection_result"].disabled is True
