"""evaluation_service 重构测试 — 新图路径与辅助函数"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.orchestration.state import (
    DimensionResult,
    AgentResultEnvelope,
    SafetyResult,
)


# ── _parse_symptoms ──────────────────────────────────────────────────────────

class TestParseSymptoms:
    """测试 _parse_symptoms 解析各种输入"""

    def test_empty_string(self):
        from app.services.evaluation_service import _parse_symptoms
        assert _parse_symptoms("") == []

    def test_none(self):
        from app.services.evaluation_service import _parse_symptoms
        assert _parse_symptoms(None) == []

    def test_json_array(self):
        from app.services.evaluation_service import _parse_symptoms
        result = _parse_symptoms('["发热", "咳嗽", "乏力"]')
        assert result == ["发热", "咳嗽", "乏力"]

    def test_json_empty_array(self):
        from app.services.evaluation_service import _parse_symptoms
        assert _parse_symptoms("[]") == []

    def test_plain_string(self):
        from app.services.evaluation_service import _parse_symptoms
        result = _parse_symptoms("发热,咳嗽")
        assert result == ["发热,咳嗽"]

    def test_invalid_json_fallback(self):
        from app.services.evaluation_service import _parse_symptoms
        result = _parse_symptoms("{not valid json")
        assert result == ["{not valid json"]

    def test_json_non_list(self):
        """JSON 解析成功但不是 list 类型，返回原始字符串"""
        from app.services.evaluation_service import _parse_symptoms
        result = _parse_symptoms('"just a string"')
        assert result == ['"just a string"']


# ── _build_evaluation_from_state ─────────────────────────────────────────────

class TestBuildEvaluationFromState:
    """测试从 LangGraph final state 构建 Evaluation ORM 对象"""

    def _make_state(self, **overrides):
        """构建一个最小可用的 final state dict"""
        base = {
            "run_id": "test-run-001",
            "graph_version": "evaluation-graph-v1",
            "scoring_policy_version": "v1",
            "dimension_results": {
                "inquiry": DimensionResult(
                    dimension="inquiry", status="scored", score=85.0, analysis="问诊良好"
                ),
                "knowledge": DimensionResult(
                    dimension="knowledge", status="scored", score=72.0, analysis="知识掌握一般"
                ),
                "humanistic": DimensionResult(
                    dimension="humanistic", status="scored", score=90.0, analysis="人文关怀优秀"
                ),
                "diagnosis": DimensionResult(
                    dimension="diagnosis", status="scored", score=78.0, analysis="诊断基本准确"
                ),
                "treatment": DimensionResult(
                    dimension="treatment", status="scored", score=80.0, analysis="治疗方案合理"
                ),
            },
            "agent_results": [],
            "total_score": 81.0,
            "overall_summary": "综合表现良好",
            "improvement_suggestions": ["加强知识学习", "提升沟通技巧"],
            "evaluation_status": "completed",
            "human_review_needed": False,
            "review_reason": None,
            "safety_result": SafetyResult(
                risk_level="low",
                reasoning_summary="无安全风险",
            ),
        }
        base.update(overrides)
        return base

    def test_basic_construction(self):
        """基本字段正确映射"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state()
        evaluation = _build_evaluation_from_state(state, consultation_id=42)

        assert evaluation.consultation_id == 42
        assert evaluation.inquiry_score == 85.0
        assert evaluation.knowledge_score == 72.0
        assert evaluation.humanistic_score == 90.0
        assert evaluation.diagnosis_score == 78.0
        assert evaluation.treatment_score == 80.0
        assert evaluation.total_score == 81.0
        assert evaluation.overall_summary == "综合表现良好"
        assert evaluation.evaluation_status == "completed"
        assert evaluation.human_review_needed is False
        assert evaluation.run_id == "test-run-001"
        assert evaluation.graph_version == "evaluation-graph-v1"
        assert evaluation.scoring_policy_version == "v1"

    def test_improvement_suggestions_joined(self):
        """改进建议列表拼接为文本"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state()
        evaluation = _build_evaluation_from_state(state, consultation_id=1)

        assert "加强知识学习" in evaluation.improvement_suggestions
        assert "提升沟通技巧" in evaluation.improvement_suggestions

    def test_safety_data_serialized(self):
        """safety_result 被序列化为 dict"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state()
        evaluation = _build_evaluation_from_state(state, consultation_id=1)

        assert evaluation.safety_data is not None
        assert evaluation.safety_data["risk_level"] == "low"

    def test_no_safety_result(self):
        """safety_result 为 None 时 safety_data 为 None"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state(safety_result=None)
        evaluation = _build_evaluation_from_state(state, consultation_id=1)
        assert evaluation.safety_data is None

    def test_dimension_not_scored_gives_zero(self):
        """未评分维度分数为 0"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state(
            dimension_results={
                "inquiry": DimensionResult(
                    dimension="inquiry", status="not_submitted", score=None, analysis="未提交"
                ),
            }
        )
        evaluation = _build_evaluation_from_state(state, consultation_id=1)
        assert evaluation.inquiry_score == 0
        assert evaluation.inquiry_analysis == "未提交"

    def test_rag_fields_from_knowledge_agent(self):
        """knowledge agent 的 citations/trace 正确提取"""
        from app.services.evaluation_service import _build_evaluation_from_state

        citations = [{"text": "参考1", "source": "指南.pdf"}]
        trace = {"retrieved_chunks": 5}
        state = self._make_state(
            agent_results=[
                AgentResultEnvelope(
                    agent_name="knowledge",
                    status="success",
                    score=72.0,
                    analysis="知识核对",
                    citations=citations,
                    trace=trace,
                ),
            ]
        )
        evaluation = _build_evaluation_from_state(state, consultation_id=1)
        assert evaluation.citation_data == citations
        assert evaluation.rag_trace_data == trace
        assert evaluation.retrieval_status == "not_run"

    def test_rag_insufficient_knowledge(self):
        """knowledge agent insufficient 时设置拒答字段"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state(
            agent_results=[
                AgentResultEnvelope(
                    agent_name="knowledge",
                    status="insufficient",
                    score=None,
                    analysis="证据不足",
                ),
            ]
        )
        evaluation = _build_evaluation_from_state(state, consultation_id=1)
        assert evaluation.retrieval_status == "insufficient"
        assert evaluation.evidence_stance == "refusal"

    def test_applicable_dimensions_keys(self):
        """applicable_dimensions 为维度键列表"""
        from app.services.evaluation_service import _build_evaluation_from_state

        state = self._make_state()
        evaluation = _build_evaluation_from_state(state, consultation_id=1)
        assert set(evaluation.applicable_dimensions) == {
            "inquiry", "knowledge", "humanistic", "diagnosis", "treatment"
        }

    def test_empty_state_defaults(self):
        """空 state 使用默认值"""
        from app.services.evaluation_service import _build_evaluation_from_state

        evaluation = _build_evaluation_from_state({}, consultation_id=99)
        assert evaluation.consultation_id == 99
        assert evaluation.inquiry_score == 0
        assert evaluation.total_score is None
        assert evaluation.evaluation_status == "completed"  # default


# ── run_evaluation 分发 ──────────────────────────────────────────────────────

class TestRunEvaluationDispatch:
    """测试 run_evaluation 根据 LANGGRAPH_ENABLED 选择路径"""

    @pytest.mark.asyncio
    async def test_dispatch_to_graph_when_enabled(self):
        """LANGGRAPH_ENABLED=True 时调用 _run_evaluation_graph"""
        from app.services.evaluation_service import run_evaluation

        mock_db = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.LANGGRAPH_ENABLED = True
        with patch("app.services.evaluation_service.settings", mock_settings):
            with patch(
                "app.services.evaluation_service._run_evaluation_graph",
                new_callable=AsyncMock,
                return_value="graph_result",
            ) as mock_graph:
                result = await run_evaluation(mock_db, 1)
                mock_graph.assert_awaited_once_with(mock_db, 1)
                assert result == "graph_result"

    @pytest.mark.asyncio
    async def test_dispatch_to_legacy_when_disabled(self):
        """LANGGRAPH_ENABLED=False 时调用 _run_evaluation_legacy"""
        from app.services.evaluation_service import run_evaluation

        mock_db = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.LANGGRAPH_ENABLED = False
        with patch("app.services.evaluation_service.settings", mock_settings):
            with patch(
                "app.services.evaluation_service._run_evaluation_legacy",
                new_callable=AsyncMock,
                return_value="legacy_result",
            ) as mock_legacy:
                result = await run_evaluation(mock_db, 1)
                mock_legacy.assert_awaited_once_with(mock_db, 1)
                assert result == "legacy_result"
