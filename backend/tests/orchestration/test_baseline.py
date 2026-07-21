# -*- coding: utf-8 -*-
"""基线测试：记录现有评估 API 返回 Schema，作为 LangGraph 重构的兼容门禁

所有测试均不需要 LLM 调用，纯结构/签名验证。
"""

import inspect

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# a) Schema 兼容性测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationModelFields:
    """验证 Evaluation ORM 模型包含所有必要字段"""

    def test_evaluation_model_fields(self):
        """验证 Evaluation 模型包含所有五维分数、总分和 RAG 审计字段"""
        from app.models.evaluation import Evaluation

        mapper = Evaluation.__table__.columns

        # 五维分数字段
        assert "inquiry_score" in mapper, "缺少 inquiry_score 字段"
        assert "inquiry_analysis" in mapper, "缺少 inquiry_analysis 字段"
        assert "knowledge_score" in mapper, "缺少 knowledge_score 字段"
        assert "knowledge_analysis" in mapper, "缺少 knowledge_analysis 字段"
        assert "humanistic_score" in mapper, "缺少 humanistic_score 字段"
        assert "humanistic_analysis" in mapper, "缺少 humanistic_analysis 字段"
        assert "diagnosis_score" in mapper, "缺少 diagnosis_score 字段"
        assert "diagnosis_analysis" in mapper, "缺少 diagnosis_analysis 字段"
        assert "treatment_score" in mapper, "缺少 treatment_score 字段"
        assert "treatment_analysis" in mapper, "缺少 treatment_analysis 字段"

        # 综合评分
        assert "total_score" in mapper, "缺少 total_score 字段"
        assert "overall_summary" in mapper, "缺少 overall_summary 字段"
        assert "improvement_suggestions" in mapper, "缺少 improvement_suggestions 字段"

        # RAG 审计字段
        assert "citation_data" in mapper, "缺少 citation_data 字段"
        assert "retrieval_status" in mapper, "缺少 retrieval_status 字段"
        assert "evidence_stance" in mapper, "缺少 evidence_stance 字段"
        assert "human_review_needed" in mapper, "缺少 human_review_needed 字段"
        assert "review_reason" in mapper, "缺少 review_reason 字段"
        assert "rag_trace_data" in mapper, "缺少 rag_trace_data 字段"

        # 评估状态
        assert "evaluation_status" in mapper, "缺少 evaluation_status 字段"

        # 关联字段
        assert "consultation_id" in mapper, "缺少 consultation_id 字段"
        assert "created_at" in mapper, "缺少 created_at 字段"

    def test_evaluation_status_values(self):
        """验证 evaluation_status 的可能值：completed / needs_review"""
        from app.models.evaluation import Evaluation

        col = Evaluation.__table__.columns["evaluation_status"]
        # 默认值为 'completed'
        assert col.default.arg == "completed", (
            f"evaluation_status 默认值应为 'completed'，实际为 {col.default.arg!r}"
        )
        # 字段长度 20，足以容纳 'needs_review'
        assert col.type.length >= 12, "evaluation_status 字段长度不足"

    def test_null_score_semantics(self):
        """验证 knowledge_score 和 total_score 默认 None，其余维度分数默认 0"""
        from app.models.evaluation import Evaluation

        mapper = Evaluation.__table__.columns

        # knowledge_score 默认 None（RAG 拒答场景）
        # SQLAlchemy: default=None 时 column.default 为 None 对象本身
        assert mapper["knowledge_score"].default is None, (
            "knowledge_score 默认值应为 None（RAG 拒答场景）"
        )
        # total_score 默认 None（所有维度均未评估时）
        assert mapper["total_score"].default is None, (
            "total_score 默认值应为 None"
        )
        # 其余维度分数默认 0（确保即使写入也有有效值）
        for field in ["inquiry_score", "humanistic_score",
                      "diagnosis_score", "treatment_score"]:
            assert mapper[field].default.arg == 0, (
                f"{field} 默认值应为 0，实际为 {mapper[field].default.arg!r}"
            )

    def test_dimension_score_types(self):
        """验证各维度分数的类型约束：均为 Float"""
        from sqlalchemy import Float

        from app.models.evaluation import Evaluation

        mapper = Evaluation.__table__.columns
        score_fields = [
            "inquiry_score", "knowledge_score", "humanistic_score",
            "diagnosis_score", "treatment_score", "total_score",
        ]
        for field in score_fields:
            assert isinstance(mapper[field].type, Float), (
                f"{field} 类型应为 Float，实际为 {type(mapper[field].type).__name__}"
            )

    def test_rag_audit_field_defaults(self):
        """验证 RAG 审计字段的默认值"""
        from app.models.evaluation import Evaluation

        mapper = Evaluation.__table__.columns

        assert mapper["retrieval_status"].default.arg == "not_run"
        assert mapper["evidence_stance"].default.arg == "undetermined"
        assert mapper["human_review_needed"].default.arg is False
        assert mapper["evaluation_status"].default.arg == "completed"

        # nullable 字段
        assert mapper["citation_data"].nullable is True
        assert mapper["review_reason"].nullable is True
        assert mapper["rag_trace_data"].nullable is True


class TestEvaluationOutSchema:
    """验证 EvaluationOut Pydantic Schema 字段"""

    def test_evaluation_out_fields(self):
        """验证 EvaluationOut 包含所有必要字段及正确类型"""
        from app.schemas.evaluation import EvaluationOut

        fields = EvaluationOut.model_fields

        # 五维分数
        assert "inquiry_score" in fields
        assert "knowledge_score" in fields
        assert "humanistic_score" in fields
        assert "diagnosis_score" in fields
        assert "treatment_score" in fields
        assert "total_score" in fields

        # knowledge_score 和 total_score 为 Optional
        assert fields["knowledge_score"].is_required() is False
        assert fields["total_score"].is_required() is False

        # RAG 审计字段
        assert "citation_data" in fields
        assert "retrieval_status" in fields
        assert "evidence_stance" in fields
        assert "human_review_needed" in fields
        assert "review_reason" in fields
        assert "rag_trace_data" in fields
        assert "evaluation_status" in fields

    def test_evaluation_out_defaults(self):
        """验证 EvaluationOut 的默认值"""
        from app.schemas.evaluation import EvaluationOut

        fields = EvaluationOut.model_fields

        assert fields["retrieval_status"].default == "not_run"
        assert fields["evidence_stance"].default == "undetermined"
        assert fields["human_review_needed"].default is False
        assert fields["evaluation_status"].default == "completed"
        assert fields["citation_data"].default is None
        assert fields["review_reason"].default is None
        assert fields["rag_trace_data"].default is None


# ═══════════════════════════════════════════════════════════════════════════════
# b) 评分引擎基线测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoringEngine:
    """评分引擎权重的加权计算基线"""

    def test_scoring_weights(self):
        """验证当前五维权重配置"""
        from app.services.agents.scoring_agent import SCORING_WEIGHTS

        assert SCORING_WEIGHTS == {
            "inquiry": 0.25,
            "knowledge": 0.25,
            "humanistic": 0.20,
            "diagnosis": 0.15,
            "treatment": 0.15,
        }, f"权重配置已变更: {SCORING_WEIGHTS}"

    def test_scoring_weights_sum_to_one(self):
        """验证权重之和为 1.0"""
        from app.services.agents.scoring_agent import SCORING_WEIGHTS

        total = sum(SCORING_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"权重之和应为 1.0，实际为 {total}"

    def test_calculate_total_all_dimensions(self):
        """验证全部维度有分数时的加权计算"""
        from app.services.agents.scoring_agent import calculate_total

        scores = {
            "inquiry": 80,
            "knowledge": 70,
            "humanistic": 75,
            "diagnosis": 65,
            "treatment": 60,
        }
        result = calculate_total(scores)

        # 手动计算验证：
        # (80*0.25 + 70*0.25 + 75*0.20 + 65*0.15 + 60*0.15) / 1.0
        # = 20 + 17.5 + 15 + 9.75 + 9 = 71.25 → round → 71
        assert result is not None
        assert isinstance(result, int), f"返回类型应为 int，实际为 {type(result).__name__}"
        assert 0 <= result <= 100, f"结果应在 0-100 范围内，实际为 {result}"
        assert result == 71, f"加权计算结果应为 71，实际为 {result}"

    def test_calculate_total_with_none(self):
        """验证 None 维度时的权重重分配"""
        from app.services.agents.scoring_agent import calculate_total

        # knowledge=None 时，其余四维权重重分配
        # 有效权重: inquiry=0.25, humanistic=0.20, diagnosis=0.15, treatment=0.15
        # 总有效权重 = 0.75
        # 加权: (80*0.25 + 75*0.20 + 65*0.15 + 60*0.15) / 0.75
        #       = (20 + 15 + 9.75 + 9) / 0.75 = 53.75 / 0.75 = 71.666... → round → 72
        scores = {
            "inquiry": 80,
            "knowledge": None,
            "humanistic": 75,
            "diagnosis": 65,
            "treatment": 60,
        }
        result = calculate_total(scores)

        assert result is not None
        assert isinstance(result, int)
        assert 0 <= result <= 100
        assert result == 72, f"knowledge=None 时加权结果应为 72，实际为 {result}"

    def test_calculate_total_all_none(self):
        """验证全部 None 时返回 None"""
        from app.services.agents.scoring_agent import calculate_total

        scores = {
            "inquiry": None,
            "knowledge": None,
            "humanistic": None,
            "diagnosis": None,
            "treatment": None,
        }
        result = calculate_total(scores)
        assert result is None, f"全部 None 时应返回 None，实际为 {result}"

    def test_calculate_total_single_dimension(self):
        """验证单一维度时的计算"""
        from app.services.agents.scoring_agent import calculate_total

        scores = {
            "inquiry": 80,
            "knowledge": None,
            "humanistic": None,
            "diagnosis": None,
            "treatment": None,
        }
        result = calculate_total(scores)
        # 仅 inquiry 有效，权重归一化后即为 80
        assert result == 80, f"单一维度时应返回该维度分数 80，实际为 {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# c) RAG 类型基线测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGTypes:
    """RAG 数据契约类型基线"""

    def test_retrieval_status_values(self):
        """验证 retrieval_status 的可能值：not_run / sufficient / insufficient / unavailable / error"""
        from app.services.rag.types import KnowledgeAssessment

        field = KnowledgeAssessment.model_fields["retrieval_status"]
        # Pydantic v2: Literal 类型存储在 annotation 中
        import typing
        args = typing.get_args(field.annotation)
        expected = {"not_run", "sufficient", "insufficient", "unavailable", "error"}
        assert set(args) == expected, f"retrieval_status 值已变更: {set(args)} != {expected}"

    def test_evidence_stance_values(self):
        """验证 evidence_stance 的可能值：supports / contradicts / mixed / undetermined"""
        from app.services.rag.types import KnowledgeAssessment

        field = KnowledgeAssessment.model_fields["evidence_stance"]
        import typing
        args = typing.get_args(field.annotation)
        expected = {"supports", "contradicts", "mixed", "undetermined"}
        assert set(args) == expected, f"evidence_stance 值已变更: {set(args)} != {expected}"

    def test_citation_structure(self):
        """验证 Citation Pydantic 模型字段"""
        from app.services.rag.types import Citation

        cite = Citation(
            citation_id="test:1:0:0",
            claim="test claim",
            source="test.pdf",
            page=1,
        )
        assert cite.citation_id == "test:1:0:0"
        assert cite.claim == "test claim"
        assert cite.source == "test.pdf"
        assert cite.page == 1
        # 可选字段默认值
        assert cite.heading_path == ""
        assert cite.text_snippet == ""
        assert cite.rerank_score is None

    def test_citation_required_fields(self):
        """验证 Citation 必填字段"""
        from app.services.rag.types import Citation

        # 缺少必填字段应抛出异常
        with pytest.raises(Exception):
            Citation(claim="test", source="test.pdf")  # 缺少 citation_id

        with pytest.raises(Exception):
            Citation(citation_id="x", source="test.pdf")  # 缺少 claim

        with pytest.raises(Exception):
            Citation(citation_id="x", claim="test")  # 缺少 source

    def test_retrieval_bundle_status_values(self):
        """验证 RetrievalBundle.status 的可能值"""
        from app.services.rag.types import RetrievalBundle

        field = RetrievalBundle.model_fields["status"]
        import typing
        args = typing.get_args(field.annotation)
        expected = {"candidate", "insufficient", "unavailable", "error"}
        assert set(args) == expected, f"RetrievalBundle.status 值已变更: {set(args)}"

    def test_knowledge_assessment_defaults(self):
        """验证 KnowledgeAssessment 默认值"""
        from app.services.rag.types import KnowledgeAssessment

        ka = KnowledgeAssessment()
        assert ka.score is None
        assert ka.confidence == 0.5
        assert ka.retrieval_status == "not_run"
        assert ka.evidence_stance == "undetermined"
        assert ka.human_review_needed is False
        assert ka.review_reason is None
        assert ka.citations == []


# ═══════════════════════════════════════════════════════════════════════════════
# d) Agent 接口基线测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSignatures:
    """验证所有 Agent 函数的存在性、async 属性和参数签名"""

    def test_agent_functions_are_async(self):
        """验证所有 Agent 主函数均为 async"""
        from app.services.agents.diagnosis_agent import run_diagnosis_evaluation
        from app.services.agents.humanistic_agent import run_humanistic_evaluation
        from app.services.agents.inquiry_agent import run_inquiry_analysis
        from app.services.agents.knowledge_agent import run_knowledge_check
        from app.services.agents.scoring_agent import run_scoring
        from app.services.agents.suggestion_agent import run_suggestion
        from app.services.agents.treatment_agent import run_treatment_evaluation

        assert inspect.iscoroutinefunction(run_inquiry_analysis), \
            "run_inquiry_analysis 应为 async"
        assert inspect.iscoroutinefunction(run_knowledge_check), \
            "run_knowledge_check 应为 async"
        assert inspect.iscoroutinefunction(run_humanistic_evaluation), \
            "run_humanistic_evaluation 应为 async"
        assert inspect.iscoroutinefunction(run_diagnosis_evaluation), \
            "run_diagnosis_evaluation 应为 async"
        assert inspect.iscoroutinefunction(run_treatment_evaluation), \
            "run_treatment_evaluation 应为 async"
        assert inspect.iscoroutinefunction(run_scoring), \
            "run_scoring 应为 async"
        assert inspect.iscoroutinefunction(run_suggestion), \
            "run_suggestion 应为 async"

    def test_inquiry_agent_signature(self):
        """验证 run_inquiry_analysis 参数签名"""
        from app.services.agents.inquiry_agent import run_inquiry_analysis

        sig = inspect.signature(run_inquiry_analysis)
        params = list(sig.parameters.keys())
        assert params == ["conversation_text", "patient_info"], (
            f"run_inquiry_analysis 参数已变更: {params}"
        )

    def test_knowledge_agent_signature(self):
        """验证 run_knowledge_check 参数签名"""
        from app.services.agents.knowledge_agent import run_knowledge_check

        sig = inspect.signature(run_knowledge_check)
        params = list(sig.parameters.keys())
        assert params == [
            "conversation_text", "patient_info",
            "doctor_diagnosis", "treatment_plan", "enable_hyde",
        ], f"run_knowledge_check 参数已变更: {params}"
        # enable_hyde 默认值为 True
        assert sig.parameters["enable_hyde"].default is True

    def test_humanistic_agent_signature(self):
        """验证 run_humanistic_evaluation 参数签名"""
        from app.services.agents.humanistic_agent import run_humanistic_evaluation

        sig = inspect.signature(run_humanistic_evaluation)
        params = list(sig.parameters.keys())
        assert params == ["conversation_text", "patient_info"], (
            f"run_humanistic_evaluation 参数已变更: {params}"
        )

    def test_diagnosis_agent_signature(self):
        """验证 run_diagnosis_evaluation 参数签名"""
        from app.services.agents.diagnosis_agent import run_diagnosis_evaluation

        sig = inspect.signature(run_diagnosis_evaluation)
        params = list(sig.parameters.keys())
        assert params == ["conversation_text", "patient_info", "doctor_diagnosis", "knowledge_citations"], (
            f"run_diagnosis_evaluation 参数已变更: {params}"
        )

    def test_treatment_agent_signature(self):
        """验证 run_treatment_evaluation 参数签名"""
        from app.services.agents.treatment_agent import run_treatment_evaluation

        sig = inspect.signature(run_treatment_evaluation)
        params = list(sig.parameters.keys())
        assert params == [
            "conversation_text", "patient_info",
            "doctor_diagnosis", "treatment_plan", "knowledge_citations",
        ], f"run_treatment_evaluation 参数已变更: {params}"

    def test_scoring_agent_signature(self):
        """验证 run_scoring 参数签名"""
        from app.services.agents.scoring_agent import run_scoring

        sig = inspect.signature(run_scoring)
        params = list(sig.parameters.keys())
        assert params == [
            "inquiry_score", "inquiry_analysis",
            "knowledge_score", "knowledge_analysis",
            "humanistic_score", "humanistic_analysis",
            "diagnosis_score", "diagnosis_analysis",
            "treatment_score", "treatment_analysis",
        ], f"run_scoring 参数已变更: {params}"

        # knowledge_score 无默认值（必须显式传入，可为 None）
        assert sig.parameters["knowledge_score"].default is inspect.Parameter.empty

        # diagnosis_score 和 treatment_score 有默认值 None
        assert sig.parameters["diagnosis_score"].default is None
        assert sig.parameters["treatment_score"].default is None

    def test_suggestion_agent_signature(self):
        """验证 run_suggestion 参数签名"""
        from app.services.agents.suggestion_agent import run_suggestion

        sig = inspect.signature(run_suggestion)
        params = list(sig.parameters.keys())
        assert params == [
            "conversation_text", "patient_info",
            "inquiry_result", "knowledge_result", "humanistic_result",
        ], f"run_suggestion 参数已变更: {params}"

    def test_agent_return_types(self):
        """验证所有 Agent 函数返回类型注解为 dict"""
        from app.services.agents.diagnosis_agent import run_diagnosis_evaluation
        from app.services.agents.humanistic_agent import run_humanistic_evaluation
        from app.services.agents.inquiry_agent import run_inquiry_analysis
        from app.services.agents.knowledge_agent import run_knowledge_check
        from app.services.agents.scoring_agent import run_scoring
        from app.services.agents.suggestion_agent import run_suggestion
        from app.services.agents.treatment_agent import run_treatment_evaluation

        for fn in [
            run_inquiry_analysis, run_knowledge_check,
            run_humanistic_evaluation, run_diagnosis_evaluation,
            run_treatment_evaluation, run_suggestion,
        ]:
            sig = inspect.signature(fn)
            assert sig.return_annotation is dict or sig.return_annotation == "dict", (
                f"{fn.__name__} 返回类型应为 dict，实际为 {sig.return_annotation}"
            )

        # run_scoring 返回 dict（包含 raw_response, total_score, summary）
        sig = inspect.signature(run_scoring)
        assert sig.return_annotation is dict
