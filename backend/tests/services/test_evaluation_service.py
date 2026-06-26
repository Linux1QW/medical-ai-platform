# -*- coding: utf-8 -*-
"""evaluation_service.py 单元测试

测试策略：
- Mock 数据库会话、Agent 调用和 WebSocket 管理器
- 覆盖核心路径：评估触发、结果处理、JSON 解析、评分计算、统计查询
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.models.consultation import Consultation, ConsultationMessage
from app.models.evaluation import Evaluation
from app.models.patient import VirtualPatient
from app.models.user import User
from app.services.evaluation_service import (
    _extract_json,
    _parse_symptoms,
    _score_range_label,
    EvaluationValidationError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    """创建模拟数据库会话"""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.flush = AsyncMock()
    return db


@pytest.fixture
def sample_consultation():
    return Consultation(
        id=1,
        doctor_id=10,
        patient_id=20,
        status="completed",
        started_at=datetime(2025, 1, 1, 10, 0),
        ended_at=datetime(2025, 1, 1, 11, 0),
        diagnosis="上呼吸道感染",
        treatment_plan="多喝水，休息",
    )


@pytest.fixture
def sample_patient():
    return VirtualPatient(
        id=20,
        name="张三",
        age=45,
        gender="男",
        personality_type="配合型",
        chief_complaint="头痛三天",
        medical_history="无特殊病史",
        symptoms='["头痛", "发热"]',
        system_prompt="患者张三，头痛三天。",
        expected_diagnosis="上呼吸道感染",
    )


@pytest.fixture
def sample_messages():
    return [
        ConsultationMessage(id=1, consultation_id=1, role="doctor", content="哪里不舒服？", sequence=1),
        ConsultationMessage(id=2, consultation_id=1, role="patient", content="头痛。", sequence=2),
        ConsultationMessage(id=3, consultation_id=1, role="doctor", content="多久了？", sequence=3),
        ConsultationMessage(id=4, consultation_id=1, role="patient", content="三天了。", sequence=4),
    ]


# ── 测试 _extract_json ────────────────────────────────────────────────────────

class TestExtractJson:
    """测试从 LLM 返回文本中提取 JSON"""

    def test_valid_json(self):
        """直接解析有效 JSON"""
        text = '{"score": 85, "analysis": "良好"}'
        result = _extract_json(text)
        assert result["score"] == 85
        assert result["analysis"] == "良好"

    def test_json_with_markdown_code_block(self):
        """解析包含 markdown 代码块的 JSON"""
        text = '```json\n{"score": 85, "analysis": "良好"}\n```'
        result = _extract_json(text)
        assert result["score"] == 85

    def test_json_with_surrounding_text(self):
        """解析包含前后文本的 JSON"""
        text = '以下是评估结果：\n{"score": 85, "analysis": "良好"}\n希望对你有帮助。'
        result = _extract_json(text)
        assert result["score"] == 85

    def test_empty_text_raises_error(self):
        """空文本抛出异常"""
        with pytest.raises(EvaluationValidationError):
            _extract_json("")

    def test_none_text_raises_error(self):
        """None 文本抛出异常"""
        with pytest.raises(EvaluationValidationError):
            _extract_json(None)

    def test_invalid_json_raises_error(self):
        """无效 JSON 抛出异常"""
        with pytest.raises(EvaluationValidationError):
            _extract_json("这不是 JSON")

    def test_error_contains_raw_response(self):
        """异常包含原始返回体"""
        raw = "invalid content"
        with pytest.raises(EvaluationValidationError) as exc_info:
            _extract_json(raw)
        assert exc_info.value.raw_response == raw


# ── 测试 _parse_symptoms ──────────────────────────────────────────────────────

class TestParseSymptoms:
    """测试症状字段解析"""

    def test_json_array(self):
        """解析 JSON 数组格式症状"""
        result = _parse_symptoms('["头痛", "发热", "咳嗽"]')
        assert result == ["头痛", "发热", "咳嗽"]

    def test_plain_string(self):
        """纯文本症状作为单元素列表"""
        result = _parse_symptoms("头痛")
        assert result == ["头痛"]

    def test_empty_string(self):
        """空字符串返回空列表"""
        result = _parse_symptoms("")
        assert result == []

    def test_none(self):
        """None 返回空列表"""
        result = _parse_symptoms(None)
        assert result == []

    def test_non_list_json(self):
        """非列表 JSON 作为单元素列表"""
        result = _parse_symptoms('{"symptom": "头痛"}')
        assert result == ['{"symptom": "头痛"}']


# ── 测试 _score_range_label ───────────────────────────────────────────────────

class TestScoreRangeLabel:
    """测试分数区间标签"""

    def test_excellent(self):
        assert _score_range_label(95) == "优秀(90-100)"
        assert _score_range_label(90) == "优秀(90-100)"

    def test_good(self):
        assert _score_range_label(85) == "良好(80-89)"
        assert _score_range_label(80) == "良好(80-89)"

    def test_average(self):
        assert _score_range_label(70) == "一般(60-79)"
        assert _score_range_label(60) == "一般(60-79)"

    def test_fail(self):
        assert _score_range_label(50) == "不及格(<60)"
        assert _score_range_label(0) == "不及格(<60)"


# ── 测试 run_evaluation（旧编排路径）──────────────────────────────────────────

class TestRunEvaluationLegacy:
    """测试旧编排路径的评估触发"""

    @pytest.mark.asyncio
    @patch("app.services.evaluation_service.settings")
    @patch("app.services.evaluation_service.manager")
    @patch("app.services.evaluation_service.run_suggestion", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_scoring", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_treatment_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_diagnosis_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_humanistic_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_knowledge_check", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_inquiry_analysis", new_callable=AsyncMock)
    async def test_legacy_evaluation_success(
        self,
        mock_inquiry,
        mock_knowledge,
        mock_humanistic,
        mock_diagnosis,
        mock_treatment,
        mock_scoring,
        mock_suggestion,
        mock_manager,
        mock_settings,
        mock_db,
        sample_consultation,
        sample_patient,
        sample_messages,
    ):
        """旧编排路径成功运行评估"""
        from app.services.evaluation_service import run_evaluation

        # 设置 LANGGRAPH_ENABLED=False 走旧路径
        mock_settings.LANGGRAPH_ENABLED = False

        # mock manager.send_progress 为异步方法
        mock_manager.send_progress = AsyncMock()

        # mock 数据库查询
        mock_consult_result = MagicMock()
        mock_consult_result.scalar_one.return_value = sample_consultation

        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient

        mock_msgs_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_messages
        mock_msgs_result.scalars.return_value = mock_scalars

        mock_eval_result = MagicMock()
        mock_eval_result.scalar_one_or_none.return_value = None

        # 按调用顺序返回不同结果
        mock_db.execute = AsyncMock(
            side_effect=[mock_consult_result, mock_patient_result, mock_msgs_result, mock_eval_result]
        )

        async def mock_refresh(obj):
            obj.id = 1
        mock_db.refresh = mock_refresh

        # mock Agent 返回
        mock_inquiry.return_value = {"raw_response": json.dumps({"score": 85, "analysis": "良好"})}
        mock_knowledge.return_value = {
            "raw_response": json.dumps({"score": 80, "analysis": "通过"}),
            "retrieval_status": "success",
            "evidence_stance": "supported",
            "human_review_needed": False,
        }
        mock_humanistic.return_value = {"raw_response": json.dumps({"score": 90, "analysis": "沟通良好"})}
        mock_diagnosis.return_value = {"raw_response": json.dumps({"score": 75, "analysis": "诊断基本正确"})}
        mock_treatment.return_value = {"raw_response": json.dumps({"score": 70, "analysis": "方案合理"})}
        mock_scoring.return_value = {"raw_response": json.dumps({"total_score": 80, "summary": "整体良好"})}
        mock_suggestion.return_value = {"raw_response": json.dumps({"suggestions": "继续保持"})}

        result = await run_evaluation(mock_db, 1)

        assert result is not None
        assert result.consultation_id == 1
        assert result.inquiry_score == 85
        assert result.knowledge_score == 80
        assert result.humanistic_score == 90
        assert result.diagnosis_score == 75
        assert result.treatment_score == 70
        assert result.total_score == 80
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    @patch("app.services.evaluation_service.settings")
    @patch("app.services.evaluation_service.manager")
    @patch("app.services.evaluation_service.run_suggestion", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_scoring", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_treatment_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_diagnosis_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_humanistic_evaluation", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_knowledge_check", new_callable=AsyncMock)
    @patch("app.services.evaluation_service.run_inquiry_analysis", new_callable=AsyncMock)
    async def test_legacy_knowledge_refusal_sets_null_scores(
        self,
        mock_inquiry,
        mock_knowledge,
        mock_humanistic,
        mock_diagnosis,
        mock_treatment,
        mock_scoring,
        mock_suggestion,
        mock_manager,
        mock_settings,
        mock_db,
        sample_consultation,
        sample_patient,
        sample_messages,
    ):
        """知识代理拒答时 knowledge_score 和 total_score 置为 None"""
        from app.services.evaluation_service import run_evaluation

        mock_settings.LANGGRAPH_ENABLED = False

        # mock manager.send_progress 为异步方法
        mock_manager.send_progress = AsyncMock()

        mock_consult_result = MagicMock()
        mock_consult_result.scalar_one.return_value = sample_consultation
        mock_patient_result = MagicMock()
        mock_patient_result.scalar_one.return_value = sample_patient
        mock_msgs_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_messages
        mock_msgs_result.scalars.return_value = mock_scalars
        mock_eval_result = MagicMock()
        mock_eval_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[mock_consult_result, mock_patient_result, mock_msgs_result, mock_eval_result]
        )

        async def mock_refresh(obj):
            obj.id = 1
        mock_db.refresh = mock_refresh

        mock_inquiry.return_value = {"raw_response": json.dumps({"score": 85, "analysis": "良好"})}
        mock_knowledge.return_value = {
            "raw_response": json.dumps({"score": 60, "analysis": "证据不足"}),
            "retrieval_status": "insufficient",
            "evidence_stance": "refusal",
            "human_review_needed": True,
            "review_reason": "证据不充分",
        }
        mock_humanistic.return_value = {"raw_response": json.dumps({"score": 90, "analysis": "良好"})}
        mock_diagnosis.return_value = {"raw_response": json.dumps({"score": 75, "analysis": "基本正确"})}
        mock_treatment.return_value = {"raw_response": json.dumps({"score": 70, "analysis": "合理"})}
        mock_scoring.return_value = {"raw_response": json.dumps({"total_score": 80, "summary": "整体良好"})}
        mock_suggestion.return_value = {"raw_response": json.dumps({"suggestions": "改进知识"})}

        result = await run_evaluation(mock_db, 1)

        # 拒答时 knowledge_score 和 total_score 应为 None
        assert result.knowledge_score is None
        assert result.total_score is None
        assert result.human_review_needed is True
        assert result.evaluation_status == "needs_review"


# ── 测试 get_evaluation_by_consultation ───────────────────────────────────────

class TestGetEvaluationByConsultation:
    """测试获取评估结果"""

    @pytest.mark.asyncio
    async def test_found(self, mock_db):
        """找到评估记录时返回对象"""
        from app.services.evaluation_service import get_evaluation_by_consultation

        mock_eval = Evaluation(
            id=1,
            consultation_id=1,
            inquiry_score=85,
            total_score=80,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_eval
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_evaluation_by_consultation(mock_db, 1)
        assert result is not None
        assert result.consultation_id == 1
        assert result.total_score == 80

    @pytest.mark.asyncio
    async def test_not_found(self, mock_db):
        """未找到评估记录时返回 None"""
        from app.services.evaluation_service import get_evaluation_by_consultation

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await get_evaluation_by_consultation(mock_db, 999)
        assert result is None


# ── 测试 EvaluationValidationError ────────────────────────────────────────────

class TestEvaluationValidationError:
    """测试自定义异常"""

    def test_error_attributes(self):
        """异常包含正确的属性"""
        error = EvaluationValidationError("解析失败", '{"invalid": }')
        assert error.message == "解析失败"
        assert error.raw_response == '{"invalid": }'
        assert str(error) == "解析失败"
