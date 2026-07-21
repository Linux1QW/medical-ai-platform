"""Agent 适配器单元测试"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.orchestration.adapters.diagnosis import DiagnosisAdapter
from app.orchestration.adapters.humanistic import HumanisticAdapter
from app.orchestration.adapters.inquiry import InquiryAdapter
from app.orchestration.adapters.knowledge import KnowledgeAdapter
from app.orchestration.adapters.registry import _REGISTRY, get_adapter, list_adapters, register_adapter
from app.orchestration.adapters.treatment import TreatmentAdapter
from app.orchestration.state import AgentResultEnvelope, EvaluationContext


@pytest.fixture
def sample_context():
    return EvaluationContext(
        conversation_text="医生: 你怎么不舒服\n患者: 我肚子疼",
        patient_age=45,
        patient_gender="female",
        chief_complaint="腹痛",
        medical_history="高血压",
        symptoms=["腹痛", "恶心"],
        doctor_diagnosis="慢性胃炎",
        treatment_plan="奥美拉唑 20mg qd",
    )


# ── Inquiry Adapter Tests ───────────────────────────────────────────────────

class TestInquiryAdapter:
    @pytest.mark.asyncio
    async def test_normal_path(self, sample_context):
        """测试正常路径：inquiry agent 返回有效分数和分析"""
        adapter = InquiryAdapter()
        mock_raw = {
            "raw_response": json.dumps({
                "score": 85,
                "analysis": "问诊覆盖良好，关键信息采集完整。",
                "details": {"coverage": {"score": 90}}
            })
        }
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert isinstance(result, AgentResultEnvelope)
        assert result.agent_name == "inquiry"
        assert result.status == "success"
        assert result.score == 85.0
        assert "问诊覆盖良好" in result.analysis

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_context):
        """测试异常处理：Agent 抛异常 → error envelope"""
        adapter = InquiryAdapter()
        with patch.object(adapter, "_call_agent", new=AsyncMock(side_effect=RuntimeError("LLM failed"))):
            result = await adapter.run(sample_context)

        assert result.status == "error"
        assert result.human_review_needed is True
        assert "inquiry_error" in result.review_reason

    @pytest.mark.asyncio
    async def test_score_clamp_high(self, sample_context):
        """测试分数 clamp：>100 → 100"""
        adapter = InquiryAdapter()
        mock_raw = {"raw_response": json.dumps({"score": 150, "analysis": "test"})}
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.score == 100.0

    @pytest.mark.asyncio
    async def test_score_clamp_low(self, sample_context):
        """测试分数 clamp：<0 → 0"""
        adapter = InquiryAdapter()
        mock_raw = {"raw_response": json.dumps({"score": -10, "analysis": "test"})}
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.score == 0.0


# ── Diagnosis Adapter Tests ─────────────────────────────────────────────────

class TestDiagnosisAdapter:
    @pytest.mark.asyncio
    async def test_normal_path(self, sample_context):
        """测试正常路径"""
        adapter = DiagnosisAdapter()
        mock_raw = {"raw_response": json.dumps({"score": 78, "analysis": "诊断基本正确"})}
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.agent_name == "diagnosis"
        assert result.status == "success"
        assert result.score == 78.0

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_context):
        """测试异常处理"""
        adapter = DiagnosisAdapter()
        with patch.object(adapter, "_call_agent", new=AsyncMock(side_effect=ValueError("API error"))):
            result = await adapter.run(sample_context)

        assert result.status == "error"
        assert result.human_review_needed is True


# ── Treatment Adapter Tests ─────────────────────────────────────────────────

class TestTreatmentAdapter:
    @pytest.mark.asyncio
    async def test_normal_path(self, sample_context):
        """测试正常路径"""
        adapter = TreatmentAdapter()
        mock_raw = {"raw_response": json.dumps({"score": 92, "analysis": "治疗方案规范"})}
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.agent_name == "treatment"
        assert result.status == "success"
        assert result.score == 92.0

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_context):
        """测试异常处理"""
        adapter = TreatmentAdapter()
        with patch.object(adapter, "_call_agent", new=AsyncMock(side_effect=Exception("timeout"))):
            result = await adapter.run(sample_context)

        assert result.status == "error"


# ── Knowledge Adapter Tests ─────────────────────────────────────────────────

class TestKnowledgeAdapter:
    @pytest.mark.asyncio
    async def test_normal_path(self, sample_context):
        """测试正常路径"""
        adapter = KnowledgeAdapter()
        mock_raw = {
            "score": 88,
            "analysis": "与指南一致",
            "citations": [{"source": "CSCO指南"}],
            "human_review_needed": False,
            "review_reason": None,
            "rag_trace": {"level_used": 1},
        }
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.agent_name == "knowledge"
        assert result.status == "success"
        assert result.score == 88.0
        assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_insufficient_path(self, sample_context):
        """测试拒答路径：score=None → insufficient"""
        adapter = KnowledgeAdapter()
        mock_raw = {
            "score": None,
            "analysis": "证据不足",
            "citations": [],
            "human_review_needed": True,
            "review_reason": "检索状态: insufficient",
            "rag_trace": {},
        }
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.status == "insufficient"
        assert result.score is None
        assert result.human_review_needed is True

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_context):
        """测试异常处理"""
        adapter = KnowledgeAdapter()
        with patch.object(adapter, "_call_agent", new=AsyncMock(side_effect=ConnectionError("RAG fail"))):
            result = await adapter.run(sample_context)

        assert result.status == "error"
        assert result.human_review_needed is True


# ── Humanistic Adapter Tests ────────────────────────────────────────────────

class TestHumanisticAdapter:
    @pytest.mark.asyncio
    async def test_normal_path(self, sample_context):
        """测试正常路径"""
        adapter = HumanisticAdapter()
        mock_raw = {
            "raw_response": json.dumps({
                "score": 75,
                "analysis": "共情表现一般",
                "details": {"empathy": {"score": 70}}
            })
        }
        with patch.object(adapter, "_call_agent", new=AsyncMock(return_value=mock_raw)):
            result = await adapter.run(sample_context)

        assert result.agent_name == "humanistic"
        assert result.status == "success"
        assert result.score == 75.0

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_context):
        """测试异常处理"""
        adapter = HumanisticAdapter()
        with patch.object(adapter, "_call_agent", new=AsyncMock(side_effect=TimeoutError())):
            result = await adapter.run(sample_context)

        assert result.status == "error"


# ── Registry Tests ──────────────────────────────────────────────────────────

class TestRegistry:
    def setup_method(self):
        """每个测试前清空注册表"""
        _REGISTRY.clear()

    def test_register_and_get(self):
        """测试注册和查找"""
        adapter = InquiryAdapter()
        register_adapter(adapter)

        retrieved = get_adapter("inquiry")
        assert retrieved is adapter

    def test_get_unregistered_raises(self):
        """测试获取未注册的适配器抛出 KeyError"""
        with pytest.raises(KeyError, match="未注册的适配器"):
            get_adapter("nonexistent")

    def test_list_adapters(self):
        """测试列出所有已注册的适配器"""
        register_adapter(InquiryAdapter())
        register_adapter(DiagnosisAdapter())

        adapters = list_adapters()
        assert "inquiry" in adapters
        assert "diagnosis" in adapters
        assert len(adapters) == 2
