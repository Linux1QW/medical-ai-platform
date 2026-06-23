"""KnowledgeAdapter Feature Flag 切换测试

测试 ENABLE_TOOL_USE 开关控制知识 Agent 走不同路径。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.orchestration.adapters.knowledge import KnowledgeAdapter
from app.orchestration.state import EvaluationContext, AgentResultEnvelope


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def eval_context():
    """创建测试用 EvaluationContext"""
    return EvaluationContext(
        conversation_text="患者：我最近总是咳嗽\n医生：咳嗽多久了？\n患者：大概一周了",
        patient_age=45,
        patient_gender="male",
        chief_complaint="咳嗽一周",
        symptoms=["咳嗽", "咳痰"],
        doctor_diagnosis="急性支气管炎",
        treatment_plan="头孢克洛 0.25g tid × 7天",
    )


@pytest.fixture
def adapter():
    return KnowledgeAdapter()


# ── 测试用例 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_flag_false_uses_legacy(mock_settings, mock_legacy, eval_context, adapter):
    """ENABLE_TOOL_USE=false 时调用 run_knowledge_check"""
    mock_settings.ENABLE_TOOL_USE = False

    mock_legacy.return_value = {
        "raw_response": '{"score": 75, "analysis": "test"}',
        "score": 75,
        "analysis": "test analysis",
        "retrieval_status": "sufficient",
        "evidence_stance": "supports",
        "citations": [],
        "human_review_needed": False,
        "review_reason": None,
        "confidence": 0.8,
        "rag_trace": {},
        "degraded": False,
    }

    result = await adapter._call_agent(eval_context)

    mock_legacy.assert_called_once()
    assert result["score"] == 75


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check_with_tools", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_flag_true_uses_tool_use(mock_settings, mock_tool_use, eval_context, adapter):
    """ENABLE_TOOL_USE=true 时调用 run_knowledge_check_with_tools"""
    mock_settings.ENABLE_TOOL_USE = True

    mock_tool_use.return_value = {
        "raw_response": '{"score": 85, "analysis": "tool use result"}',
        "score": 85,
        "analysis": "tool use analysis",
        "retrieval_status": "sufficient",
        "evidence_stance": "supports",
        "citations": [],
        "human_review_needed": False,
        "review_reason": None,
        "confidence": 0.9,
        "rag_trace": {},
        "tool_trace": [],
        "degraded": False,
    }

    result = await adapter._call_agent(eval_context)

    mock_tool_use.assert_called_once()
    assert result["score"] == 85


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.run_knowledge_check_with_tools", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_fallback_on_error(mock_settings, mock_tool_use, mock_legacy, eval_context, adapter):
    """Tool Use 异常 + FALLBACK=true → 回退旧路径"""
    mock_settings.ENABLE_TOOL_USE = True
    mock_settings.TOOL_USE_FALLBACK_TO_LEGACY = True

    mock_tool_use.side_effect = RuntimeError("Tool Use failed")
    mock_legacy.return_value = {
        "raw_response": '{"score": 60, "analysis": "legacy fallback"}',
        "score": 60,
        "analysis": "legacy fallback analysis",
        "retrieval_status": "sufficient",
        "evidence_stance": "mixed",
        "citations": [],
        "human_review_needed": False,
        "review_reason": None,
        "confidence": 0.6,
        "rag_trace": {},
        "degraded": False,
    }

    result = await adapter._call_agent(eval_context)

    mock_tool_use.assert_called_once()
    mock_legacy.assert_called_once()
    assert result["score"] == 60


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check_with_tools", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_no_fallback_raises(mock_settings, mock_tool_use, eval_context, adapter):
    """Tool Use 异常 + FALLBACK=false → 抛出异常"""
    mock_settings.ENABLE_TOOL_USE = True
    mock_settings.TOOL_USE_FALLBACK_TO_LEGACY = False

    mock_tool_use.side_effect = RuntimeError("Tool Use failed")

    with pytest.raises(RuntimeError, match="Tool Use failed"):
        await adapter._call_agent(eval_context)


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check_with_tools", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_trace_in_envelope(mock_settings, mock_tool_use, eval_context, adapter):
    """tool_trace 正确传入 AgentResultEnvelope.trace"""
    mock_settings.ENABLE_TOOL_USE = True

    tool_trace_data = [
        {"tool_name": "search_medical_kb", "status": "success", "elapsed_ms": 100},
    ]
    mock_tool_use.return_value = {
        "raw_response": '{"score": 80, "analysis": "test"}',
        "score": 80,
        "analysis": "test analysis",
        "retrieval_status": "sufficient",
        "evidence_stance": "supports",
        "citations": [],
        "human_review_needed": False,
        "review_reason": None,
        "confidence": 0.8,
        "rag_trace": {"queries": ["test"]},
        "tool_trace": tool_trace_data,
        "degraded": False,
    }

    raw = await adapter._call_agent(eval_context)
    envelope = adapter._parse_result(raw)

    assert isinstance(envelope, AgentResultEnvelope)
    assert envelope.agent_name == "knowledge"
    assert envelope.score == 80.0
    assert "rag_trace" in envelope.trace
    assert "tool_trace" in envelope.trace
    assert envelope.trace["tool_trace"] == tool_trace_data


@pytest.mark.asyncio
@patch("app.orchestration.adapters.knowledge.run_knowledge_check_with_tools", new_callable=AsyncMock)
@patch("app.orchestration.adapters.knowledge.settings")
async def test_parse_result_score_none(mock_settings, mock_tool_use, eval_context, adapter):
    """score=None 时 status=insufficient"""
    mock_settings.ENABLE_TOOL_USE = True

    mock_tool_use.return_value = {
        "raw_response": '{"score": null, "analysis": "证据不足"}',
        "score": None,
        "analysis": "证据不足",
        "retrieval_status": "insufficient",
        "evidence_stance": "undetermined",
        "citations": [],
        "human_review_needed": True,
        "review_reason": "insufficient_evidence",
        "confidence": 0.3,
        "rag_trace": {},
        "tool_trace": [],
        "degraded": False,
    }

    raw = await adapter._call_agent(eval_context)
    envelope = adapter._parse_result(raw)

    assert envelope.status == "insufficient"
    assert envelope.score is None
    assert envelope.human_review_needed is True
