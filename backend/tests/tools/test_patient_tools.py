# -*- coding: utf-8 -*-
"""患者专属工具单元测试（RAG/LLM 全 mock）"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.tools.base import ToolContext
from app.services.tools.patient.plausible_symptom import (
    QueryPlausibleSymptom,
    QueryPlausibleSymptomArgs,
)


def _context():
    return ToolContext(run_id="t-1", agent_name="patient_agent")


def _bundle(texts):
    items = [SimpleNamespace(text=t, source="内科学", heading_path="消化系统", rrf_score=0.9) for t in texts]
    return SimpleNamespace(candidates=items, level_used="base", trace=None)


class TestQueryPlausibleSymptom:
    @pytest.mark.asyncio
    async def test_present_verdict(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="夜间痛醒", diagnosis="十二指肠溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(return_value=_bundle(["十二指肠溃疡典型表现为夜间痛、饥饿痛"]))), \
             patch("app.services.tools.patient.plausible_symptom.call_qwen_chat", new=AsyncMock(return_value='{"verdict": "present", "reason": "夜间痛是典型伴随症状"}')):
            result = await tool.execute(args, _context())
        assert result["verdict"] == "present"
        assert result["degraded"] is False

    @pytest.mark.asyncio
    async def test_invalid_verdict_coerced_to_uncertain(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="头疼", diagnosis="胃溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(return_value=_bundle(["…"]))), \
             patch("app.services.tools.patient.plausible_symptom.call_qwen_chat", new=AsyncMock(return_value='{"verdict": "maybe", "reason": "?"}')):
            result = await tool.execute(args, _context())
        assert result["verdict"] == "uncertain"

    @pytest.mark.asyncio
    async def test_retrieval_failure_degrades(self):
        tool = QueryPlausibleSymptom()
        args = QueryPlausibleSymptomArgs(symptom="头疼", diagnosis="胃溃疡")
        with patch("app.services.tools.patient.plausible_symptom.tiered_retrieve", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await tool.execute(args, _context())
        assert result == {"verdict": "uncertain", "reason": "知识库裁决失败，保守处理", "degraded": True}


from app.services.tools.patient.physiology import (
    PhysiologyCalculator,
    PhysiologyCalculatorArgs,
)


class TestPhysiologyCalculator:
    @pytest.mark.asyncio
    async def test_deterministic_same_seed(self):
        tool = PhysiologyCalculator()
        args = PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=False)
        r1 = await tool.execute(args, _context())
        r2 = await tool.execute(args, _context())
        assert r1 == r2
        assert r1["unit"] == "℃"
        assert 36.0 <= float(r1["value"]) <= 37.2

    @pytest.mark.asyncio
    async def test_abnormal_range_differs(self):
        tool = PhysiologyCalculator()
        normal = await tool.execute(PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=False), _context())
        fever = await tool.execute(PhysiologyCalculatorArgs(vital="body_temperature", consultation_id=42, abnormal=True), _context())
        assert float(fever["value"]) >= 37.8
        assert float(fever["value"]) != float(normal["value"])

    @pytest.mark.asyncio
    async def test_blood_pressure_format(self):
        tool = PhysiologyCalculator()
        r = await tool.execute(PhysiologyCalculatorArgs(vital="blood_pressure", consultation_id=7, abnormal=False), _context())
        assert "/" in r["value"] and r["unit"] == "mmHg"

    @pytest.mark.asyncio
    async def test_unknown_vital_rejected(self):
        tool = PhysiologyCalculator()
        r = await tool.execute(PhysiologyCalculatorArgs(vital="unknown_thing", consultation_id=1, abnormal=False), _context())
        assert r.get("error")
