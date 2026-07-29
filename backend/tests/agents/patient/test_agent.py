# -*- coding: utf-8 -*-
"""agent.py 单元测试：PatientAgent 编排（LLM 全 mock）"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.agents.patient.agent import PatientAgent
from app.services.agents.patient.memory import Fact, MemoryState
from app.services.agents.patient.prompts import PATIENT_ROLE_WRAPPER


def _agent():
    memory = MemoryState(facts=[
        Fact(fact_id="sym_001", content="上腹隐痛", status="disclosed"),
        Fact(fact_id="his_001", category="history", content="青霉素过敏", status="denied"),
        Fact(fact_id="sym_002", content="反酸烧心"),
    ])
    patient = SimpleNamespace(system_prompt="45岁男性，上腹痛两周", personality_type="配合型")
    return PatientAgent(patient, memory)


def test_wrapper_migrated_with_placeholder():
    assert "{system_prompt}" in PATIENT_ROLE_WRAPPER
    # 向后兼容：consultation_service 仍可导入同名常量
    from app.services import consultation_service
    assert consultation_service.PATIENT_ROLE_WRAPPER is PATIENT_ROLE_WRAPPER


class TestBuildSystemPrompt:
    def test_contains_ledger_sections(self):
        prompt = _agent()._build_system_prompt()
        assert "45岁男性，上腹痛两周" in prompt
        assert "你已经告诉过医生的信息" in prompt and "上腹隐痛" in prompt
        assert "绝对不能再承认" in prompt and "青霉素过敏" in prompt

    def test_empty_ledger_no_sections(self):
        agent = _agent()
        agent.memory = MemoryState()
        prompt = agent._build_system_prompt()
        assert "你已经告诉过医生的信息" not in prompt
        assert "绝对不能再承认" not in prompt


class TestRespond:
    @pytest.mark.asyncio
    async def test_normal_flow_updates_memory(self):
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="有点反酸烧心。")) as llm, \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()) as ledger:
            reply = await agent.respond("还有什么症状？", [])
        assert reply == "有点反酸烧心。"
        assert agent.memory.turn == 1
        assert agent.memory.stage_history == [agent.memory.stage]
        llm.assert_awaited_once()
        ledger.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stage_transition(self):
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="肚子上面疼。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("您好，哪里不舒服？", [])
        assert agent.memory.stage == "chief_complaint"

    @pytest.mark.asyncio
    async def test_contradiction_triggers_regeneration(self):
        agent = _agent()
        llm = AsyncMock(side_effect=["对，我青霉素过敏。", "没有，我没有过敏。"])
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=llm), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            reply = await agent.respond("你有青霉素过敏吗？", [])
        assert llm.await_count == 2
        assert reply == "没有，我没有过敏。"

    @pytest.mark.asyncio
    async def test_llm_failure_leaves_memory_untouched(self):
        """LLM 调用失败向上抛时，记忆不应留下幽灵轮次"""
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(RuntimeError):
                await agent.respond("哪里不舒服？", [])
        assert agent.memory.turn == 0
        assert agent.memory.stage_history == []
