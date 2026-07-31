# -*- coding: utf-8 -*-
"""agent.py 单元测试：PatientAgent 编排（LLM 全 mock）"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.services.agents.patient.agent import PatientAgent
from app.services.agents.patient.memory import Fact, MemoryState
from app.services.agents.patient.prompts import PATIENT_ROLE_WRAPPER, PATIENT_TOOL_GUIDE
from app.services.qwen_client import ToolCallResult


@pytest.fixture(autouse=True)
def _disable_patient_tool_use(monkeypatch):
    """默认走纯文本路径；工具路径由 TestRespondWithTools 内显式开启"""
    monkeypatch.setattr(settings, "ENABLE_PATIENT_TOOL_USE", False)


def _agent():
    memory = MemoryState(facts=[
        Fact(fact_id="sym_001", content="上腹隐痛", status="disclosed"),
        Fact(fact_id="his_001", category="history", content="青霉素过敏", status="denied"),
        Fact(fact_id="sym_002", content="反酸烧心"),
    ])
    patient = SimpleNamespace(
        system_prompt="45岁男性，上腹痛两周", personality_type="配合型",
        expected_diagnosis="胃溃疡",
    )
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

    def test_personality_anchor_injected(self):
        # 本轮风格段应锚定具体人格，强化长对话人格一致性
        assert "你始终是配合型患者" in _agent()._build_system_prompt()


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

    @pytest.mark.asyncio
    async def test_emotion_updated_each_turn(self):
        agent = _agent()
        agent.memory.emotion = "焦虑"
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("别担心，慢慢说。", [])
        assert agent.memory.emotion == "缓和"

    def test_emotion_injected_into_prompt(self):
        agent = _agent()
        agent.memory.emotion = "恐慌"
        assert "当前情绪状态：恐慌" in agent._build_system_prompt()

    @pytest.mark.asyncio
    async def test_trust_rises_on_comfort(self):
        agent = _agent()
        agent.memory.trust = 0.4
        with patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯。")), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            await agent.respond("别担心，慢慢说。", [])
        assert abs(agent.memory.trust - 0.5) < 1e-9


def _tool_result(content="有点反酸。", degraded=False, error=None):
    return ToolCallResult(content=content, messages=[], tool_calls=[], degraded=degraded, error=error)


class TestRespondWithTools:
    """LLM 自主 function-calling 主回复路径（ENABLE_PATIENT_TOOL_USE=True）"""

    @pytest.mark.asyncio
    async def test_tool_path_sends_two_patient_schemas(self, monkeypatch):
        """开关开启时走 call_qwen_with_tools，只下发 2 个患者工具 schema（emotion_engine 不下发）"""
        monkeypatch.setattr(settings, "ENABLE_PATIENT_TOOL_USE", True)
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_with_tools", new=AsyncMock(return_value=_tool_result())) as fc, \
             patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock()) as chat, \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            reply = await agent.respond("还有什么症状？", [])
        assert reply == "有点反酸。"
        fc.assert_awaited_once()
        chat.assert_not_awaited()
        tools = fc.await_args.kwargs["tools"]
        names = {t["function"]["name"] for t in tools}
        assert names == {"query_plausible_symptom", "physiology_calculator"}
        # system prompt 追加了工具使用指引
        messages = fc.await_args.args[0]
        assert PATIENT_TOOL_GUIDE in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_degraded_falls_back_to_chat(self, monkeypatch):
        """工具路径降级时回退现行 call_qwen_chat 纯文本路径"""
        monkeypatch.setattr(settings, "ENABLE_PATIENT_TOOL_USE", True)
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_with_tools",
                   new=AsyncMock(return_value=_tool_result(content="", degraded=True, error="boom"))), \
             patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯，有点疼。")) as chat, \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            reply = await agent.respond("还有什么症状？", [])
        assert reply == "嗯，有点疼。"
        chat.assert_awaited_once()
        assert agent.memory.turn == 1

    @pytest.mark.asyncio
    async def test_switch_off_never_touches_tool_path(self):
        """开关关闭（autouse fixture）时行为与现行完全一致"""
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_with_tools", new=AsyncMock()) as fc, \
             patch("app.services.agents.patient.agent.call_qwen_chat", new=AsyncMock(return_value="嗯。")) as chat, \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()):
            reply = await agent.respond("哪里不舒服？", [])
        assert reply == "嗯。"
        fc.assert_not_awaited()
        chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_path_memory_updates_unchanged(self, monkeypatch):
        """工具路径下 turn/stage/账本更新逻辑与纯文本路径一致"""
        monkeypatch.setattr(settings, "ENABLE_PATIENT_TOOL_USE", True)
        agent = _agent()
        with patch("app.services.agents.patient.agent.call_qwen_with_tools", new=AsyncMock(return_value=_tool_result())), \
             patch("app.services.agents.patient.agent.update_ledger", new=AsyncMock()) as ledger:
            await agent.respond("还有什么症状？", [])
        assert agent.memory.turn == 1
        assert agent.memory.stage_history == [agent.memory.stage]
        ledger.assert_awaited_once()
