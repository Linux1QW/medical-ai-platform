# -*- coding: utf-8 -*-
"""guard.py 单元测试：账本更新（LLM+规则兜底）与矛盾检测"""
from unittest.mock import AsyncMock, patch

import pytest

from app.services.agents.patient.guard import (
    _rule_based_update,
    check_contradiction,
    update_ledger,
)
from app.services.agents.patient.memory import Fact, MemoryState


def _memory():
    return MemoryState(turn=2, facts=[
        Fact(fact_id="sym_001", content="上腹隐痛"),
        Fact(fact_id="sym_002", content="反酸烧心"),
        Fact(fact_id="his_001", category="history", content="青霉素过敏"),
    ])


class TestUpdateLedger:
    @pytest.mark.asyncio
    async def test_llm_marks_disclosed_and_denied(self):
        m = _memory()
        out = '{"disclosed": ["sym_001"], "denied": ["his_001"]}'
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=AsyncMock(return_value=out)):
            await update_ledger(m, "哪里不舒服？有过敏吗？", "肚子上面隐隐地疼。没有过敏。")
        assert m.find_fact("sym_001").status == "disclosed"
        assert m.find_fact("sym_001").disclosed_at_turn == 2
        assert m.find_fact("his_001").status == "denied"
        assert m.find_fact("sym_002").status == "undisclosed"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self):
        m = _memory()
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await update_ledger(m, "还有什么症状？", "有点反酸烧心。")  # 不抛异常
        assert m.find_fact("sym_002").status == "disclosed"

    @pytest.mark.asyncio
    async def test_no_pending_facts_skips_llm(self):
        m = _memory()
        for f in m.facts:
            f.status = "disclosed"
        mock_llm = AsyncMock()
        with patch("app.services.agents.patient.guard.call_qwen_chat", new=mock_llm):
            await update_ledger(m, "嗯", "嗯。")
        mock_llm.assert_not_called()


class TestRuleBasedUpdate:
    def test_token_match_marks_disclosed(self):
        m = _memory()
        _rule_based_update(m, "就是反酸烧心，晚上厉害。")
        assert m.find_fact("sym_002").status == "disclosed"
        assert m.find_fact("sym_001").status == "undisclosed"


class TestCheckContradiction:
    def test_denied_fact_reasserted_is_contradiction(self):
        m = _memory()
        m.mark(["his_001"], "denied")
        assert check_contradiction(m, "对，我青霉素过敏。") is True

    def test_denied_fact_with_negation_ok(self):
        m = _memory()
        m.mark(["his_001"], "denied")
        assert check_contradiction(m, "没有，我没有青霉素过敏。") is False

    def test_no_denied_facts_never_contradicts(self):
        m = _memory()
        assert check_contradiction(m, "青霉素过敏。") is False
