# -*- coding: utf-8 -*-
"""memory.py 单元测试：披露账本模型与序列化"""
import pytest

from app.services.agents.patient.memory import Fact, MemoryState


def _make_memory():
    return MemoryState(facts=[
        Fact(fact_id="sym_001", category="symptom", content="上腹隐痛"),
        Fact(fact_id="his_001", category="history", content="十年前胃溃疡"),
    ])


class TestMemoryState:
    def test_json_roundtrip(self):
        m = _make_memory()
        m.turn = 3
        restored = MemoryState.from_json(m.to_json())
        assert restored is not None
        assert restored.turn == 3
        assert [f.fact_id for f in restored.facts] == ["sym_001", "his_001"]

    def test_from_json_invalid_returns_none(self):
        assert MemoryState.from_json(None) is None
        assert MemoryState.from_json("") is None
        assert MemoryState.from_json("{broken json") is None

    def test_mark_disclosed_records_turn(self):
        m = _make_memory()
        m.turn = 5
        m.mark(["sym_001"], "disclosed")
        fact = m.find_fact("sym_001")
        assert fact.status == "disclosed"
        assert fact.disclosed_at_turn == 5

    def test_mark_unknown_id_ignored(self):
        m = _make_memory()
        m.mark(["nonexistent"], "denied")  # 不抛异常
        assert all(f.status == "undisclosed" for f in m.facts)

    def test_facts_by_status(self):
        m = _make_memory()
        m.mark(["his_001"], "denied")
        assert [f.fact_id for f in m.facts_by_status("denied")] == ["his_001"]
        assert [f.fact_id for f in m.facts_by_status("undisclosed")] == ["sym_001"]

    def test_mark_denied_clears_disclosed_turn(self):
        """先披露后否认：disclosed_at_turn 应清除，避免与 status 矛盾"""
        m = _make_memory()
        m.turn = 3
        m.mark(["sym_001"], "disclosed")
        m.mark(["sym_001"], "denied")
        fact = m.find_fact("sym_001")
        assert fact.status == "denied"
        assert fact.disclosed_at_turn is None


from unittest.mock import AsyncMock, patch  # noqa: E402

from app.services.agents.patient.memory import _rule_based_facts, extract_facts  # noqa: E402


class TestExtractFacts:
    def test_rule_based_json_list_symptoms(self):
        facts = _rule_based_facts("头痛三天", "无特殊病史", '["头痛", "低热"]')
        contents = [f.content for f in facts]
        assert "头痛三天" in contents and "头痛" in contents and "低热" in contents
        assert all(f.status == "undisclosed" for f in facts)
        # "无特殊病史" 不应产生事实
        assert not [f for f in facts if f.category == "history"]

    def test_rule_based_plain_text_symptoms(self):
        facts = _rule_based_facts("", "十年前胃溃疡，青霉素过敏", "反酸，烧心")
        cats = {f.content: f.category for f in facts}
        assert cats["反酸"] == "symptom" and cats["烧心"] == "symptom"
        assert cats["十年前胃溃疡"] == "history" and cats["青霉素过敏"] == "history"

    @pytest.mark.asyncio
    async def test_extract_facts_llm_success(self):
        llm_out = '{"facts": [{"category": "symptom", "content": "上腹隐痛", "disclosure_condition": "direct_ask"}, {"category": "lifestyle", "content": "长期饮酒", "disclosure_condition": "empathy_unlock"}]}'
        with patch("app.services.agents.patient.memory.call_qwen_chat", new=AsyncMock(return_value=llm_out)):
            facts = await extract_facts("上腹痛", "无", "[]")
        assert [f.fact_id for f in facts] == ["sym_001", "lif_001"]
        assert facts[1].disclosure_condition == "empathy_unlock"

    @pytest.mark.asyncio
    async def test_extract_facts_skips_malformed_items(self):
        """facts 数组中混入非 dict 元素时跳过而非整体降级"""
        llm_out = '{"facts": ["脏数据", {"category": "symptom", "content": "上腹隐痛"}]}'
        with patch("app.services.agents.patient.memory.call_qwen_chat", new=AsyncMock(return_value=llm_out)):
            facts = await extract_facts("上腹痛", "无", "[]")
        assert [f.content for f in facts] == ["上腹隐痛"]

    @pytest.mark.asyncio
    async def test_extract_facts_llm_failure_falls_back(self):
        with patch("app.services.agents.patient.memory.call_qwen_chat", new=AsyncMock(side_effect=RuntimeError("boom"))):
            facts = await extract_facts("头痛三天", "无", '["头痛"]')
        assert [f.content for f in facts] == ["头痛三天", "头痛"]
