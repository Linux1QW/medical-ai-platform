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
