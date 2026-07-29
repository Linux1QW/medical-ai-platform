# -*- coding: utf-8 -*-
"""dynamics.py 单元测试：信任动力学与敏感事实解锁"""
from app.services.agents.patient.dynamics import (
    BEHAVIOR_TRUST_DELTA,
    INITIAL_TRUST,
    TRUST_UNLOCK_THRESHOLD,
    apply_turn_dynamics,
    initial_trust,
    locked_facts,
)
from app.services.agents.patient.memory import Fact, MemoryState


def _memory(trust=0.4):
    return MemoryState(trust=trust, facts=[
        Fact(fact_id="sym_001", content="上腹隐痛"),
        Fact(fact_id="lif_001", category="lifestyle", content="长期饮酒", disclosure_condition="empathy_unlock"),
        Fact(fact_id="lif_002", category="lifestyle", content="已披露隐私", disclosure_condition="empathy_unlock", status="disclosed"),
    ])


class TestTrustDynamics:
    def test_constants(self):
        assert INITIAL_TRUST == {"配合型": 0.7, "焦虑型": 0.5, "沉默型": 0.4, "对抗型": 0.2}
        assert BEHAVIOR_TRUST_DELTA == {"comfort": 0.10, "explain": 0.05, "instruction": 0.0, "ignore": -0.10}
        assert TRUST_UNLOCK_THRESHOLD == 0.5

    def test_initial_trust_fallback(self):
        assert initial_trust("配合型") == 0.7
        assert initial_trust("未知人格") == 0.5

    def test_comfort_raises_trust(self):
        m = _memory(trust=0.4)
        apply_turn_dynamics(m, "comfort")
        assert abs(m.trust - 0.5) < 1e-9

    def test_trust_clamped(self):
        m = _memory(trust=0.05)
        apply_turn_dynamics(m, "ignore")
        assert m.trust == 0.0
        m2 = _memory(trust=0.95)
        apply_turn_dynamics(m2, "comfort")
        assert m2.trust == 1.0


class TestLockedFacts:
    def test_low_trust_locks_undisclosed_sensitive(self):
        m = _memory(trust=0.3)
        assert [f.fact_id for f in locked_facts(m)] == ["lif_001"]

    def test_high_trust_unlocks_all(self):
        m = _memory(trust=0.6)
        assert locked_facts(m) == []
