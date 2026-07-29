# -*- coding: utf-8 -*-
"""信任动力学 — 医生行为驱动患者信任度变化，信任解锁敏感事实

行为四分类与 humanistic_agent 评估侧同模型：医生安慰/解释提升信任，
忽视降低信任；trust >= 阈值时 empathy_unlock 事实才允许披露。
"""
from .memory import Fact, MemoryState

INITIAL_TRUST: dict[str, float] = {"配合型": 0.7, "焦虑型": 0.5, "沉默型": 0.4, "对抗型": 0.2}
BEHAVIOR_TRUST_DELTA: dict[str, float] = {"comfort": 0.10, "explain": 0.05, "instruction": 0.0, "ignore": -0.10}
TRUST_UNLOCK_THRESHOLD = 0.5


def initial_trust(personality: str) -> float:
    """人格 -> 初始信任度；未知人格取中性 0.5"""
    return INITIAL_TRUST.get(personality, 0.5)


def apply_turn_dynamics(memory: MemoryState, behavior: str) -> None:
    """按医生行为更新信任度，限幅 [0, 1]"""
    delta = BEHAVIOR_TRUST_DELTA.get(behavior, 0.0)
    memory.trust = max(0.0, min(1.0, memory.trust + delta))


def locked_facts(memory: MemoryState) -> list[Fact]:
    """信任不足时仍锁定的敏感事实（empathy_unlock 且未披露）"""
    if memory.trust >= TRUST_UNLOCK_THRESHOLD:
        return []
    return [
        f for f in memory.facts
        if f.disclosure_condition == "empathy_unlock" and f.status == "undisclosed"
    ]
