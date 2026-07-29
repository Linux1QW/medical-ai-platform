# -*- coding: utf-8 -*-
"""披露策略矩阵 — 人格×问诊阶段 决定回复风格提示

只产出 prompt 风格提示，不硬控制生成；未登记组合逐级回退，永不报错。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DisclosureStrategy:
    reply_length: str      # 极短 / 简短 / 正常
    volunteer_info: bool   # 是否允许少量主动补充
    ask_back: bool         # 是否倾向反问医生
    tone_hint: str         # 语气提示（直接注入 prompt）


_DEFAULTS: dict[str, DisclosureStrategy] = {
    "配合型": DisclosureStrategy("简短", False, False, "态度友好，回答清楚"),
    "焦虑型": DisclosureStrategy("简短", False, True, "语气担忧，容易往坏处想"),
    "沉默型": DisclosureStrategy("极短", False, False, "惜字如金，语气冷淡"),
    "对抗型": DisclosureStrategy("简短", False, False, "不耐烦，语气带刺但不拒答"),
}

# 仅登记与默认行为不同的组合，其余回退 _DEFAULTS
_STRATEGY_MATRIX: dict[tuple[str, str], DisclosureStrategy] = {
    ("配合型", "chief_complaint"): DisclosureStrategy("简短", False, False, "清楚说出主要不适，不展开细节"),
    ("配合型", "assessment_communication"): DisclosureStrategy("正常", True, True, "关心诊断结果，可主动确认注意事项"),
    ("焦虑型", "greeting"): DisclosureStrategy("简短", False, True, "寒暄中带焦虑，急于说病情"),
    ("焦虑型", "hpi"): DisclosureStrategy("简短", False, True, "描述症状时追问“是不是很严重”"),
    ("焦虑型", "assessment_communication"): DisclosureStrategy("正常", False, True, "反复确认风险，担心预后"),
    ("沉默型", "physical_exam"): DisclosureStrategy("极短", False, False, "配合检查但不多说一字"),
    ("对抗型", "greeting"): DisclosureStrategy("简短", False, False, "开场就显不耐烦"),
    ("对抗型", "past_history"): DisclosureStrategy("简短", False, True, "质疑“问这些有什么用”但仍回答"),
    ("对抗型", "assessment_communication"): DisclosureStrategy("正常", False, True, "对诊断持怀疑态度，追问依据"),
}

_FALLBACK = _DEFAULTS["配合型"]


def get_strategy(personality: str, stage: str) -> DisclosureStrategy:
    """查询人格×阶段策略；组合未登记回退人格默认，人格未知回退配合型"""
    return _STRATEGY_MATRIX.get((personality, stage)) or _DEFAULTS.get(personality, _FALLBACK)
