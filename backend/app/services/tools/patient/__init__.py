# -*- coding: utf-8 -*-
"""患者智能体专属工具包 — 注册函数与预算配置"""
from app.services.tools.registry import ToolRegistry

from .emotion import EmotionEngine, classify_doctor_behavior, update_emotion
from .physiology import PhysiologyCalculator
from .plausible_symptom import QueryPlausibleSymptom

# 会话级调用预算：未登记的工具不限次（本地确定性工具无需限制）
PATIENT_TOOL_BUDGETS: dict[str, int] = {"query_plausible_symptom": 5}


def register_patient_tools(registry: ToolRegistry) -> None:
    """注册患者专属工具（幂等）"""
    registry.register(QueryPlausibleSymptom())
    registry.register(PhysiologyCalculator())
    registry.register(EmotionEngine())


__all__ = [
    "EmotionEngine",
    "PATIENT_TOOL_BUDGETS",
    "PhysiologyCalculator",
    "QueryPlausibleSymptom",
    "classify_doctor_behavior",
    "register_patient_tools",
    "update_emotion",
]
