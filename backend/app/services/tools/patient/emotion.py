# -*- coding: utf-8 -*-
"""情绪引擎 — 医生行为四分类（与 humanistic_agent 评估侧同模型）+ 情绪状态转移

纯规则实现：医生安慰/解释会缓和情绪，忽视会恶化情绪，
情绪字符串注入患者 prompt 影响语气（由 PatientAgent 使用）。
"""
from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext

# 行为分类关键词（命中优先级：comfort > explain > instruction，均未命中为 ignore）
_BEHAVIOR_KEYWORDS = {
    "comfort": ("别担心", "不用担心", "别紧张", "别害怕", "放宽心", "不要焦虑", "慢慢说", "不严重", "一起想办法", "理解你"),
    "explain": ("原因是", "因为", "这个病", "意味着", "也就是说", "机制", "所以会", "解释", "通俗地讲"),
    "instruction": ("？", "?", "多久", "哪里", "什么时候", "有没有", "是不是", "量一下", "做个检查", "建议", "按时"),
}

# 情绪转移表：(当前情绪, 行为) -> 新情绪；未登记组合保持原情绪
_EMOTION_TRANSITION = {
    ("焦虑", "comfort"): "缓和",
    ("焦虑", "explain"): "缓和",
    ("焦虑", "ignore"): "恐慌",
    ("缓和", "ignore"): "焦虑",
    ("平静", "ignore"): "不满",
    ("平静", "comfort"): "安心",
    ("不满", "comfort"): "平静",
    ("不满", "explain"): "平静",
    ("不满", "ignore"): "愤怒",
    ("恐慌", "comfort"): "焦虑",
    ("愤怒", "comfort"): "不满",
}


def classify_doctor_behavior(message: str) -> str:
    """医生消息 -> comfort/explain/instruction/ignore（与评估侧 BEHAVIOR_TYPES 对齐）"""
    text = (message or "").strip()
    if not text:
        return "ignore"
    for behavior in ("comfort", "explain", "instruction"):
        if any(kw in text for kw in _BEHAVIOR_KEYWORDS[behavior]):
            return behavior
    return "ignore"


def update_emotion(current: str, behavior: str, personality: str) -> str:
    """查表转移情绪；未登记组合保持原情绪（personality 预留给后续差异化转移）"""
    return _EMOTION_TRANSITION.get((current, behavior), current)


class EmotionEngineArgs(BaseModel):
    doctor_message: str = Field(description="医生本轮消息")
    current_emotion: str = Field(description="患者当前情绪")
    personality: str = Field(description="患者人格类型")


class EmotionEngine(BaseTool):
    name = "emotion_engine"
    description = "分类医生行为并更新患者情绪状态"
    args_schema = EmotionEngineArgs
    timeout_seconds = 5
    critical = False

    async def execute(self, args: EmotionEngineArgs, context: ToolContext) -> dict:
        behavior = classify_doctor_behavior(args.doctor_message)
        emotion = update_emotion(args.current_emotion, behavior, args.personality)
        return {"behavior": behavior, "emotion": emotion}
