"""Rules-first + LLM fallback intent classification."""
from __future__ import annotations

from typing import Callable, Literal, get_args

from app.agent_runtime.contracts import CoachIntent

# Keyword rules for fast-path classification
INTENT_RULES: list[tuple[list[str], CoachIntent]] = [
    # (keywords, intent) — order matters, first match wins
    (["你好", "早上好", "请坐", "您好"], "rapport"),
    (["哪里不舒服", "什么症状", "主要原因", "来看"], "chief_complaint"),
    (["什么时候开始", "发病", " onset"], "hpi_onset"),
    (["加重", "减轻", "变化", "进展"], "hpi_progression"),
    (["严重", "剧烈", "难以忍受", "程度"], "hpi_severity"),
    (["持续", "间断", "频率", "多久"], "hpi_timing"),
    (["伴随", "还有", "同时", "其他症状"], "hpi_associated"),
    (["以前得过", "既往", "手术", "住院", "病史"], "past_history"),
    (["吃药", "用药", "药物", "服用"], "medication"),
    (["过敏", "皮疹", "药物过敏"], "allergy"),
    (["家人", "家属", "遗传", "吸烟", "喝酒"], "family_social"),
    (["检查", "化验", "体征", "血压", "体温"], "examination"),
    (["诊断", "病情", "结果", "解释"], "assessment_communication"),
    (["治疗", "方案", "用药", "手术"], "treatment_communication"),
    (["还有问题", "其他", "再见", "结束"], "closing"),
]


def classify_intent(
    message: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> CoachIntent:
    """Classify intent from a message.

    Strategy: rules-first (keyword match) → LLM fallback (temperature=0).
    If no rule matches and no LLM provided, default to "off_topic".
    """
    # Rules-first pass
    msg_lower = message.lower()
    for keywords, intent in INTENT_RULES:
        if any(kw in msg_lower for kw in keywords):
            return intent

    # LLM fallback
    if llm_fn is not None:
        try:
            result = llm_fn(message)
            if result in get_args(CoachIntent):
                return result
        except Exception:
            pass

    return "off_topic"
