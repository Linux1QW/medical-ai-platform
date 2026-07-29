# -*- coding: utf-8 -*-
"""一致性守卫 — 披露账本更新与矛盾检测

账本更新优先用一次低温 LLM 调用做对话状态跟踪，失败降级为规则匹配；
矛盾检测为纯规则（已否认事实在回复中被再次承认）。
"""
import logging

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_dict_from_text

from .memory import MemoryState

logger = logging.getLogger(__name__)

_NEGATION_WORDS = ("没有", "没", "不", "无", "从来没", "记不清", "不清楚")

_LEDGER_SYSTEM = (
    "你是医患对话状态跟踪器。给定患者档案事实清单与本轮对话，判断：\n"
    "1. disclosed：患者本轮回复中披露（承认/描述）了哪些事实\n"
    "2. denied：患者本轮回复中明确否认了哪些事实\n"
    '只输出 JSON：{"disclosed": ["fact_id"], "denied": ["fact_id"]}，没有则给空数组。'
)


def _split_tokens(text: str) -> list[str]:
    tokens = [text]
    for ch in "，,、;；。 ：:":
        tokens = [seg for tk in tokens for seg in tk.split(ch)]
    return [t.strip() for t in tokens if t.strip()]


async def update_ledger(memory: MemoryState, doctor_message: str, patient_reply: str) -> None:
    """回复后更新披露账本。LLM 判定失败时降级为规则匹配，绝不向上抛异常"""
    pending = memory.facts_by_status("undisclosed")
    if not pending:
        return
    fact_lines = "\n".join(f"{f.fact_id}: {f.content}" for f in pending)
    try:
        raw = await call_qwen_chat(
            [{"role": "system", "content": _LEDGER_SYSTEM},
             {"role": "user", "content": (
                 f"事实清单：\n{fact_lines}\n\n医生：{doctor_message}\n患者：{patient_reply}"
             )}],
            temperature=0.0, max_tokens=200,
        )
        data = extract_json_dict_from_text(raw)
        disclosed = [x for x in data.get("disclosed", []) if isinstance(x, str)]
        denied = [x for x in data.get("denied", []) if isinstance(x, str)]
        memory.mark(disclosed, "disclosed")
        memory.mark(denied, "denied")
    except Exception as e:
        logger.warning(f"账本 LLM 更新失败，降级为规则匹配: {e}")
        _rule_based_update(memory, patient_reply)


def _rule_based_update(memory: MemoryState, patient_reply: str) -> None:
    """规则兜底：事实内容的关键片段（≥2字）出现在回复中即视为已披露"""
    reply = patient_reply or ""
    for fact in memory.facts:
        if fact.status != "undisclosed":
            continue
        tokens = [t for t in _split_tokens(fact.content) if len(t) >= 2]
        if tokens and any(t in reply for t in tokens):
            fact.status = "disclosed"
            fact.disclosed_at_turn = memory.turn


def check_contradiction(memory: MemoryState, reply: str) -> bool:
    """检测回复是否承认了已否认（denied）的事实——即前后矛盾"""
    text = reply or ""
    for fact in memory.facts_by_status("denied"):
        tokens = [t for t in _split_tokens(fact.content) if len(t) >= 2]
        if not tokens:
            continue
        if any(t in text for t in tokens) and not any(neg in text for neg in _NEGATION_WORDS):
            return True
    return False
