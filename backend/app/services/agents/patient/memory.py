# -*- coding: utf-8 -*-
"""患者智能体记忆管理 — 披露账本（Disclosure Ledger）与会话记忆状态

三层记忆中的 L2 情节记忆：把"患者说过什么/否认过什么"从 LLM 软记忆
升级为结构化状态，保证长对话一致性。序列化后存于 consultations.memory_state。
"""
import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_dict_from_text

logger = logging.getLogger(__name__)

FactCategory = Literal["symptom", "history", "medication", "exam", "lifestyle"]
FactStatus = Literal["undisclosed", "disclosed", "denied"]
DisclosureCondition = Literal["direct_ask", "empathy_unlock", "never_volunteer"]


class Fact(BaseModel):
    """患者档案中的一条原子事实"""
    fact_id: str
    category: FactCategory = "symptom"
    content: str
    status: FactStatus = "undisclosed"
    disclosed_at_turn: Optional[int] = None
    disclosure_condition: DisclosureCondition = "direct_ask"


class MemoryState(BaseModel):
    """会话级记忆状态：披露账本 + 信任 + 情绪 + 问诊阶段"""
    facts: list[Fact] = Field(default_factory=list)
    trust: float = 0.5
    emotion: str = "平静"
    stage: str = "greeting"
    stage_history: list[str] = Field(default_factory=list)
    turn: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: Optional[str]) -> Optional["MemoryState"]:
        """解析持久化 JSON，失败返回 None（调用方走初始化路径）"""
        if not raw:
            return None
        try:
            return cls.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"memory_state 解析失败，将重新初始化: {e}")
            return None

    def facts_by_status(self, status: FactStatus) -> list[Fact]:
        return [f for f in self.facts if f.status == status]

    def find_fact(self, fact_id: str) -> Optional[Fact]:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        return None

    def mark(self, fact_ids: list[str], status: FactStatus) -> None:
        """批量更新事实状态；置为 disclosed 时记录披露轮次，否则清除。未知 id 静默忽略"""
        for fid in fact_ids:
            fact = self.find_fact(fid)
            if fact is None:
                continue
            fact.status = status
            fact.disclosed_at_turn = self.turn if status == "disclosed" else None


# ── 事实抽取 ─────────────────────────────────────────────────────────────────────

_SPLIT_CHARS = "，,、;；\n"
_EMPTY_MARKERS = ("无", "无特殊病史", "无特殊", "没有")

_CATEGORY_PREFIX = {"symptom": "sym", "history": "his", "medication": "med", "exam": "exm", "lifestyle": "lif"}

_FACT_EXTRACT_SYSTEM = (
    "你是医学病历结构化助手。请把患者档案拆分为原子事实列表，每条事实只含一个信息点。\n"
    "category 取值：symptom(症状)/history(病史)/medication(用药)/exam(检查)/lifestyle(生活史)。\n"
    "disclosure_condition 取值：direct_ask(被直接问到即回答)/"
    "empathy_unlock(敏感隐私信息，需医生共情建立信任后才愿意说)/never_volunteer(绝不主动提)。\n"
    '只输出 JSON：{"facts": [{"category": "...", "content": "...", "disclosure_condition": "..."}]}'
)


def _split_items(text: str) -> list[str]:
    """按中文/英文标点拆分为条目，过滤空值与'无'类占位"""
    items = [text]
    for ch in _SPLIT_CHARS:
        items = [seg for item in items for seg in item.split(ch)]
    return [s.strip() for s in items if s.strip() and s.strip() not in _EMPTY_MARKERS]


def _rule_based_facts(chief_complaint: str, medical_history: str, symptoms_raw: str) -> list[Fact]:
    """规则兜底：主诉一条 + 症状逐项 + 病史逐句"""
    facts: list[Fact] = []
    symptom_items: list[str] = []
    try:
        parsed = json.loads(symptoms_raw or "[]")
        if isinstance(parsed, list):
            symptom_items = [str(x).strip() for x in parsed if str(x).strip()]
        elif isinstance(parsed, dict):
            symptom_items = [f"{k}: {v}" for k, v in parsed.items()]
        else:
            symptom_items = _split_items(str(parsed))
    except (json.JSONDecodeError, TypeError):
        symptom_items = _split_items(symptoms_raw or "")

    if chief_complaint and chief_complaint.strip():
        facts.append(Fact(fact_id="sym_000", category="symptom", content=chief_complaint.strip()))
    for i, item in enumerate(symptom_items, start=1):
        facts.append(Fact(fact_id=f"sym_{i:03d}", category="symptom", content=item))
    for i, item in enumerate(_split_items(medical_history or ""), start=1):
        facts.append(Fact(fact_id=f"his_{i:03d}", category="history", content=item))
    return facts


async def extract_facts(chief_complaint: str, medical_history: str, symptoms_raw: str) -> list[Fact]:
    """患者档案 → 原子事实列表。优先 LLM 结构化抽取，失败降级为规则拆分"""
    profile = f"主诉：{chief_complaint}\n病史：{medical_history}\n症状：{symptoms_raw}"
    try:
        raw = await call_qwen_chat(
            [{"role": "system", "content": _FACT_EXTRACT_SYSTEM},
             {"role": "user", "content": profile}],
            temperature=0.1, max_tokens=1500,
        )
        data = extract_json_dict_from_text(raw)
        facts: list[Fact] = []
        counters: dict[str, int] = {}
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            category = item.get("category", "symptom")
            if category not in _CATEGORY_PREFIX:
                category = "symptom"
            condition = item.get("disclosure_condition", "direct_ask")
            if condition not in ("direct_ask", "empathy_unlock", "never_volunteer"):
                condition = "direct_ask"
            counters[category] = counters.get(category, 0) + 1
            facts.append(Fact(
                fact_id=f"{_CATEGORY_PREFIX[category]}_{counters[category]:03d}",
                category=category, content=content, disclosure_condition=condition,
            ))
        if facts:
            return facts
        logger.warning("LLM 事实抽取结果为空，降级为规则拆分")
    except Exception as e:
        logger.warning(f"LLM 事实抽取失败，降级为规则拆分: {e}")
    return _rule_based_facts(chief_complaint, medical_history, symptoms_raw)
