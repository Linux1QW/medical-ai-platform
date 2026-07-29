# -*- coding: utf-8 -*-
"""患者智能体记忆管理 — 披露账本（Disclosure Ledger）与会话记忆状态

三层记忆中的 L2 情节记忆：把"患者说过什么/否认过什么"从 LLM 软记忆
升级为结构化状态，保证长对话一致性。序列化后存于 consultations.memory_state。
"""
import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

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
        """批量更新事实状态；置为 disclosed 时记录披露轮次。未知 id 静默忽略"""
        for fid in fact_ids:
            fact = self.find_fact(fid)
            if fact is None:
                continue
            fact.status = status
            if status == "disclosed":
                fact.disclosed_at_turn = self.turn
