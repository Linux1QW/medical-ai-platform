# -*- coding: utf-8 -*-
"""患者智能体编排 — 账本注入、阶段跟踪、矛盾重生成

每次请求新建实例，全部会话状态存于传入的 MemoryState（由调用方持久化）。
respond 内部失败会向上抛出，由 consultation_service 回退旧无记忆路径。
"""
import logging

from app.services.qwen_client import call_qwen_chat

from .guard import check_contradiction, update_ledger
from .memory import MemoryState
from .planner import classify_stage
from .prompts import PATIENT_ROLE_WRAPPER

logger = logging.getLogger(__name__)

_REGEN_INSTRUCTION = (
    "注意：你上一版回复与你此前已否认的信息矛盾。"
    "请重新回答医生的问题，绝对不能承认下面列出的【已否认信息】。"
)


class PatientAgent:
    """基于披露账本的患者回复生成器"""

    def __init__(self, patient, memory: MemoryState):
        self.patient = patient
        self.memory = memory

    def _build_system_prompt(self) -> str:
        """角色包装 + 披露账本注入（已披露保持一致 / 已否认绝不翻供）"""
        sections = [PATIENT_ROLE_WRAPPER.format(system_prompt=self.patient.system_prompt or "")]
        disclosed = self.memory.facts_by_status("disclosed")
        if disclosed:
            lines = "\n".join(f"- {f.content}" for f in disclosed)
            sections.append(
                "【你已经告诉过医生的信息】（再被问到时保持说法一致，不要当作新信息重复展开）\n" + lines
            )
        denied = self.memory.facts_by_status("denied")
        if denied:
            lines = "\n".join(f"- {f.content}" for f in denied)
            sections.append("【你已明确否认过的信息（绝对不能再承认）】\n" + lines)
        return "\n\n".join(sections)

    async def respond(self, doctor_message: str, chat_history: list[dict]) -> str:
        """生成一条患者回复并更新记忆状态"""
        self.memory.turn += 1
        stage = classify_stage(doctor_message, self.memory.stage)
        self.memory.stage = stage
        self.memory.stage_history.append(stage)

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": doctor_message})

        reply = await call_qwen_chat(messages, temperature=0.3)

        if check_contradiction(self.memory, reply):
            logger.warning("患者回复与已否认事实矛盾，触发一次重生成")
            denied_lines = "\n".join(f"- {f.content}" for f in self.memory.facts_by_status("denied"))
            retry_messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"{_REGEN_INSTRUCTION}\n【已否认信息】\n{denied_lines}\n\n"
                    f"医生刚才的问题是：{doctor_message}"
                )},
            ]
            regenerated = await call_qwen_chat(retry_messages, temperature=0.2)
            if not check_contradiction(self.memory, regenerated):
                reply = regenerated

        await update_ledger(self.memory, doctor_message, reply)
        return reply
