# -*- coding: utf-8 -*-
"""患者智能体编排 — 账本注入、阶段跟踪、矛盾重生成

每次请求新建实例，全部会话状态存于传入的 MemoryState（由调用方持久化）。
respond 内部失败会向上抛出，由 consultation_service 回退旧无记忆路径。
"""
import logging
import uuid

from app.core.config import settings
from app.services.qwen_client import call_qwen_chat, call_qwen_with_tools
from app.services.tools import (
    PATIENT_TOOL_BUDGETS,
    ToolContext,
    ToolExecutorBridge,
    ToolRegistry,
    create_tool_budget,
    create_tool_executor,
    get_allowed_tools,
    register_patient_tools,
)
from app.services.tools.patient.emotion import classify_doctor_behavior, update_emotion

from .dynamics import apply_turn_dynamics, initial_trust, locked_facts
from .guard import check_contradiction, update_ledger
from .memory import MemoryState
from .planner import classify_stage
from .prompts import PATIENT_ROLE_WRAPPER, PATIENT_TOOL_GUIDE
from .strategy import get_strategy

logger = logging.getLogger(__name__)

# 主回复只下发两个工具 schema：emotion_engine 由 respond 内纯函数驱动，不下发避免情绪双写
_PATIENT_TOOL_SCHEMAS = ["physiology_calculator", "query_plausible_symptom"]

_REGEN_INSTRUCTION = (
    "注意：你上一版回复与你此前已否认的信息矛盾。"
    "请重新回答医生的问题，绝对不能承认下面列出的【已否认信息】。"
)


class PatientAgent:
    """基于披露账本的患者回复生成器"""

    def __init__(self, patient, memory: MemoryState, consultation_id: int | None = None):
        self.patient = patient
        self.memory = memory
        self.consultation_id = consultation_id
        # 首轮按人格初始化信任度（后续轮次沿用持久化值）
        if memory.turn == 0:
            memory.trust = initial_trust(patient.personality_type or "")

    def _build_system_prompt(self) -> str:
        """角色包装 + 披露账本注入（已披露保持一致 / 已否认绝不翻供）"""
        sections = [PATIENT_ROLE_WRAPPER.format(system_prompt=self.patient.system_prompt or "")]
        sections.append(f"【当前情绪状态：{self.memory.emotion}】请在语气中自然体现这种情绪。")
        locked = locked_facts(self.memory)
        if locked:
            lines = "\n".join(f"- {f.content}" for f in locked)
            sections.append(
                "【你暂时不愿意透露的隐私信息】（对医生信任不够，被问到时含糊回避，"
                "如“这个……不好说”；若医生安慰共情你，后续可以坦白）\n" + lines
            )
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
        strategy = get_strategy(self.patient.personality_type or "", self.memory.stage)
        sections.append(
            f"【本轮回复风格】长度：{strategy.reply_length}；语气：{strategy.tone_hint}"
            + ("；可少量主动补充相关信息" if strategy.volunteer_info else "")
            + ("；可向医生反问一个问题" if strategy.ask_back else "")
        )
        return "\n\n".join(sections)

    async def _respond_with_tools(self, messages: list[dict]) -> str | None:
        """LLM 自主 function-calling 主回复；任何降级返回 None，由调用方回退纯文本路径"""
        try:
            context = ToolContext(
                run_id=str(uuid.uuid4()),
                agent_name="patient_agent",
                budgets=dict(PATIENT_TOOL_BUDGETS),
                allowed_tools=get_allowed_tools("patient_agent"),
                extras={
                    "consultation_id": self.consultation_id or 0,
                    "diagnosis": getattr(self.patient, "expected_diagnosis", "") or "",
                },
            )
            registry = ToolRegistry()
            register_patient_tools(registry)
            executor = create_tool_executor(registry, max_result_chars=settings.TOOL_USE_MAX_RESULT_CHARS)
            budget = create_tool_budget(context.budgets, context.run_id or "")
            bridge = ToolExecutorBridge(executor, context, budget)
            tool_schemas = registry.get_openai_schemas(_PATIENT_TOOL_SCHEMAS)

            # system prompt 追加工具使用指引（不污染原 messages，回退路径不带指引）
            tool_messages = [dict(messages[0]), *messages[1:]]
            tool_messages[0]["content"] += "\n\n" + PATIENT_TOOL_GUIDE

            result = await call_qwen_with_tools(
                tool_messages,
                tools=tool_schemas,
                tool_executor=bridge,
                temperature=0.3,
                max_tool_rounds=settings.PATIENT_TOOL_MAX_ROUNDS,
                max_tool_calls=settings.PATIENT_TOOL_MAX_CALLS,
            )
            if result.tool_calls:
                summary = "; ".join(
                    f"{t.tool_name}:{t.status}:{round(t.elapsed_ms)}ms" for t in result.tool_calls
                )
                logger.info(f"[PatientToolUse] 工具调用: {summary}")
            if result.degraded or not (result.content or "").strip():
                logger.warning(f"[PatientToolUse] 工具路径降级，回退纯文本路径: {result.error}")
                return None
            return result.content
        except Exception as e:
            logger.warning(f"[PatientToolUse] 工具路径异常，回退纯文本路径: {e}", exc_info=True)
            return None

    async def respond(self, doctor_message: str, chat_history: list[dict]) -> str:
        """生成一条患者回复并更新记忆状态"""
        stage = classify_stage(doctor_message, self.memory.stage)

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": doctor_message})

        reply = None
        if settings.ENABLE_PATIENT_TOOL_USE:
            reply = await self._respond_with_tools(messages)
        if reply is None:
            reply = await call_qwen_chat(messages, temperature=0.3)

        # 首次 LLM 调用成功后才落记忆状态，异常向上抛时不留幽灵轮次
        self.memory.turn += 1
        self.memory.stage = stage
        self.memory.stage_history.append(stage)

        # 情绪前置路由：行为分类 + 情绪转移（纯规则，零成本）
        behavior = classify_doctor_behavior(doctor_message)
        self.memory.emotion = update_emotion(
            self.memory.emotion, behavior, self.patient.personality_type or ""
        )
        apply_turn_dynamics(self.memory, behavior)

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
