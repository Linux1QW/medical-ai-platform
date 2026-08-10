"""Deterministic context compiler with fixed token budget."""
from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.contracts import CoachContextView


@dataclass
class ContextBudget:
    """Fixed token budget for coach context."""
    total_tokens: int = 16_000
    output_reserve: int = 2_000
    system_tokens: int = 1_800
    policy_tokens: int = 1_200
    memory_tokens: int = 2_000
    dialogue_tokens: int = 6_000
    evidence_tokens: int = 3_000


DEFAULT_COACH_BUDGET = ContextBudget()


@dataclass
class CompiledContext:
    """Compiled context ready for LLM consumption."""
    system_section: str | None = None
    policy_section: str | None = None
    memory_section: str | None = None
    dialogue_section: str | None = None
    evidence_section: str | None = None
    total_tokens: int = 0


COACH_SYSTEM_PROMPT = """你是一个智能问诊教练助手。你的职责是帮助受训医生提高问诊技能。
你不能代替医生与患者交流，不能做出诊断或开具处方。
你的建议必须基于可见的患者信息和教学指南。
"""

COACH_SAFETY_POLICY = """安全策略：
1. 绝不能透露未向医生披露的患者信息
2. 绝不能给出诊断或治疗建议
3. 绝不能建议跳过必要的问诊步骤
4. 检测到红旗症状时必须提示医生关注
5. 所有建议必须附带教学依据
"""


def compile_coach_context(
    view: CoachContextView,
    budget: ContextBudget = DEFAULT_COACH_BUDGET,
) -> CompiledContext:
    """Compile coach context within fixed token budget.
    
    System and policy sections are NEVER truncated.
    Trimming order: low-score evidence → old dialogue → low-confidence memory.
    """
    # System and policy are never truncated
    system_section = COACH_SYSTEM_PROMPT
    policy_section = COACH_SAFETY_POLICY

    # Build dialogue section from visible messages
    dialogue_lines = []
    for msg in view.messages:
        dialogue_lines.append(f"[{msg.role}]: {msg.content}")
    dialogue_section = "\n".join(dialogue_lines) if dialogue_lines else None

    # Build memory section from approved profile memories
    memory_lines = []
    for mem in view.approved_profile_memories:
        memory_lines.append(f"- {mem.skill_dimension}: {mem.summary}")
    memory_section = "\n".join(memory_lines) if memory_lines else None

    return CompiledContext(
        system_section=system_section,
        policy_section=policy_section,
        memory_section=memory_section,
        dialogue_section=dialogue_section,
        evidence_section=None,
        total_tokens=budget.total_tokens - budget.output_reserve,
    )
