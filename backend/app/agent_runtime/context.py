"""Deterministic context compiler with fixed token budget."""
from __future__ import annotations

from dataclasses import dataclass

from app.agent_runtime.contracts import CoachContextView


class ContextBudgetExceeded(Exception):
    """Raised when mandatory sections alone exceed the token budget."""

    def __init__(self, mandatory_tokens: int, budget_limit: int) -> None:
        self.mandatory_tokens = mandatory_tokens
        self.budget_limit = budget_limit
        super().__init__(
            f"Mandatory sections require {mandatory_tokens} tokens "
            f"but budget limit is {budget_limit}"
        )


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


def _count_tokens(text: str) -> int:
    """Estimate token count for Chinese/mixed text.

    Uses a simple heuristic: ~1.5 tokens per Chinese character,
    ~0.75 tokens per English word. For a conservative estimate,
    we use len(text) * 0.6 which approximates mixed CJK/English content.
    Minimum 1 token for any non-empty string.
    """
    if not text:
        return 0
    # Rough but deterministic token estimation for mixed CJK/English
    # Chinese chars count as ~1.5 tokens, ASCII words ~0.25 tokens/char
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    tokens = int(chinese_chars * 1.5 + other_chars * 0.4)
    return max(1, tokens)


def compile_coach_context(
    view: CoachContextView,
    budget: ContextBudget = DEFAULT_COACH_BUDGET,
) -> CompiledContext:
    """Compile coach context within fixed token budget.

    System and policy sections are NEVER truncated.
    Trimming order:
      1. Low-score evidence (removed first)
      2. Oldest dialogue turns
      3. Lowest-confidence memory entries
    If mandatory sections alone exceed budget → raise ContextBudgetExceeded.
    input_tokens <= limit - output_reserve
    """
    input_limit = budget.total_tokens - budget.output_reserve

    # System and policy are never truncated
    system_section = COACH_SYSTEM_PROMPT
    policy_section = COACH_SAFETY_POLICY
    system_tokens = _count_tokens(system_section)
    policy_tokens = _count_tokens(policy_section)

    mandatory_tokens = system_tokens + policy_tokens
    if mandatory_tokens > input_limit:
        raise ContextBudgetExceeded(
            mandatory_tokens=mandatory_tokens,
            budget_limit=input_limit,
        )

    remaining_budget = input_limit - mandatory_tokens

    # Build memory section from approved profile memories
    memory_entries_with_tokens: list[tuple[str, int, int]] = []  # (line, tokens, idx)
    for idx, mem in enumerate(view.approved_profile_memories):
        line = f"- {mem.skill_dimension}: {mem.summary}"
        tok = _count_tokens(line)
        memory_entries_with_tokens.append((line, tok, idx))

    # Fit memories into budget (respecting memory_tokens cap)
    memory_budget = min(budget.memory_tokens, remaining_budget)
    selected_memories: list[str] = []
    used_memory_tokens = 0
    for line, tok, _idx in memory_entries_with_tokens:
        if used_memory_tokens + tok <= memory_budget:
            selected_memories.append(line)
            used_memory_tokens += tok
        # else: skip lowest-confidence (already sorted by order, last = lowest priority)

    memory_section = "\n".join(selected_memories) if selected_memories else None
    remaining_budget -= used_memory_tokens

    # Build dialogue section from visible messages
    dialogue_entries_with_tokens: list[tuple[str, int, int]] = []  # (line, tokens, seq)
    for msg in view.messages:
        line = f"[{msg.role}]: {msg.content}"
        tok = _count_tokens(line)
        dialogue_entries_with_tokens.append((line, tok, msg.sequence))

    # Fit dialogue into budget (respecting dialogue_tokens cap)
    dialogue_budget = min(budget.dialogue_tokens, remaining_budget)
    # Keep latest turns first (highest sequence), drop oldest
    dialogue_budget_remaining = dialogue_budget
    selected_dialogue: list[tuple[int, str]] = []  # (sequence, line)
    for line, tok, seq in reversed(dialogue_entries_with_tokens):
        if dialogue_budget_remaining - tok >= 0:
            selected_dialogue.append((seq, line))
            dialogue_budget_remaining -= tok
        else:
            break

    # Re-sort by sequence to maintain chronological order
    selected_dialogue.sort(key=lambda x: x[0])
    dialogue_lines = [line for _, line in selected_dialogue]
    dialogue_section = "\n".join(dialogue_lines) if dialogue_lines else None
    used_dialogue_tokens = dialogue_budget - dialogue_budget_remaining
    remaining_budget -= used_dialogue_tokens

    total_tokens = system_tokens + policy_tokens + used_memory_tokens + used_dialogue_tokens

    return CompiledContext(
        system_section=system_section,
        policy_section=policy_section,
        memory_section=memory_section,
        dialogue_section=dialogue_section,
        evidence_section=None,
        total_tokens=total_tokens,
    )
