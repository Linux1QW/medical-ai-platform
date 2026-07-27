# -*- coding: utf-8 -*-
"""按 agent 角色的工具白名单策略（最小权限原则）

此前各 agent 通过 register_all_tools 拿到全部工具，隔离仅靠注册面收窄的
隐式约定。此处将白名单显式化：executor 在执行前强制校验，
LLM 侧也只下发角色允许的工具 schema，双层收紧。
"""

from app.core.config import settings

# knowledge 系列角色：检索 + 引用校验工具
_KNOWLEDGE_TOOLS = frozenset({
    "search_medical_kb",
    "expand_query",
    "generate_hyde_query",
    "rerank_evidence",
    "verify_citation",
})

# 各 agent 角色允许调用的工具集合
AGENT_TOOL_WHITELIST: dict[str, frozenset[str]] = {
    "knowledge_agent": _KNOWLEDGE_TOOLS,
    "knowledge_agent_react": _KNOWLEDGE_TOOLS,
    "reflection_agent": frozenset({
        "check_score_consistency",
        "check_evidence_sufficiency",
        "detect_score_contradictions",
        "summarize_evaluation",
    }),
}


def get_allowed_tools(agent_name: str) -> frozenset[str] | None:
    """返回 agent 角色允许的工具集合

    开关关闭或角色未登记时返回 None（不限制），保持向后兼容。
    """
    if not settings.TOOL_ROLE_WHITELIST_ENABLED:
        return None
    return AGENT_TOOL_WHITELIST.get(agent_name)
