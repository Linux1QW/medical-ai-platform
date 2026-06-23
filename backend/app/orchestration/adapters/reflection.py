"""反思智能体适配器 — 用于 LangGraph 集成

注意：Reflection Agent 与其他评估 Agent 不同，它不是通过 Send fan-out 并行执行的，
而是作为评分后的验证步骤串行执行。因此本适配器主要用于测试和直接调用场景，
实际集成通过 graph.py 中的 reflection_check 节点函数完成。
"""

import logging

from app.core.config import settings
from app.orchestration.state import AgentResultEnvelope, EvaluationContext
from app.services.agents.reflection_agent import run_reflection

logger = logging.getLogger(__name__)


async def run_reflection_from_context(
    dimension_results: dict,
    total_score: float | None = None,
) -> dict:
    """从评估上下文运行反思评估

    这是 Reflection Agent 的主要入口点，由 graph.py 中的节点函数调用。

    Args:
        dimension_results: 各维度评估结果
        total_score: 总分

    Returns:
        dict 包含反思报告
    """
    if not settings.ENABLE_REACT_REFLECTION:
        logger.debug("Reflection agent disabled, returning empty result")
        return {
            "overall_quality": "acceptable",
            "confidence": 0.0,
            "issues_found": [],
            "consistency_score": 0.5,
            "evidence_adequacy_score": 0.5,
            "summary": "反思智能体未启用",
            "needs_review": False,
            "review_reasons": [],
            "react_trace": [],
            "react_steps_count": 0,
            "dimension_count": 0,
            "disabled": True,
        }

    return await run_reflection(
        dimension_results=dimension_results,
        total_score=total_score,
    )
