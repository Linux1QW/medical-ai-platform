"""建议指导智能体适配器 — 用于 LangGraph 集成

注意：Suggestion Agent 与其他评估 Agent 不同，它不是通过 Send fan-out 并行执行的，
而是作为评分和反思后的改进建议生成步骤串行执行。因此本适配器主要用于测试和直接调用场景，
实际集成通过 graph.py 中的 generate_suggestion 节点函数完成。
"""

import logging

from app.core.config import settings
from app.services.agents.suggestion_agent import run_suggestion

logger = logging.getLogger(__name__)


async def run_suggestion_from_context(
    conversation_text: str,
    patient_info: str,
    inquiry_result: str,
    knowledge_result: str,
    humanistic_result: str,
) -> dict:
    """从评估上下文运行建议生成

    这是 Suggestion Agent 的主要入口点，由 graph.py 中的 generate_suggestion 节点调用。

    Args:
        conversation_text: 问诊对话记录
        patient_info: 患者基本信息
        inquiry_result: 问诊分析评估结果
        knowledge_result: 医学知识评估结果
        humanistic_result: 人文关怀评估结果

    Returns:
        dict 包含 raw_response（JSON字符串），解析后包括：
            - suggestions: 格式化的建议文本
            - missing_questions: 缺失的关键问题列表
            - improvement_suggestions: 改进措施列表
            - priority: 优先级 (high/middle/low)
            - ideal_inquiry_summary: 理想问诊摘要
    """
    if not settings.ENABLE_LLM_SUGGESTION:
        logger.debug("LLM suggestion disabled, returning empty result")
        return {
            "raw_response": '{"suggestions": "LLM 建议生成未启用", "missing_questions": [], "improvement_suggestions": [], "priority": "middle", "ideal_inquiry_summary": ""}',
            "disabled": True,
        }

    return await run_suggestion(
        conversation_text=conversation_text,
        patient_info=patient_info,
        inquiry_result=inquiry_result,
        knowledge_result=knowledge_result,
        humanistic_result=humanistic_result,
    )
