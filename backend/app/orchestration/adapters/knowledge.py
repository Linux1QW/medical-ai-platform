"""医学知识核对 Agent 适配器"""

import logging

from app.core.config import settings
from app.orchestration.adapters.base import BaseAgentAdapter
from app.orchestration.state import AgentResultEnvelope, EvaluationContext
from app.services.agents.knowledge_agent import (
    run_knowledge_check,
    run_knowledge_check_react,
    run_knowledge_check_with_tools,
)

logger = logging.getLogger(__name__)


class KnowledgeAdapter(BaseAgentAdapter):
    agent_name = "knowledge"

    async def _call_agent(self, context: EvaluationContext) -> dict:
        patient_info = self._build_patient_info(context)
        doctor_diagnosis = context.doctor_diagnosis or ""
        treatment_plan = context.treatment_plan or ""

        if settings.ENABLE_REACT_KNOWLEDGE:
            logger.info("Using ReAct path for knowledge agent")
            consultation = {
                "conversation_text": context.conversation_text,
                "patient_info": patient_info,
                "doctor_diagnosis": doctor_diagnosis,
                "treatment_plan": treatment_plan,
            }
            try:
                return await run_knowledge_check_react(
                    consultation=consultation,
                    diagnosis_text=doctor_diagnosis,
                    treatment_text=treatment_plan,
                )
            except Exception as e:
                if settings.TOOL_USE_FALLBACK_TO_LEGACY:
                    logger.warning("Fallback to Tool Use knowledge check: %s", e)
                    return await run_knowledge_check_with_tools(
                        consultation=consultation,
                        diagnosis_text=doctor_diagnosis,
                        treatment_text=treatment_plan,
                    )
                else:
                    raise
        elif settings.ENABLE_TOOL_USE:
            logger.info("Using Tool Use path for knowledge agent")
            # 构建 consultation dict 供 run_knowledge_check_with_tools 使用
            consultation = {
                "conversation_text": context.conversation_text,
                "patient_info": patient_info,
                "doctor_diagnosis": doctor_diagnosis,
                "treatment_plan": treatment_plan,
            }
            try:
                return await run_knowledge_check_with_tools(
                    consultation=consultation,
                    diagnosis_text=doctor_diagnosis,
                    treatment_text=treatment_plan,
                )
            except Exception as e:
                if settings.TOOL_USE_FALLBACK_TO_LEGACY:
                    logger.warning("Fallback to legacy knowledge check: %s", e)
                    return await run_knowledge_check(
                        context.conversation_text,
                        patient_info,
                        doctor_diagnosis,
                        treatment_plan,
                        enable_hyde=True,
                    )
                else:
                    raise
        else:
            logger.info("Using legacy path for knowledge agent")
            return await run_knowledge_check(
                context.conversation_text,
                patient_info,
                doctor_diagnosis,
                treatment_plan,
                enable_hyde=True,
            )

    def _parse_result(self, raw: dict) -> AgentResultEnvelope:
        score = raw.get("score")
        analysis = raw.get("analysis", "")
        citations = raw.get("citations", [])
        human_review_needed = raw.get("human_review_needed", False)
        review_reason = raw.get("review_reason")
        rag_trace = raw.get("rag_trace", {})
        tool_trace = raw.get("tool_trace")

        # 分数范围校验
        if score is not None:
            score = max(0.0, min(100.0, float(score)))

        # 引用校验加固：score=null 且 review_reason 为引用校验失败时，强制 human_review_needed=True
        if score is None and review_reason == "citation_verification_failed":
            human_review_needed = True

        # 确定状态：score=None 表示拒答/证据不足
        if score is None:
            status = "insufficient"
        else:
            status = "success"

        # 构建 trace：整合 rag_trace 和 tool_trace
        trace = {"rag_trace": rag_trace}
        if tool_trace is not None:
            trace["tool_trace"] = tool_trace

        # 引用校验结果写入 trace
        if review_reason == "citation_verification_failed":
            trace["citation_verification"] = {
                "valid": False,
                "invalid_citation_ids": raw.get("invalid_citation_ids", []),
            }

        return AgentResultEnvelope(
            agent_name="knowledge",
            status=status,
            score=score,
            analysis=analysis,
            human_review_needed=human_review_needed,
            review_reason=review_reason,
            citations=citations,
            trace=trace,
        )
