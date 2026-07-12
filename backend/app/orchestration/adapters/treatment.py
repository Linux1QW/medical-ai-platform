"""治疗方案评估 Agent 适配器"""

import logging

from app.orchestration.adapters.base import BaseAgentAdapter
from app.orchestration.state import AgentResultEnvelope, EvaluationContext
from app.services.agents.treatment_agent import run_treatment_evaluation
from app.utils.json_parser import extract_json_from_text

logger = logging.getLogger(__name__)


class TreatmentAdapter(BaseAgentAdapter):
    agent_name = "treatment"

    async def _call_agent(self, context: EvaluationContext) -> dict:
        patient_info = self._build_patient_info(context)
        doctor_diagnosis = context.doctor_diagnosis or ""
        treatment_plan = context.treatment_plan or ""
        knowledge_citations = getattr(context, "knowledge_citations", None) or None
        return await run_treatment_evaluation(
            context.conversation_text,
            patient_info,
            doctor_diagnosis,
            treatment_plan,
            knowledge_citations=knowledge_citations,
        )

    def _parse_result(self, raw: dict) -> AgentResultEnvelope:
        raw_response = raw.get("raw_response", "")
        data = extract_json_from_text(
            raw_response, default={}, raise_on_failure=False
        ) if isinstance(raw_response, str) else raw_response

        score = data.get("score")
        analysis = data.get("analysis", "")

        # 分数范围校验
        if score is not None:
            score = max(0.0, min(100.0, float(score)))

        return AgentResultEnvelope(
            agent_name="treatment",
            status="success",
            score=score,
            analysis=analysis,
        )
