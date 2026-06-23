"""治疗方案评估 Agent 适配器"""

import json
import re
import logging

from app.orchestration.adapters.base import BaseAgentAdapter
from app.orchestration.state import AgentResultEnvelope, EvaluationContext
from app.services.agents.treatment_agent import run_treatment_evaluation

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


class TreatmentAdapter(BaseAgentAdapter):
    agent_name = "treatment"

    async def _call_agent(self, context: EvaluationContext) -> dict:
        patient_info = self._build_patient_info(context)
        doctor_diagnosis = context.doctor_diagnosis or ""
        treatment_plan = context.treatment_plan or ""
        return await run_treatment_evaluation(
            context.conversation_text, patient_info, doctor_diagnosis, treatment_plan
        )

    def _parse_result(self, raw: dict) -> AgentResultEnvelope:
        raw_response = raw.get("raw_response", "")
        data = _extract_json_from_text(raw_response) if isinstance(raw_response, str) else raw_response

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
