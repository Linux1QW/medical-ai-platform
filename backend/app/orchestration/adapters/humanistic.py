"""人文关怀评估 Agent 适配器"""

import json
import logging

from app.orchestration.adapters.base import BaseAgentAdapter
from app.orchestration.state import AgentResultEnvelope, EvaluationContext
from app.services.agents.humanistic_agent import run_humanistic_evaluation

logger = logging.getLogger(__name__)


class HumanisticAdapter(BaseAgentAdapter):
    agent_name = "humanistic"

    async def _call_agent(self, context: EvaluationContext) -> dict:
        patient_info = self._build_patient_info(context)
        return await run_humanistic_evaluation(context.conversation_text, patient_info)

    def _parse_result(self, raw: dict) -> AgentResultEnvelope:
        raw_response = raw.get("raw_response", "")
        try:
            data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        except json.JSONDecodeError:
            logger.warning("humanistic_agent raw_response JSON 解析失败，使用默认值")
            data = {}

        score = data.get("score")
        analysis = data.get("analysis", "")

        # 分数范围校验
        if score is not None:
            score = max(0.0, min(100.0, float(score)))

        return AgentResultEnvelope(
            agent_name="humanistic",
            status="success",
            score=score,
            analysis=analysis,
            trace={"details": data.get("details", {})},
        )
