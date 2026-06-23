"""Agent 适配器基类"""

import json
import logging
from abc import ABC, abstractmethod

from app.orchestration.state import AgentResultEnvelope, EvaluationContext

logger = logging.getLogger(__name__)


class BaseAgentAdapter(ABC):
    """Agent 适配器基类

    子类实现 _call_agent() 和 _parse_result()。
    run() 负责统一异常处理、JSON 解析和 Pydantic 校验。
    """

    agent_name: str = ""

    async def run(self, context: EvaluationContext) -> AgentResultEnvelope:
        """统一执行入口：调用 Agent → 解析 → 校验 → 返回 Envelope"""
        try:
            raw = await self._call_agent(context)
            return self._parse_result(raw)
        except Exception as e:
            logger.error(f"Agent {self.agent_name} failed: {e}", exc_info=True)
            return AgentResultEnvelope(
                agent_name=self.agent_name,
                status="error",
                analysis=f"Agent执行异常: {type(e).__name__}",
                human_review_needed=True,
                review_reason=f"{self.agent_name}_error: {str(e)[:200]}",
            )

    @abstractmethod
    async def _call_agent(self, context: EvaluationContext) -> dict:
        """调用底层 Agent 函数，返回原始 dict"""
        ...

    @abstractmethod
    def _parse_result(self, raw: dict) -> AgentResultEnvelope:
        """将原始 dict 转换为 AgentResultEnvelope"""
        ...

    def _build_patient_info(self, context: EvaluationContext) -> str:
        """构建 patient_info 字符串（兼容现有 Agent 接口）"""
        parts = []
        if context.patient_age is not None:
            parts.append(f"年龄:{context.patient_age}岁")
        if context.patient_gender:
            parts.append(f"性别:{context.patient_gender}")
        if context.chief_complaint:
            parts.append(f"主诉:{context.chief_complaint}")
        if context.medical_history:
            parts.append(f"病史:{context.medical_history}")
        if context.symptoms:
            parts.append(f"症状:{','.join(context.symptoms)}")
        return "，".join(parts) if parts else "无额外患者信息"
