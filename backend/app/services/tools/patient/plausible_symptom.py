# -*- coding: utf-8 -*-
"""档案外症状裁决工具 — 医生问到档案未写明的症状时，用 RAG 检索该诊断的
典型临床表现，由低温 LLM 裁决该症状应否存在，避免患者乱编或一律否认。
任何异常降级为 uncertain（患者回答"记不清/不确定"）。
"""
import logging

from pydantic import BaseModel, Field

from app.services.qwen_client import call_qwen_chat
from app.services.rag.retriever import tiered_retrieve
from app.services.rag.types import RetrievalQuery
from app.services.tools.base import BaseTool, ToolContext
from app.utils.json_parser import extract_json_dict_from_text

logger = logging.getLogger(__name__)

_VALID_VERDICTS = ("present", "absent", "uncertain")

_VERDICT_SYSTEM = (
    "你是临床医学知识裁决助手。根据检索到的医学证据，判断某症状在给定诊断下"
    "是否为合理伴随症状。present=典型/常见伴随；absent=医学上不相符；"
    "uncertain=证据不足。只输出 JSON："
    '{"verdict": "present|absent|uncertain", "reason": "一句话理由"}'
)


class QueryPlausibleSymptomArgs(BaseModel):
    symptom: str = Field(description="医生问到的、患者档案未写明的症状")
    diagnosis: str = Field(description="患者的预期诊断或主要病情")


class QueryPlausibleSymptom(BaseTool):
    name = "query_plausible_symptom"
    description = "裁决档案外症状在该诊断下是否合理存在（present/absent/uncertain）"
    args_schema = QueryPlausibleSymptomArgs
    timeout_seconds = 30
    critical = False

    async def execute(self, args: QueryPlausibleSymptomArgs, context: ToolContext) -> dict:
        try:
            query = RetrievalQuery(
                query_type="diagnosis",
                text=f"{args.diagnosis} 典型症状 临床表现 {args.symptom}",
                source="clinical_facts",
            )
            bundle = await tiered_retrieve(queries=[query], top_k_per_query=3)
            evidence_text = "\n".join(
                f"[{item.source}] {item.text[:300]}" for item in (bundle.candidates or [])[:3]
            ) or "（未检索到相关证据）"

            raw = await call_qwen_chat(
                [{"role": "system", "content": _VERDICT_SYSTEM},
                 {"role": "user", "content": (
                     f"诊断：{args.diagnosis}\n待裁决症状：{args.symptom}\n\n医学证据：\n{evidence_text}"
                 )}],
                temperature=0.1, max_tokens=300,
            )
            data = extract_json_dict_from_text(raw)
            verdict = data.get("verdict", "uncertain")
            if verdict not in _VALID_VERDICTS:
                verdict = "uncertain"
            return {"verdict": verdict, "reason": str(data.get("reason", "")), "degraded": False}
        except Exception as e:
            logger.warning(f"query_plausible_symptom 裁决失败，降级 uncertain: {e}")
            return {"verdict": "uncertain", "reason": "知识库裁决失败，保守处理", "degraded": True}
