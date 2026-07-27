# -*- coding: utf-8 -*-
"""Tool Use 模式知识核对

知识 Agent 通过 Function Calling 自主调用检索/重排工具，
完成知识一致性评估。返回结构与 run_knowledge_check() 完全兼容。
"""

import json
import logging
import uuid
from typing import Optional

from app.core.config import settings
from app.services.agents.knowledge.facts import extract_clinical_facts
from app.services.agents.knowledge.queries import build_queries
from app.services.agents.knowledge.scoring import (
    _extract_json,
    _map_consistency_to_score_v2,
)
from app.services.prompts import get_prompt
from app.services.qwen_client import call_qwen_with_tools
from app.services.rag.types import EvidenceItem
from app.services.tools import register_all_tools
from app.services.tools.base import ToolContext
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor
from app.services.tools.registry import ToolRegistry
from app.services.tools.runtime import create_tool_budget, create_tool_executor

logger = logging.getLogger(__name__)

# ── Tool Use System Prompt ───────────────────────────────────────────────────

TOOL_USE_SYSTEM_PROMPT = get_prompt("knowledge.tool_use_system")


# ── 辅助函数：从 consultation 对象提取字段 ─────────────────────────────────────

def _extract_consultation_data(consultation) -> tuple:
    """从 consultation 对象/字典中提取 (conversation_text, patient_info, doctor_diagnosis, treatment_plan)"""
    if isinstance(consultation, dict):
        patient_info = consultation.get("patient_info", "")
        if not patient_info:
            parts = []
            if consultation.get("patient_age") is not None:
                parts.append(f"年龄：{consultation['patient_age']}岁")
            if consultation.get("patient_gender"):
                parts.append(f"性别：{consultation['patient_gender']}")
            if consultation.get("chief_complaint"):
                parts.append(f"主诉：{consultation['chief_complaint']}")
            if consultation.get("symptoms"):
                parts.append(f"症状：{'、'.join(consultation['symptoms'])}")
            if consultation.get("medical_history"):
                parts.append(f"病史：{consultation['medical_history']}")
            patient_info = "\n".join(parts)
        return (
            consultation.get("conversation_text", ""),
            patient_info,
            consultation.get("doctor_diagnosis", ""),
            consultation.get("treatment_plan", ""),
        )
    else:
        patient_info = getattr(consultation, "patient_info", "")
        if not patient_info:
            parts = []
            if getattr(consultation, "patient_age", None) is not None:
                parts.append(f"年龄：{consultation.patient_age}岁")
            if getattr(consultation, "patient_gender", None):
                parts.append(f"性别：{consultation.patient_gender}")
            if getattr(consultation, "chief_complaint", ""):
                parts.append(f"主诉：{consultation.chief_complaint}")
            if getattr(consultation, "symptoms", None):
                parts.append(f"症状：{'、'.join(consultation.symptoms)}")
            if getattr(consultation, "medical_history", ""):
                parts.append(f"病史：{consultation.medical_history}")
            patient_info = "\n".join(parts)
        return (
            getattr(consultation, "conversation_text", ""),
            patient_info,
            getattr(consultation, "doctor_diagnosis", ""),
            getattr(consultation, "treatment_plan", ""),
        )


# ── 辅助类：ToolExecutor 桥接器 ─────────────────────────────────────────────

class _ToolExecutorBridge:
    """桥接 call_qwen_with_tools 和 ToolExecutor 的适配器

    call_qwen_with_tools 调用 tool_executor.execute(tool_name, arguments_json)，
    而 ToolExecutor.execute 需要 context 和 budget 参数，此桥接器绑定这些参数。
    """

    def __init__(self, executor: ToolExecutor, context: ToolContext, budget: ToolBudget):
        self.executor = executor
        self.context = context
        self.budget = budget

    async def execute(self, tool_name: str, arguments_json: str) -> dict:
        return await self.executor.execute(
            tool_name, arguments_json,
            context=self.context,
            budget=self.budget,
        )


# ── 主函数：Tool Use 模式知识核对 ─────────────────────────────────────────────

async def run_knowledge_check_with_tools(  # noqa: C901
    consultation,
    diagnosis_text: str = "",
    treatment_text: str = "",
) -> dict:
    """基于 Tool Use / Function Calling 的医学知识核对

    让 LLM 通过 Function Call 自主调用检索/重排工具，完成知识一致性评估。
    返回结构与 run_knowledge_check() 完全兼容。

    Args:
        consultation: 评估上下文（dict 或 EvaluationContext 对象）
        diagnosis_text: 医生诊断文本（覆盖 consultation 中的值）
        treatment_text: 治疗方案文本（覆盖 consultation 中的值）

    Returns:
        dict 与 run_knowledge_check() 返回结构完全兼容
    """
    try:
        # ── Step 1: 提取病例数据 ──
        conversation_text, patient_info, orig_diagnosis, orig_treatment = \
            _extract_consultation_data(consultation)

        # 参数覆盖：优先使用显式传入的 diagnosis_text / treatment_text
        doctor_diagnosis = diagnosis_text or orig_diagnosis
        treatment_plan = treatment_text or orig_treatment

        # ── Step 2: 提取结构化病例事实 + 构建查询（复用确定性函数）──
        facts = extract_clinical_facts(
            conversation_text, patient_info, doctor_diagnosis, treatment_plan
        )
        queries = build_queries(facts)
        logger.info(
            f"[ToolUse] 病例事实提取完成：症状={len(facts.symptoms)}个, "
            f"诊断={len(facts.doctor_diagnoses)}个, 治疗项={len(facts.treatment_items)}个, "
            f"查询={len(queries)}条"
        )

        # ── Step 3: 构造 ToolContext ──
        context = ToolContext(
            run_id=str(uuid.uuid4()),
            agent_name="knowledge_agent",
            budgets={
                "search_medical_kb": settings.KNOWLEDGE_TOOL_MAX_RAG_CALLS,
                "expand_query": settings.KNOWLEDGE_TOOL_MAX_MQE_CALLS,
                "generate_hyde_query": settings.KNOWLEDGE_TOOL_MAX_HYDE_CALLS,
            },
            allowed_citation_ids=set(),
            evidence_cache={},
        )

        # ── Step 4: 构造 ToolRegistry + ToolExecutor + ToolBudget ──
        registry = ToolRegistry()
        register_all_tools(registry)
        executor = create_tool_executor(registry, max_result_chars=settings.TOOL_USE_MAX_RESULT_CHARS)
        budget = create_tool_budget(context.budgets, context.run_id)
        bridge = _ToolExecutorBridge(executor, context, budget)

        # ── Step 5: 构造 System Prompt ──
        system_prompt = TOOL_USE_SYSTEM_PROMPT

        # ── Step 6: 构造初始 messages ──
        user_parts = [f"【患者信息】\n{patient_info}"]
        if conversation_text:
            user_parts.append(f"【问诊对话摘要】\n{conversation_text[:1500]}")
        if doctor_diagnosis:
            user_parts.append(f"【医生诊断】\n{doctor_diagnosis}")
        if treatment_plan:
            user_parts.append(f"【治疗方案】\n{treatment_plan}")
        user_parts.append(
            "请基于检索到的医学证据，评估医生诊疗方案与循证医学指南的一致性。"
        )
        user_content = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # ── Step 7: 调用 call_qwen_with_tools ──
        tool_result = await call_qwen_with_tools(
            messages,
            tools=registry.get_openai_schemas(),
            tool_executor=bridge,
            temperature=0.2,
            max_tokens=2000,
        )

        if tool_result.degraded:
            logger.warning(f"[ToolUse] LLM 调用降级: {tool_result.error}")
            return _build_error_result(tool_result.error or "LLM 调用降级", executor.get_traces())

        # ── Step 8: 解析最终 JSON（复用 _extract_json）──
        try:
            parsed = _extract_json(tool_result.content)
        except ValueError as e:
            logger.warning(f"[ToolUse] JSON 解析失败: {e}")
            return _build_error_result(f"JSON 解析失败: {e}", executor.get_traces())

        # 校验 consistency / confidence 字段
        consistency = parsed.get("consistency", "undetermined")
        valid_stances = {"supports", "contradicts", "mixed", "undetermined"}
        if consistency not in valid_stances:
            consistency = "undetermined"
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        evidence_sufficiency = parsed.get("evidence_sufficiency", "insufficient")
        analysis_text = parsed.get("analysis", "")
        parsed.get("key_findings", [])
        used_citation_ids = parsed.get("used_citation_ids", [])

        # ── Step 9-10: 收集 allowed_citation_ids 并执行引用校验后处理 ──
        context.allowed_citation_ids = set(context.evidence_cache.keys())

        verify_result = await executor.execute(
            "verify_citation",
            json.dumps({"used_citation_ids": used_citation_ids}, ensure_ascii=False),
            context=context,
        )

        invalid_ids = []
        if verify_result.get("ok") and verify_result.get("data"):
            invalid_ids = verify_result["data"].get("invalid_citation_ids", [])

        if invalid_ids:
            # 移除非法引用
            valid_ids = [cid for cid in used_citation_ids if cid not in invalid_ids]
            parsed["used_citation_ids"] = valid_ids

            # 尝试一次修正重试
            correction_messages = list(messages)  # 浅拷贝原始消息
            correction_messages.append(
                {"role": "assistant", "content": tool_result.content}
            )
            correction_messages.append({
                "role": "user",
                "content": (
                    f"你使用了以下非法引用ID：{invalid_ids}。"
                    f"这些引用不存在于检索结果中。请修正你的评估，"
                    f"只使用合法的引用ID：{list(context.allowed_citation_ids)[:20]}。"
                    f"重新输出完整的 JSON 评估结果。"
                ),
            })

            correction_result = await call_qwen_with_tools(
                correction_messages,
                tools=registry.get_openai_schemas(),
                tool_executor=bridge,
                temperature=0.1,
                max_tokens=2000,
            )

            if not correction_result.degraded:
                try:
                    corrected = _extract_json(correction_result.content)
                    corrected_ids = corrected.get("used_citation_ids", [])

                    # 再次校验
                    re_verify = await executor.execute(
                        "verify_citation",
                        json.dumps({"used_citation_ids": corrected_ids}, ensure_ascii=False),
                        context=context,
                    )
                    re_invalid = []
                    if re_verify.get("ok") and re_verify.get("data"):
                        re_invalid = re_verify["data"].get("invalid_citation_ids", [])

                    if not re_invalid:
                        # 修正成功，使用修正后的结果
                        parsed = corrected
                        consistency = parsed.get("consistency", "undetermined")
                        if consistency not in valid_stances:
                            consistency = "undetermined"
                        confidence = float(parsed.get("confidence", 0.5))
                        confidence = max(0.0, min(1.0, confidence))
                        evidence_sufficiency = parsed.get("evidence_sufficiency", "insufficient")
                        analysis_text = parsed.get("analysis", "")
                        parsed.get("key_findings", [])
                        used_citation_ids = parsed.get("used_citation_ids", [])
                        # 合并修正轮次的 trace
                        tool_result.tool_calls.extend(correction_result.tool_calls)
                    else:
                        # 修正后仍有非法引用 → 强制失败
                        return _build_citation_failed_result(
                            analysis_text, executor.get_traces()
                        )
                except ValueError:
                    # 修正后 JSON 解析失败 → 强制失败
                    return _build_citation_failed_result(
                        analysis_text, executor.get_traces()
                    )
            else:
                # 修正重试降级 → 强制失败
                return _build_citation_failed_result(
                    analysis_text, executor.get_traces()
                )

        # ── Step 11: 确定性映射 knowledge_score ──
        needs_review = False
        review_reason: Optional[str] = None

        if evidence_sufficiency == "insufficient":
            needs_review = True
            review_reason = "insufficient_evidence"
        elif consistency == "undetermined":
            needs_review = True
            review_reason = "knowledge_undetermined"

        score = None
        if not needs_review:
            score = _map_consistency_to_score_v2(consistency, confidence)
            if score is None:
                needs_review = True
                review_reason = "knowledge_undetermined"

        # ── Step 12: 构建返回 dict（兼容旧版格式）──
        # 收集引用信息
        citations = []
        final_citation_ids = parsed.get("used_citation_ids", used_citation_ids)
        for cid in final_citation_ids:
            evidence_item = context.evidence_cache.get(cid)
            if evidence_item and isinstance(evidence_item, EvidenceItem):
                citations.append({
                    "citation_id": cid,
                    "claim": evidence_item.text[:200],
                    "source": evidence_item.source,
                    "page": evidence_item.page,
                    "heading_path": evidence_item.heading_path,
                    "text_snippet": evidence_item.text[:500],
                    "rerank_score": evidence_item.rerank_score,
                })

        # 构建 rag_trace（从 search_medical_kb 工具结果中提取）
        rag_trace = _build_rag_trace(executor.get_traces(), context)

        # 构建分析文本
        if needs_review:
            if review_reason == "insufficient_evidence":
                final_analysis = "现有证据不足，无法可靠评价医生诊疗方案与指南的一致性。"
            elif review_reason == "knowledge_undetermined":
                final_analysis = "现有证据不足以确定医生诊疗方案与指南的一致性。"
            else:
                final_analysis = analysis_text or "知识核对无法完成评估。"
        else:
            final_analysis = analysis_text
            if len(final_analysis) < 150:
                final_analysis += " 建议医生在后续诊疗中持续关注指南更新，确保诊疗方案符合最新的循证医学证据。"
            final_analysis = final_analysis[:300]

        # 构建 tool_trace
        tool_trace = _format_tool_trace(executor.get_traces())

        result = {
            "raw_response": json.dumps(
                {"score": score, "analysis": final_analysis},
                ensure_ascii=False,
            ),
            "score": score,
            "analysis": final_analysis,
            "retrieval_status": "sufficient" if not needs_review else "insufficient",
            "evidence_stance": consistency,
            "citations": citations,
            "human_review_needed": needs_review,
            "review_reason": review_reason,
            "confidence": confidence,
            "rag_trace": rag_trace,
            "tool_trace": tool_trace,
            "degraded": tool_result.degraded,
        }

        logger.info(
            f"[ToolUse] 知识核对完成：score={score}, consistency={consistency}, "
            f"confidence={confidence:.2f}, citations={len(citations)}, "
            f"review_needed={needs_review}"
        )
        return result

    except Exception as e:
        logger.error(f"[ToolUse] 知识核对流程异常: {e}", exc_info=True)
        return _build_error_result(f"系统异常: {str(e)}", [])


# ── Tool Use 辅助函数 ─────────────────────────────────────────────────────────


def _build_rag_trace(tool_traces: list[dict], context: ToolContext) -> dict:
    """从工具执行 trace 中提取检索相关信息构建 rag_trace"""
    rag_trace: dict = {
        "queries": [],
        "retrieval_level": "level0",
        "evidence_count": len(context.evidence_cache),
    }
    for trace in tool_traces:
        if trace.get("tool_name") == "search_medical_kb" and trace.get("status") == "success":
            query_summary = trace.get("arguments_summary", {}).get("query", "")
            if query_summary:
                rag_trace["queries"].append(query_summary)
            # 取最高检索级别
            rag_trace["retrieval_level"] = "level1"
    return rag_trace


def _format_tool_trace(tool_traces: list[dict]) -> list[dict]:
    """格式化工具调用 trace 为返回结构"""
    return [
        {
            "tool_name": t.get("tool_name", ""),
            "status": t.get("status", ""),
            "elapsed_ms": t.get("elapsed_ms", 0),
            "arguments_summary": t.get("arguments_summary", {}),
            "error": t.get("error"),
        }
        for t in tool_traces
    ]


def _build_error_result(error_msg: str, tool_traces: list[dict]) -> dict:
    """构建错误/降级返回结构"""
    fallback_analysis = "医学知识核对过程中遇到技术问题，无法完成评估。建议人工复核。"
    return {
        "raw_response": json.dumps(
            {"score": None, "analysis": fallback_analysis},
            ensure_ascii=False,
        ),
        "score": None,
        "analysis": fallback_analysis,
        "retrieval_status": "error",
        "evidence_stance": "undetermined",
        "citations": [],
        "human_review_needed": True,
        "review_reason": f"系统异常: {error_msg}",
        "confidence": 0.5,
        "rag_trace": {},
        "tool_trace": _format_tool_trace(tool_traces),
        "degraded": True,
    }


def _build_citation_failed_result(analysis_text: str, tool_traces: list[dict]) -> dict:
    """构建引用校验失败返回结构"""
    return {
        "raw_response": json.dumps(
            {"score": None, "analysis": analysis_text or "引用校验失败"},
            ensure_ascii=False,
        ),
        "score": None,
        "analysis": analysis_text or "引用校验失败",
        "retrieval_status": "error",
        "evidence_stance": "undetermined",
        "citations": [],
        "human_review_needed": True,
        "review_reason": "citation_verification_failed",
        "confidence": 0.5,
        "rag_trace": {},
        "tool_trace": _format_tool_trace(tool_traces),
        "degraded": True,
    }
