# -*- coding: utf-8 -*-
"""ReAct 模式知识核对 — 显式 Thought → Action → Observation 推理链

与 Tool Use 模式的区别：
- 传统 Tool Use：LLM 隐式决定调用工具，推理过程不透明
- ReAct：LLM 必须先输出 Thought 解释原因，再输出 Action 调用工具
"""

import json
import logging
import re
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.services.agents.knowledge.facts import extract_clinical_facts
from app.services.agents.knowledge.queries import build_queries
from app.services.agents.knowledge.scoring import (
    _extract_json,
    _map_consistency_to_score_v2,
)
from app.services.agents.knowledge.tool_use import (
    _build_error_result,
    _build_rag_trace,
    _extract_consultation_data,
    _format_tool_trace,
    _ToolExecutorBridge,
)
from app.services.prompts import get_prompt
from app.services.rag.types import EvidenceItem
from app.services.tools import register_all_tools
from app.services.tools.base import ToolContext
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ── ReAct System Prompt ──────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = get_prompt("knowledge.react_system")


def _parse_react_step(text: str) -> dict:
    """解析 ReAct 推理步骤，提取 Thought、Action、Action Input

    Returns:
        dict 含 thought, action, action_input, final_answer, is_final
    """
    result: dict = {
        "thought": "",
        "action": "",
        "action_input": {},
        "final_answer": None,
        "is_final": False,
    }

    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 检查是否为最终答案
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["is_final"] = True
        result["final_answer"] = final_match.group(1).strip()
        return result

    # 提取 Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # 提取 Action Input
    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            # 尝试修复常见 JSON 问题
            raw = input_match.group(1)
            # 移除尾部逗号
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            try:
                result["action_input"] = json.loads(raw)
            except json.JSONDecodeError:
                result["action_input"] = {}

    return result


async def run_knowledge_check_react(  # noqa: C901
    consultation,
    diagnosis_text: str = "",
    treatment_text: str = "",
) -> dict:
    """基于 ReAct 模式的医学知识核对

    实现显式的 Thought → Action → Observation 推理链，
    让 LLM 在每一步明确表达推理过程，再决定调用哪个工具。

    与 run_knowledge_check_with_tools 的区别：
    - 传统 Tool Use：LLM 隐式决定调用工具，推理过程不透明
    - ReAct：LLM 必须先输出 Thought 解释原因，再输出 Action 调用工具

    Args:
        consultation: 评估上下文（dict 或 EvaluationContext 对象）
        diagnosis_text: 医生诊断文本
        treatment_text: 治疗方案文本

    Returns:
        dict 与 run_knowledge_check() 返回结构完全兼容
    """
    from app.services.qwen_client import call_qwen_chat

    try:
        # ── Step 1: 提取病例数据 ──
        conversation_text, patient_info, orig_diagnosis, orig_treatment = \
            _extract_consultation_data(consultation)

        doctor_diagnosis = diagnosis_text or orig_diagnosis
        treatment_plan = treatment_text or orig_treatment

        # ── Step 2: 提取结构化病例事实 + 构建查询（复用确定性函数）──
        facts = extract_clinical_facts(
            conversation_text, patient_info, doctor_diagnosis, treatment_plan
        )
        build_queries(facts)
        logger.info(
            f"[ReAct] 病例事实提取完成：症状={len(facts.symptoms)}个, "
            f"诊断={len(facts.doctor_diagnoses)}个, 治疗项={len(facts.treatment_items)}个"
        )

        # ── Step 3: 构造 ToolContext + 工具注册 ──
        context = ToolContext(
            run_id=str(uuid.uuid4()),
            agent_name="knowledge_agent_react",
            budgets={
                "search_medical_kb": settings.KNOWLEDGE_TOOL_MAX_RAG_CALLS,
                "expand_query": settings.KNOWLEDGE_TOOL_MAX_MQE_CALLS,
                "generate_hyde_query": settings.KNOWLEDGE_TOOL_MAX_HYDE_CALLS,
            },
            allowed_citation_ids=set(),
            evidence_cache={},
        )

        registry = ToolRegistry()
        register_all_tools(registry)
        executor = ToolExecutor(registry, max_result_chars=settings.TOOL_USE_MAX_RESULT_CHARS)
        budget = ToolBudget(context.budgets)
        bridge = _ToolExecutorBridge(executor, context, budget)

        # ── Step 4: 构建初始输入 ──
        user_parts = [
            f"【患者信息】\n{patient_info}",
        ]
        if conversation_text:
            user_parts.append(f"【问诊对话摘要】\n{conversation_text[:1500]}")
        if doctor_diagnosis:
            user_parts.append(f"【医生诊断】\n{doctor_diagnosis}")
        if treatment_plan:
            user_parts.append(f"【治疗方案】\n{treatment_plan}")
        user_parts.append(
            "请使用 ReAct 框架逐步评估医生诊疗方案与循证医学指南的一致性。"
            "先检索相关医学证据，再分析一致性，最后输出评估结果。"
        )
        user_content = "\n\n".join(user_parts)

        # ── Step 5: ReAct 推理循环 ──
        max_steps = settings.REACT_MAX_STEPS
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        react_trace = []  # 记录每步推理过程
        final_parsed = None
        step_count = 0

        for step_idx in range(max_steps):
            step_count += 1
            step_start = time.monotonic()

            # 调用 LLM 生成推理步骤（不使用 tool calling，而是让 LLM 输出文本格式的 Action）
            response = await call_qwen_chat(messages, temperature=0.2)
            elapsed_ms = round((time.monotonic() - step_start) * 1000, 1)

            parsed_step = _parse_react_step(response)
            react_trace.append({
                "step": step_idx + 1,
                "thought": parsed_step["thought"][:200],
                "action": parsed_step["action"],
                "elapsed_ms": elapsed_ms,
            })

            # 如果是最终答案，解析并退出
            if parsed_step["is_final"]:
                try:
                    final_parsed = _extract_json(parsed_step["final_answer"])
                except ValueError:
                    # 尝试从整个 response 中提取
                    try:
                        final_parsed = _extract_json(response)
                    except ValueError:
                        final_parsed = None
                break

            # 执行工具调用
            if parsed_step["action"] and parsed_step["action_input"]:
                tool_name = parsed_step["action"]
                tool_args = parsed_step["action_input"]

                try:
                    tool_result = await bridge.execute(
                        tool_name,
                        json.dumps(tool_args, ensure_ascii=False),
                    )
                    observation = json.dumps(tool_result, ensure_ascii=False, default=str)
                    # 截断过长的观察结果
                    if len(observation) > 3000:
                        observation = observation[:3000] + "...(结果已截断)"
                except Exception as e:
                    observation = f"工具执行失败: {type(e).__name__}: {str(e)[:200]}"
                    logger.warning(f"[ReAct] 工具 {tool_name} 执行失败: {e}")

                react_trace[-1]["observation_summary"] = observation[:200]

                # 将推理步骤和观察结果追加到消息历史
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续推理。如果证据充足，输出 Final Answer。",
                })
            else:
                # LLM 没有输出有效的 Action，提示继续
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "请继续你的推理。如果需要更多信息，请使用工具；如果证据充足，请输出 Final Answer。",
                })

        # ── Step 6: 处理结果 ──
        if final_parsed is None:
            # 达到最大步数仍未输出最终答案，尝试最后一次 LLM 调用强制输出
            logger.warning(f"[ReAct] 达到最大步数 {max_steps}，尝试强制获取最终答案")
            messages.append({
                "role": "user",
                "content": "推理步骤已达上限。请立即基于已收集的证据输出 Final Answer（JSON 格式）。",
            })
            forced_response = await call_qwen_chat(messages, temperature=0.1)
            try:
                final_parsed = _extract_json(forced_response)
            except ValueError:
                return _build_error_result("ReAct 推理未能得出最终结论", executor.get_traces())

        # ── Step 7: 校验和映射（与 Tool Use 模式一致）──
        consistency = final_parsed.get("consistency", "undetermined")
        valid_stances = {"supports", "contradicts", "mixed", "undetermined"}
        if consistency not in valid_stances:
            consistency = "undetermined"
        confidence = float(final_parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        evidence_sufficiency = final_parsed.get("evidence_sufficiency", "insufficient")
        analysis_text = final_parsed.get("analysis", "")
        final_parsed.get("key_findings", [])
        used_citation_ids = final_parsed.get("used_citation_ids", [])

        # ── Step 8: 引用校验后处理 ──
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
            valid_ids = [cid for cid in used_citation_ids if cid not in invalid_ids]
            final_parsed["used_citation_ids"] = valid_ids
            used_citation_ids = valid_ids
            logger.info(f"[ReAct] 引用校验：移除 {len(invalid_ids)} 个非法引用")

        # ── Step 9: 确定性映射分数 ──
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

        # ── Step 10: 构建返回结构 ──
        citations = []
        final_citation_ids = final_parsed.get("used_citation_ids", used_citation_ids)
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

        rag_trace = _build_rag_trace(executor.get_traces(), context)
        rag_trace["react_steps"] = react_trace

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
            "react_trace": react_trace,
            "react_steps_count": step_count,
            "degraded": False,
        }

        logger.info(
            f"[ReAct] 知识核对完成：score={score}, consistency={consistency}, "
            f"confidence={confidence:.2f}, steps={step_count}, citations={len(citations)}"
        )
        return result

    except Exception as e:
        logger.error(f"[ReAct] 知识核对流程异常: {e}", exc_info=True)
        return _build_error_result(f"系统异常: {str(e)}", [])
