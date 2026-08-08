# -*- coding: utf-8 -*-
"""RAG 管线模式知识核对

流程：
1. 结构化病例事实提取（extract_clinical_facts）
2. 三类查询构建（build_queries：case / diagnosis / treatment）
3. 分级检索（tiered_retrieve：Level1→2→3 级联）
4. 两阶段重排序（two_stage_rerank：专用 reranker + LLM 精排）
5. LLM 一致性判断 + 引用绑定
6. 拒答逻辑 + 评分映射
"""

import json
import logging
import time
from typing import Optional

from app.services.agents.knowledge.facts import extract_clinical_facts
from app.services.agents.knowledge.queries import build_queries
from app.services.agents.knowledge.scoring import (
    _extract_json,
    _generate_analysis,
    _map_consistency_to_score,
)
from app.services.prompts import get_prompt
from app.services.qwen_client import call_qwen_chat
from app.services.rag.reranker import two_stage_rerank
from app.services.rag.retriever import tiered_retrieve
from app.services.rag.types import (
    Citation,
    EvidenceItem,
    RetrievalConfidence,
    build_evidence_citation_id,
)

logger = logging.getLogger(__name__)

# ── System Prompt（一致性评估）──────────────────────────────────────────────────

CONSISTENCY_SYSTEM_PROMPT = get_prompt("knowledge.consistency_system")


async def run_knowledge_check(
    conversation_text: str,
    patient_info: str,
    doctor_diagnosis: str,
    treatment_plan: str,
    enable_hyde: bool = True,
) -> dict:
    """基于 RAG 分级检索与一致性评估的医学知识核对

    Args:
        conversation_text: 问诊对话记录
        patient_info: 患者基本信息
        doctor_diagnosis: 医生提交的诊断
        treatment_plan: 医生提交的治疗方案
        enable_hyde: 保留参数（分级检索内部自动控制 HyDE）

    Returns:
        dict 包含 raw_response（JSON 字符串）及新增字段
    """
    try:
        # ── Step 1: 提取结构化病例事实 ──
        facts = extract_clinical_facts(
            conversation_text, patient_info, doctor_diagnosis, treatment_plan
        )
        logger.info(
            f"病例事实提取完成：年龄={facts.age}, 性别={facts.gender}, "
            f"症状={len(facts.symptoms)}个, 诊断={len(facts.doctor_diagnoses)}个, "
            f"治疗项={len(facts.treatment_items)}个"
        )

        # ── Step 2: 构建三类查询 ──
        queries = build_queries(facts)
        logger.info(f"查询构建完成：{len(queries)}条查询 ({', '.join(q.query_type for q in queries)})")

        # ── Step 3: 分级检索 ──
        bundle = await tiered_retrieve(
            queries=queries,
            top_k_per_query=10,
            candidate_limit=20,
        )
        logger.info(
            f"分级检索完成：level={bundle.level_used}, status={bundle.status}, "
            f"候选={len(bundle.candidates)}条"
        )

        # ── Step 4: 两阶段重排序 ──
        reranked: list[EvidenceItem] = []
        rerank_degraded = False
        rerank_start = time.monotonic()
        if bundle.candidates:
            rerank_query = " | ".join(q.text for q in queries)
            # 记录 rerank 输入数量到 trace
            bundle.trace["rerank_input_count"] = len(bundle.candidates)
            bundle.trace["llm_rerank_count"] = min(len(bundle.candidates), 5)
            reranked, rerank_degraded = await two_stage_rerank(
                query=rerank_query,
                documents=bundle.candidates,
                top_k=5,
            )
            rerank_elapsed = (time.monotonic() - rerank_start) * 1000
            bundle.trace["timing"]["rerank_ms"] = round(rerank_elapsed, 1)
            logger.info(f"两阶段重排完成：{len(bundle.candidates)}条 → {len(reranked)}条 (degraded={rerank_degraded})")
        else:
            logger.info("无候选证据，跳过重排序")

        # ── Step 5: 一致性判断（LLM）──
        consistency_result = await _llm_consistency_check(
            reranked, doctor_diagnosis, treatment_plan, patient_info, conversation_text
        )
        evidence_stance = consistency_result.get("consistency", "undetermined")
        confidence = float(consistency_result.get("confidence", 0.5))
        logger.info(f"一致性判断：stance={evidence_stance}, confidence={confidence:.2f}")

        # ── Step 6: 构建引用列表 ──
        citations: list[Citation] = []
        for i, evidence in enumerate(reranked):
            citation_id = build_evidence_citation_id(evidence, i)
            citations.append(Citation(
                citation_id=citation_id,
                claim=evidence.text[:200],
                source=evidence.source,
                page=evidence.page,
                heading_path=evidence.heading_path,
                text_snippet=evidence.text[:500],
                rerank_score=evidence.rerank_score,
            ))
        logger.info(f"引用列表构建完成：{len(citations)}条引用")

        # ── Step 7: 确定 retrieval_status 和 evidence_stance ──
        retrieval_status: str = bundle.status
        if retrieval_status == "candidate":
            retrieval_status = "sufficient"

        # ── Step 8: 拒答逻辑（结合检索置信度）──
        retrieval_confidence = bundle.confidence  # "high" | "medium" | "low"
        needs_review = False
        review_reason: Optional[str] = None
        should_refuse = False
        score: Optional[int] = None

        # 低置信度 → 直接标记拒答
        if retrieval_confidence == RetrievalConfidence.LOW.value:
            should_refuse = True
            needs_review = True
            review_reason = "insufficient_evidence"
        elif retrieval_status in ("insufficient", "unavailable", "error"):
            needs_review = True
            review_reason = f"检索状态: {retrieval_status}"
        elif evidence_stance == "mixed" and confidence < 0.5:
            needs_review = True
            review_reason = f"证据立场混合且置信度低({confidence:.2f})"
        elif evidence_stance == "undetermined":
            needs_review = True
            review_reason = "证据立场无法确定"

        if not needs_review:
            score = _map_consistency_to_score(evidence_stance, confidence)

        if needs_review:
            logger.info(f"触发拒答逻辑：reason={review_reason}")

        # ── Step 9: 生成分析文本 ──
        analysis_text = _generate_analysis(
            consistency_result=consistency_result,
            facts=facts,
            doctor_diagnosis=doctor_diagnosis,
            treatment_plan=treatment_plan,
            retrieval_status=retrieval_status,
            evidence_stance=evidence_stance,
            citations=citations,
            needs_review=needs_review,
            review_reason=review_reason,
        )

        # ── Step 10: 构造返回结果 ──
        # raw_response 保持与 evaluation_service.py 的兼容性
        raw_payload = {
            "score": score,  # None 表示拒答，上层 evaluation_service 会正确处理
            "analysis": analysis_text,
        }

        result = {
            "raw_response": json.dumps(raw_payload, ensure_ascii=False),
            # 新增字段（供 Task 7 使用）
            "score": score,
            "analysis": analysis_text,
            "retrieval_status": retrieval_status,
            "evidence_stance": evidence_stance,
            "citations": [c.model_dump() for c in citations],
            "human_review_needed": needs_review,
            "review_reason": review_reason,
            "confidence": confidence,
            "retrieval_confidence": retrieval_confidence,
            "should_refuse": should_refuse,
            "rag_trace": bundle.trace,
            "degraded": bundle.degraded or rerank_degraded,
        }

        return result

    except Exception as e:
        logger.error(f"知识核对流程异常: {e}", exc_info=True)
        # 全局降级：不崩溃，返回安全默认值
        fallback_payload = {
            "score": None,
            "analysis": "医学知识核对过程中遇到技术问题，无法完成评估。建议人工复核。",
        }
        return {
            "raw_response": json.dumps(fallback_payload, ensure_ascii=False),
            "score": None,
            "analysis": fallback_payload["analysis"],
            "retrieval_status": "error",
            "evidence_stance": "undetermined",
            "citations": [],
            "human_review_needed": True,
            "review_reason": f"系统异常: {str(e)}",
            "confidence": 0.5,
            "rag_trace": {},
            "degraded": True,
        }


# ── LLM 一致性检查 ────────────────────────────────────────────────────────────

async def _llm_consistency_check(
    reranked: list[EvidenceItem],
    doctor_diagnosis: str,
    treatment_plan: str,
    patient_info: str,
    conversation_text: str,
) -> dict:
    """调用 LLM 分析重排后的证据与医生诊断/治疗方案的一致性

    Returns:
        dict 含 consistency, confidence, analysis, key_findings
    """
    if not reranked:
        logger.info("无重排证据，跳过 LLM 一致性检查")
        return {
            "consistency": "undetermined",
            "confidence": 0.3,
            "analysis": "未检索到足够的医学证据进行一致性评估",
            "key_findings": [],
        }

    # 构建证据文本
    evidence_parts = []
    for i, ev in enumerate(reranked, 1):
        snippet = ev.text[:600]
        source_info = f"（来源: {ev.source}"
        if ev.page:
            source_info += f", 第{ev.page}页"
        if ev.organization:
            source_info += f", {ev.organization}"
        source_info += "）"
        evidence_parts.append(f"证据{i}{source_info}：\n{snippet}")
    evidence_text = "\n\n".join(evidence_parts)

    user_content = (
        f"【患者信息】\n{patient_info}\n\n"
        f"【问诊对话摘要】\n{conversation_text[:1000]}\n\n"
        f"【医生诊断】\n{doctor_diagnosis}\n\n"
        f"【治疗方案】\n{treatment_plan}\n\n"
        f"【检索到的医学证据（共{len(reranked)}条）】\n{evidence_text}"
    )

    messages = [
        {"role": "system", "content": CONSISTENCY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await call_qwen_chat(messages, temperature=0.2)
        result = _extract_json(response)

        # 校验 consistency 字段值
        valid_stances = {"supports", "contradicts", "mixed", "undetermined"}
        stance = result.get("consistency", "undetermined")
        if stance not in valid_stances:
            # 兼容旧格式 true/false
            if stance is True or stance == "true":
                stance = "supports"
            elif stance is False or stance == "false":
                stance = "contradicts"
            else:
                stance = "undetermined"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "consistency": stance,
            "confidence": confidence,
            "analysis": str(result.get("analysis", "")),
            "key_findings": result.get("key_findings", []),
        }

    except Exception as e:
        logger.warning(f"LLM 一致性检查失败: {e}")
        return {
            "consistency": "undetermined",
            "confidence": 0.3,
            "analysis": f"一致性评估失败: {str(e)}",
            "key_findings": [],
        }
