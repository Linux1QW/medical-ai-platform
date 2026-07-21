# -*- coding: utf-8 -*-
"""医学知识核对智能体 — 基于 RAG 分级检索、两阶段重排与一致性评估

重构后的流程：
1. 结构化病例事实提取（extract_clinical_facts）
2. 三类查询构建（build_queries：case / diagnosis / treatment）
3. 分级检索（tiered_retrieve：Level1→2→3 级联）
4. 两阶段重排序（two_stage_rerank：专用 reranker + LLM 精排）
5. LLM 一致性判断 + 引用绑定
6. 拒答逻辑 + 评分映射
"""

import json
import logging
import re
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.services.qwen_client import call_qwen_chat, call_qwen_with_tools
from app.services.prompts import get_prompt
from app.services.rag.reranker import two_stage_rerank
from app.services.rag.retriever import tiered_retrieve
from app.services.rag.types import (
    Citation,
    ClinicalFacts,
    EvidenceItem,
    RetrievalConfidence,
    RetrievalQuery,
)
from app.services.tools import register_all_tools
from app.services.tools.base import ToolContext
from app.services.tools.budget import ToolBudget
from app.services.tools.executor import ToolExecutor
from app.services.tools.registry import ToolRegistry
from app.utils.json_parser import extract_json_from_text

logger = logging.getLogger(__name__)

# ── System Prompt（一致性评估）──────────────────────────────────────────────────

CONSISTENCY_SYSTEM_PROMPT = get_prompt("knowledge.consistency_system")


# ── 结构化病例事实提取 ──────────────────────────────────────────────────────────

def extract_clinical_facts(
    conversation_text: str,
    patient_info: str,
    doctor_diagnosis: str,
    treatment_plan: str,
) -> ClinicalFacts:
    """从评估输入中提取结构化病例事实

    使用正则和简单规则从患者信息、对话记录、诊断和治疗方案中
    提取结构化字段，用于构建三类独立查询。
    """
    # ── 年龄和性别 ──
    age: Optional[int] = None
    gender: Optional[str] = None

    age_match = re.search(r"年龄[:\s：]*(\d+)", patient_info)
    if age_match:
        age = int(age_match.group(1))
    else:
        age_match = re.search(r"(\d+)\s*岁", patient_info + " " + conversation_text)
        if age_match:
            age = int(age_match.group(1))

    gender_match = re.search(r"性别[:\s：]*(male|female|男|女)", patient_info, re.IGNORECASE)
    if gender_match:
        raw = gender_match.group(1)
        gender = "男性" if raw in ("male", "男") else "女性"
    else:
        if re.search(r"[男他]", patient_info):
            gender = "男性"
        elif re.search(r"[女她]", patient_info):
            gender = "女性"

    # ── 主诉 ──
    chief_complaint = ""
    cc_match = re.search(r"主诉[:\s：]*(.+?)(?:\n|$)", patient_info)
    if cc_match:
        chief_complaint = cc_match.group(1).strip()
    else:
        # 从对话中提取患者首次发言
        patient_msgs = re.findall(r"患者[:\s：]*(.+?)(?:\n|$)", conversation_text)
        if patient_msgs:
            chief_complaint = patient_msgs[0].strip()

    # ── 症状 ──
    symptoms: list[str] = []
    # 从 patient_info 的症状字段提取
    symptoms_match = re.search(r"症状[:\s：]*(.+?)(?:\n|$)", patient_info)
    if symptoms_match:
        raw_symptoms = re.split(r"[、，,；;/]", symptoms_match.group(1))
        symptoms = [s.strip() for s in raw_symptoms if s.strip() and len(s.strip()) >= 2]
    # 从对话中患者提及的症状补充
    symptom_keywords = [
        "咳嗽", "发热", "头痛", "头晕", "胸闷", "胸痛", "心悸",
        "腹痛", "腹胀", "恶心", "呕吐", "腹泻", "便秘",
        "乏力", "消瘦", "水肿", "呼吸困难", "气促",
        "尿频", "尿急", "尿痛", "血尿",
        "失眠", "焦虑", "抑郁", "麻木", "抽搐",
        "出血", "疼痛", "肿胀", "瘙痒", "皮疹",
    ]
    for kw in symptom_keywords:
        if kw in conversation_text and kw not in symptoms:
            symptoms.append(kw)

    # ── 时间线 ──
    timeline: list[str] = []
    time_patterns = re.findall(r"(\d+[天周月年]|\d+\s*[天周月年]|今[天日]|昨[天日]|\d+小?时前)", conversation_text)
    timeline = list(dict.fromkeys(time_patterns))  # 去重保序

    # ── 危险信号 ──
    red_flags: list[str] = []
    red_flag_keywords = [
        "咯血", "血尿", "便血", "呕血", "意识障碍", "昏迷",
        "剧烈头痛", "突发", "进行性加重", "体重下降", "消瘦",
        "高热不退", "呼吸困难", "休克",
    ]
    for kw in red_flag_keywords:
        if kw in conversation_text or kw in patient_info:
            red_flags.append(kw)

    # ── 合并症 ──
    comorbidities: list[str] = []
    comorbidity_keywords = [
        "高血压", "糖尿病", "冠心病", "房颤", "慢阻肺", "COPD",
        "乙肝", "丙肝", "肝硬化", "肾功能不全", "甲亢", "甲减",
        "哮喘", "脑梗", "心衰", "贫血",
    ]
    combined_text = conversation_text + " " + patient_info
    for kw in comorbidity_keywords:
        if kw in combined_text:
            comorbidities.append(kw)
    # 从病史字段提取
    history_match = re.search(r"病史[:\s：]*(.+?)(?:\n|$)", patient_info)
    if history_match:
        hist_text = history_match.group(1)
        for kw in comorbidity_keywords:
            if kw in hist_text and kw not in comorbidities:
                comorbidities.append(kw)

    # ── 用药 ──
    medications: list[str] = []
    med_patterns = [
        "阿司匹林", "华法林", "氯吡格雷", "利伐沙班",
        "二甲双胍", "胰岛素", "氨氯地平", "缬沙坦", "美托洛尔",
        "奥美拉唑", "阿托伐他汀", "辛伐他汀",
        "头孢", "阿莫西林", "左氧氟沙星", "甲硝唑",
        "地塞米松", "泼尼松", "布洛芬", "对乙酰氨基酚",
    ]
    for med in med_patterns:
        if med in combined_text:
            medications.append(med)

    # ── 过敏 ──
    allergies: list[str] = []
    allergy_match = re.search(r"过敏[史]?[:\s：]*(.+?)(?:\n|$)", combined_text)
    if allergy_match:
        raw_allergies = re.split(r"[、，,；;/]", allergy_match.group(1))
        allergies = [a.strip() for a in raw_allergies if a.strip() and a.strip() not in ("无", "否认", "无特殊")]
    if re.search(r"(无过敏|否认过敏|无药物过敏)", combined_text):
        allergies = []

    # ── 医生诊断列表 ──
    doctor_diagnoses: list[str] = []
    if doctor_diagnosis and doctor_diagnosis.strip() and not doctor_diagnosis.startswith("（"):
        raw_dx = re.split(r"[、，,;\n]", doctor_diagnosis)
        doctor_diagnoses = [d.strip() for d in raw_dx if d.strip()]

    # ── 治疗项目列表 ──
    treatment_items: list[str] = []
    if treatment_plan and treatment_plan.strip() and not treatment_plan.startswith("（"):
        raw_tx = re.split(r"[\n；;]", treatment_plan)
        treatment_items = [t.strip() for t in raw_tx if t.strip()]

    return ClinicalFacts(
        age=age,
        gender=gender,
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        timeline=timeline,
        red_flags=red_flags,
        comorbidities=comorbidities,
        medications=medications,
        allergies=allergies,
        doctor_diagnoses=doctor_diagnoses,
        treatment_items=treatment_items,
    )


# ── 三类查询构建 ─────────────────────────────────────────────────────────────

def build_queries(facts: ClinicalFacts) -> list[RetrievalQuery]:
    """构建三类独立查询，消除确认偏误"""
    queries = [
        RetrievalQuery(
            query_type="case",
            text=_build_case_query(facts),
            source="clinical_facts",
        )
    ]
    if facts.doctor_diagnoses:
        queries.append(
            RetrievalQuery(
                query_type="diagnosis",
                text=_build_diagnosis_query(facts),
                source="clinical_facts",
            )
        )
    if facts.treatment_items:
        queries.append(
            RetrievalQuery(
                query_type="treatment",
                text=_build_treatment_query(facts),
                source="clinical_facts",
            )
        )
    return queries


def _patient_demographic(facts: ClinicalFacts) -> str:
    """构建患者人口学描述片段"""
    parts = []
    if facts.age is not None and facts.gender:
        parts.append(f"{facts.age}岁{facts.gender}")
    elif facts.age is not None:
        parts.append(f"{facts.age}岁")
    elif facts.gender:
        parts.append(facts.gender)
    return "".join(parts)


def _build_case_query(facts: ClinicalFacts) -> str:
    """病例查询：仅包含病例事实，不包含医生诊断"""
    parts = []
    demo = _patient_demographic(facts)
    if demo:
        parts.append(demo)
    if facts.chief_complaint:
        parts.append(f"主诉：{facts.chief_complaint}")
    if facts.symptoms:
        parts.append(f"症状：{'、'.join(facts.symptoms[:8])}")
    if facts.timeline:
        parts.append(f"病程：{'，'.join(facts.timeline[:3])}")
    if facts.comorbidities:
        parts.append(f"既往史：{'、'.join(facts.comorbidities[:5])}")
    if facts.red_flags:
        parts.append(f"报警症状：{'、'.join(facts.red_flags[:3])}")
    return "，".join(parts) if parts else "病例信息查询"


def _build_diagnosis_query(facts: ClinicalFacts) -> str:
    """诊断查询：病例特征 + 医生诊断 + 鉴别诊断"""
    parts = []
    demo = _patient_demographic(facts)
    if demo:
        parts.append(demo)
    if facts.chief_complaint:
        parts.append(facts.chief_complaint)
    elif facts.symptoms:
        parts.append("、".join(facts.symptoms[:5]))
    if facts.doctor_diagnoses:
        parts.append(f"诊断：{'、'.join(facts.doctor_diagnoses)}")
    parts.append("鉴别诊断要点")
    return "，".join(parts) if parts else "诊断鉴别查询"


def _build_treatment_query(facts: ClinicalFacts) -> str:
    """治疗查询：疾病 + 分期 + 合并症 + 药物 + 剂量 + 疗程"""
    parts = []
    if facts.doctor_diagnoses:
        parts.append("、".join(facts.doctor_diagnoses))
    demo = _patient_demographic(facts)
    if demo:
        parts.append(demo)
    if facts.comorbidities:
        parts.append(f"合并{'、'.join(facts.comorbidities[:3])}")
    parts.append("治疗方案")
    if facts.treatment_items:
        # 提取治疗关键词（去掉序号和剂量细节）
        tx_keywords = []
        for item in facts.treatment_items[:5]:
            clean = re.sub(r"^\d+[.、)\s]+", "", item).strip()
            if clean:
                tx_keywords.append(clean[:30])
        if tx_keywords:
            parts.append("、".join(tx_keywords))
    parts.append("药物剂量 疗程 禁忌证")
    return "，".join(parts) if parts else "治疗方案查询"


# ── JSON 解析（委托公共模块）─────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON（三层解析策略）"""
    return extract_json_from_text(text)


# ── 评分映射 ─────────────────────────────────────────────────────────────────

def _map_consistency_to_score(stance: str, confidence: float) -> int:
    """将一致性和置信度映射为 0-100 分"""
    base_scores = {
        "supports": 90,
        "mixed": 65,
        "contradicts": 40,
        "undetermined": 50,
    }
    base = base_scores.get(stance, 50)
    return int(base * confidence + base * (1 - confidence) * 0.5)


# ── 分析文本生成 ──────────────────────────────────────────────────────────────

def _generate_analysis(
    consistency_result: dict,
    facts: ClinicalFacts,
    doctor_diagnosis: str,
    treatment_plan: str,
    retrieval_status: str,
    evidence_stance: str,
    citations: list,
    needs_review: bool,
    review_reason: Optional[str],
) -> str:
    """生成 150-300 字的分析文本"""
    if needs_review:
        analysis = f"医学知识核对无法完成自动评估。原因：{review_reason}。"
        analysis += "建议人工复核诊断和治疗方案的合理性。"
        if facts.doctor_diagnoses:
            analysis += f" 医生诊断：{'、'.join(facts.doctor_diagnoses[:3])}。"
        return analysis[:300]

    stance_desc = {
        "supports": "诊断和治疗方案与医学证据基本一致",
        "contradicts": "诊断和治疗方案与医学证据存在不一致",
        "mixed": "诊断和治疗方案与医学证据部分一致",
        "undetermined": "证据不足以确定一致性",
    }
    desc = stance_desc.get(evidence_stance, "一致性未确定")

    confidence = consistency_result.get("confidence", 0.5)
    analysis_text = consistency_result.get("analysis", "")
    key_findings = consistency_result.get("key_findings", [])

    parts = [
        f"医学知识核对结果：{desc}。",
        f"评估置信度为{confidence * 100:.0f}%。",
    ]
    if analysis_text:
        parts.append(analysis_text)
    if key_findings:
        parts.append(f"关键发现：{'；'.join(key_findings[:3])}")
    if citations:
        parts.append(f"共引用{len(citations)}条医学证据支持评估结论。")

    analysis = " ".join(parts)
    if len(analysis) < 150:
        analysis += " 建议医生在后续诊疗中持续关注指南更新，确保诊疗方案符合最新的循证医学证据。"
    return analysis[:300]


# ── 主函数 ────────────────────────────────────────────────────────────────────

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
            citation_id = f"rag-v2:{evidence.source}:{evidence.page or 0}:{i}"
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
        retrieval_status = bundle.status
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Use 模式 — 知识 Agent 通过 Function Calling 自主调用检索/重排工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ── 辅助函数：确定性分数映射（v2，禁止 LLM 干预）────────────────────────────

def _map_consistency_to_score_v2(consistency: str, confidence: float) -> float | None:
    """确定性分数映射，禁止 LLM 干预"""
    confidence = max(0.0, min(1.0, confidence))  # clamp

    if consistency == "supports":
        return round(80 + confidence * 15, 1)  # 80~95
    elif consistency == "mixed":
        return round(50 + confidence * 25, 1)  # 50~75
    elif consistency == "contradicts":
        return round(confidence * 45, 1)  # 0~45
    elif consistency == "undetermined":
        return None
    else:
        return None


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

async def run_knowledge_check_with_tools(
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
        executor = ToolExecutor(registry, max_result_chars=settings.TOOL_USE_MAX_RESULT_CHARS)
        budget = ToolBudget(context.budgets)
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
            return _build_error_result(tool_result.error, executor.get_traces())

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
    rag_trace = {
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ReAct 模式 — 显式 Thought → Action → Observation 推理链
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REACT_SYSTEM_PROMPT = get_prompt("knowledge.react_system")


def _parse_react_step(text: str) -> dict:
    """解析 ReAct 推理步骤，提取 Thought、Action、Action Input

    Returns:
        dict 含 thought, action, action_input, final_answer, is_final
    """
    result = {
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


async def run_knowledge_check_react(
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
