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
import re
import time
import logging
from typing import Optional

from app.services.qwen_client import call_qwen_chat
from app.services.rag.types import (
    RetrievalQuery,
    ClinicalFacts,
    EvidenceItem,
    RetrievalBundle,
    Citation,
    KnowledgeAssessment,
)
from app.services.rag.retriever import tiered_retrieve
from app.services.rag.reranker import two_stage_rerank

logger = logging.getLogger(__name__)

# ── System Prompt（一致性评估）──────────────────────────────────────────────────

CONSISTENCY_SYSTEM_PROMPT = """你是一名医学知识核对专家。你的任务是对比医生的诊断和治疗方案与检索到的临床指南证据，评估其一致性。

评估维度：
1. 诊断一致性：医生的诊断是否与医学证据支持的诊断方向一致
2. 治疗合理性：治疗方案是否符合临床指南推荐的标准治疗方案
3. 禁忌症检查：治疗方案中是否存在医学证据明确指出的禁忌或不当之处
4. 遗漏检查：是否遗漏了医学证据建议的必要检查或评估

输出格式（严格 JSON）：
{
  "consistency": "supports" | "contradicts" | "mixed" | "undetermined",
  "confidence": 0.0-1.0,
  "analysis": "200字以内的分析文本，概述一致性和关键发现",
  "key_findings": ["发现1", "发现2"]
}

说明：
- supports：诊断和治疗方案与医学证据基本一致
- contradicts：存在明显不一致或不当之处
- mixed：部分一致、部分不一致
- undetermined：证据不足以做出判断
- confidence 表示判断的确定程度
- 输出纯 JSON，不要包含 markdown 格式或额外说明"""


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


# ── JSON 解析（三层策略）─────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON（三层解析策略）"""
    if not text or not text.strip():
        raise ValueError("LLM 返回内容为空")

    # 1. 尝试直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. 尝试移除 markdown 代码块后解析
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. 尝试正则提取第一个 JSON 对象
    try:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass

    raise ValueError(f"无法解析 JSON: {text[:200]}...")


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

        # ── Step 8: 拒答逻辑 ──
        needs_review = False
        review_reason: Optional[str] = None
        score: Optional[int] = None

        if retrieval_status in ("insufficient", "unavailable", "error"):
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
