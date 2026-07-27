# -*- coding: utf-8 -*-
"""评分映射与分析文本生成

- JSON 解析（委托公共模块）
- v1 评分映射（RAG 管线模式）
- v2 确定性评分映射（Tool Use / ReAct 模式，禁止 LLM 干预）
- 150-300 字分析文本生成
"""

from typing import Optional

from app.services.rag.types import ClinicalFacts
from app.utils.json_parser import extract_json_dict_from_text


def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON（三层解析策略）"""
    return extract_json_dict_from_text(text)


# 机器码 → 用户可读中文（仅用于展示文本，不改变存储/比较用的原值）
REVIEW_REASON_LABELS = {
    "insufficient_evidence": "检索证据不足",
    "knowledge_undetermined": "证据立场无法确定",
}

RETRIEVAL_STATUS_LABELS = {
    "sufficient": "证据充分",
    "insufficient": "证据不足",
    "unavailable": "检索不可用",
    "error": "检索异常",
}

EVIDENCE_STANCE_LABELS = {
    "supports": "支持诊断",
    "contradicts": "与证据相悖",
    "mixed": "证据立场混合",
    "undetermined": "无法确定",
}


def humanize_review_reason(reason: str) -> str:
    """将 review_reason 中的英文机器码片段替换为中文（兼容混合文案）"""
    for code, label in {**REVIEW_REASON_LABELS, **RETRIEVAL_STATUS_LABELS}.items():
        reason = reason.replace(code, label)
    return reason


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
        reason_text = humanize_review_reason(review_reason) if review_reason else "需要人工复核"
        analysis = f"医学知识核对无法完成自动评估。原因：{reason_text}。"
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
