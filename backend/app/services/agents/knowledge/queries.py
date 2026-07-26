# -*- coding: utf-8 -*-
"""三类查询构建

基于结构化病例事实构建 case / diagnosis / treatment 三类独立查询，
消除确认偏误。
"""

import re

from app.services.rag.types import ClinicalFacts, RetrievalQuery


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
