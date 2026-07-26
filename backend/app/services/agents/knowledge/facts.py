# -*- coding: utf-8 -*-
"""结构化病例事实提取

从患者信息、对话记录、诊断和治疗方案中用正则和简单规则
提取结构化字段（ClinicalFacts），用于构建三类独立查询。
"""

import re
from typing import Optional

from app.services.rag.types import ClinicalFacts


def extract_clinical_facts(  # noqa: C901
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
