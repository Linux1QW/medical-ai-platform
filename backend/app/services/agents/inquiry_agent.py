# -*- coding: utf-8 -*-
"""问诊分析智能体 — 基于结构化建模与可计算指标的问诊过程评估"""

import json
import logging

from app.services.prompts import get_prompt
from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_from_text

# ── 临床 Schema 定义 ──
CLINICAL_SCHEMA = {
    "chief_complaint": ["symptom", "duration", "severity"],
    "history": ["onset", "progression", "trigger"],
    "past_history": ["disease", "surgery"],
    "medication": ["current_drugs"],
    "allergy": ["drug_allergy"]
}

CRITICAL_PATH = ["symptom", "onset", "duration", "severity", "associated_symptom", "risk_factor"]

IDEAL_ORDER = ["symptom", "history", "risk_factor", "past_history"]

# 槽位字段中文映射
SLOT_CN_MAP = {
    "chief_complaint.symptom": "主要症状",
    "chief_complaint.duration": "持续时间",
    "chief_complaint.severity": "严重程度",
    "history.onset": "起病方式",
    "history.progression": "病情演变",
    "history.trigger": "诱发因素",
    "past_history.disease": "既往疾病",
    "past_history.surgery": "手术史",
    "medication.current_drugs": "当前用药",
    "allergy.drug_allergy": "药物过敏",
}

# 关键路径字段中文映射
CRITICAL_CN_MAP = {
    "symptom": "主要症状",
    "onset": "起病方式",
    "duration": "持续时间",
    "severity": "严重程度",
    "associated_symptom": "伴随症状",
    "risk_factor": "危险因素",
}

# 问诊步骤中文映射
STEP_CN_MAP = {
    "symptom": "症状采集",
    "history": "病史询问",
    "risk_factor": "危险因素",
    "past_history": "既往史",
    "other": "其他",
}

# ── 权重配置 ──
WEIGHTS = {
    "coverage": 0.3,
    "critical": 0.3,
    "logic": 0.2,
    "efficiency": 0.2,
}

# ── LLM Prompts ──

# 第一次 LLM 调用：槽位填充 + 关键路径检查
SLOT_FILLING_SYSTEM_PROMPT = get_prompt("inquiry.slot_filling_system")

SLOT_FILLING_FEWSHOT_USER = """【患者信息】
姓名: 马xx, 年龄: 63, 性别: male
主诉: 间断性进食硬咽感1周
病史: 脑梗

【问诊对话记录】
医生: 咋的了，为啥来看胃肠科
患者: 这一周之内吧吃饭有点，咽进去以后噎挺
医生: 之前噎吗
患者: 之前不得
医生: 突然这一周出现的是吗？
患者: 就这一周之内。
医生: 越来越噎了，还是时有时无。
患者: 时有时无。
医生: 干的稀的凉的热的有关系吗？
患者: 没有。
医生: 感觉吃饭到哪噎，能下去不
患者: 这个位置能下去，到这个位置，能下去。
医生: 不会吐出来是吧？啥手术做的是
患者: 去年7月份脑梗
医生: 做的脑梗手术是吧？
患者: 对。
医生: 长期吃啥药吗？
患者: 阿司匹林、波维

请提取槽位填充信息。"""

SLOT_FILLING_FEWSHOT_ASSISTANT = """{
  "slots": {
    "chief_complaint": {"symptom": true, "duration": true, "severity": false},
    "history": {"onset": true, "progression": true, "trigger": false},
    "past_history": {"disease": true, "surgery": true},
    "medication": {"current_drugs": true},
    "allergy": {"drug_allergy": false}
  },
  "critical_slots": {
    "associated_symptom": false,
    "risk_factor": false
  }
}"""

# 第二次 LLM 调用：问诊步骤序列 + 问题分类
LOGIC_EFFICIENCY_SYSTEM_PROMPT = get_prompt("inquiry.logic_efficiency_system")

LOGIC_EFFICIENCY_FEWSHOT_USER = """【患者信息】
姓名: 任xx, 年龄: 29, 性别: female
主诉: 大便不成形，一天两三次，排气多

【问诊对话记录】
医生: 你为啥做检查呀？
患者: 我是之前一天可能大便两三次，排气也特别多，还不成形
医生: 那肠镜都没啥大事，胃有啥难受的
患者: 有的时候肚子这块觉得有点硌
医生: 你现在大便一天几次？
患者: 现在可能得两次
医生: 那大便啥样呢？
患者: 一般都不稀，反正就不成形
医生: 体重有啥变化没有？
患者: 没有。
医生: 有的人就是跟肠道功能有关
患者: 嗯，反正我从小就这样。
医生: 但检查显示肠道也没有炎症。
患者: 噢，那这个病理结果也没说有啥问题
医生: 这是终生性疾病，良性不会癌变
患者: 哦
医生: 息肉都已经给你除了，一年以后复查看看就行。
患者: 一年之后还得做个胃肠镜？
医生: 查肠镜就行了。

请提取问诊步骤序列和问题分类。"""

LOGIC_EFFICIENCY_FEWSHOT_ASSISTANT = """{
  "inquiry_steps": ["symptom", "symptom", "symptom", "symptom", "symptom", "other", "other", "other"],
  "question_classification": [
    {"question": "你为啥做检查呀？", "type": "symptom", "category": "relevant"},
    {"question": "那肠镜都没啥大事，胃有啥难受的", "type": "symptom", "category": "relevant"},
    {"question": "你现在大便一天几次？", "type": "symptom", "category": "relevant"},
    {"question": "那大便啥样呢？", "type": "symptom", "category": "relevant"},
    {"question": "体重有啥变化没有？", "type": "risk_factor", "category": "relevant"},
    {"question": "有的人就是跟肠道功能有关", "type": "other", "category": "relevant"},
    {"question": "这是终生性疾病，良性不会癌变", "type": "other", "category": "relevant"},
    {"question": "息肉都已经给你除了，一年以后复查看看就行", "type": "other", "category": "relevant"}
  ]
}"""


# ── Helper Functions ──

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON"""
    return extract_json_from_text(text)


def _calculate_coverage(slot_data: dict) -> float:
    """计算覆盖率得分"""
    slots = slot_data.get("slots", {})
    total_slots = 0
    filled_slots = 0

    for category, items in CLINICAL_SCHEMA.items():
        category_data = slots.get(category, {})
        for slot in items:
            total_slots += 1
            if category_data.get(slot, False):
                filled_slots += 1

    return filled_slots / total_slots if total_slots > 0 else 0.0


def _calculate_critical(slot_data: dict) -> float:
    """计算关键路径得分"""
    slots = slot_data.get("slots", {})
    critical_slots = slot_data.get("critical_slots", {})

    # 从 Schema 槽位中提取关键路径相关槽位
    critical_from_schema = {
        "symptom": slots.get("chief_complaint", {}).get("symptom", False),
        "onset": slots.get("history", {}).get("onset", False),
        "duration": slots.get("chief_complaint", {}).get("duration", False),
        "severity": slots.get("chief_complaint", {}).get("severity", False),
    }

    # 合并关键路径槽位
    all_critical = {
        **critical_from_schema,
        "associated_symptom": critical_slots.get("associated_symptom", False),
        "risk_factor": critical_slots.get("risk_factor", False),
    }

    hit_count = sum(1 for v in all_critical.values() if v)
    total_count = len(CRITICAL_PATH)

    return hit_count / total_count if total_count > 0 else 0.0


def _calculate_logic(logic_data: dict) -> float:
    """计算问诊逻辑得分"""
    steps = logic_data.get("inquiry_steps", [])

    # 过滤出有效步骤（排除 other）
    valid_steps = [s for s in steps if s in IDEAL_ORDER]

    if len(valid_steps) <= 1:
        return 1.0  # 步骤太少，默认满分

    # 计算顺序偏差
    order_violations = 0
    for i in range(len(valid_steps) - 1):
        current_idx = IDEAL_ORDER.index(valid_steps[i]) if valid_steps[i] in IDEAL_ORDER else -1
        next_idx = IDEAL_ORDER.index(valid_steps[i + 1]) if valid_steps[i + 1] in IDEAL_ORDER else -1

        if current_idx != -1 and next_idx != -1 and next_idx < current_idx:
            # 逆序违规
            order_violations += 1

    # 计算跳步（跳过中间步骤）
    for i in range(len(valid_steps) - 1):
        current_idx = IDEAL_ORDER.index(valid_steps[i]) if valid_steps[i] in IDEAL_ORDER else -1
        next_idx = IDEAL_ORDER.index(valid_steps[i + 1]) if valid_steps[i + 1] in IDEAL_ORDER else -1

        if current_idx != -1 and next_idx != -1 and next_idx > current_idx + 1:
            # 跳步
            order_violations += 0.5

    n = len(valid_steps)
    logic_score = 1.0 - (order_violations / n)
    return max(0.0, min(1.0, logic_score))


def _calculate_efficiency(logic_data: dict) -> float:
    """计算问诊效率得分"""
    classifications = logic_data.get("question_classification", [])

    if not classifications:
        return 0.5  # 默认值

    relevant_count = sum(1 for c in classifications if c.get("category") == "relevant")
    total_count = len(classifications)

    return relevant_count / total_count if total_count > 0 else 0.0


def _generate_analysis(coverage: float, critical: float, logic: float, efficiency: float,
                       slot_data: dict, logic_data: dict) -> str:
    """生成详细的分析文本"""
    slots = slot_data.get("slots", {})
    critical_slots = slot_data.get("critical_slots", {})

    # 收集已填充和未填充的槽位
    filled = []
    unfilled = []

    for category, items in CLINICAL_SCHEMA.items():
        category_data = slots.get(category, {})
        for slot in items:
            slot_name = f"{category}.{slot}"
            cn_name = SLOT_CN_MAP.get(slot_name, slot_name)
            if category_data.get(slot, False):
                filled.append(cn_name)
            else:
                unfilled.append(cn_name)

    # 关键路径状态
    critical_filled = []
    critical_unfilled = []

    if slots.get("chief_complaint", {}).get("symptom"):
        critical_filled.append(CRITICAL_CN_MAP.get("symptom", "symptom"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("symptom", "symptom"))

    if slots.get("history", {}).get("onset"):
        critical_filled.append(CRITICAL_CN_MAP.get("onset", "onset"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("onset", "onset"))

    if slots.get("chief_complaint", {}).get("duration"):
        critical_filled.append(CRITICAL_CN_MAP.get("duration", "duration"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("duration", "duration"))

    if slots.get("chief_complaint", {}).get("severity"):
        critical_filled.append(CRITICAL_CN_MAP.get("severity", "severity"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("severity", "severity"))

    if critical_slots.get("associated_symptom"):
        critical_filled.append(CRITICAL_CN_MAP.get("associated_symptom", "associated_symptom"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("associated_symptom", "associated_symptom"))

    if critical_slots.get("risk_factor"):
        critical_filled.append(CRITICAL_CN_MAP.get("risk_factor", "risk_factor"))
    else:
        critical_unfilled.append(CRITICAL_CN_MAP.get("risk_factor", "risk_factor"))

    # 问诊步骤分析
    steps = logic_data.get("inquiry_steps", [])
    classifications = logic_data.get("question_classification", [])

    relevant_count = sum(1 for c in classifications if c.get("category") == "relevant")
    redundant_count = sum(1 for c in classifications if c.get("category") == "redundant")
    irrelevant_count = sum(1 for c in classifications if c.get("category") == "irrelevant")

    # 生成分析文本
    analysis_parts = []

    # 1. 信息覆盖分析
    coverage_desc = f"信息覆盖度{coverage*100:.0f}%，共{len(filled)}个信息点已采集"
    if unfilled:
        coverage_desc += f"，遗漏{len(unfilled)}项：{', '.join(unfilled[:3])}"
        if len(unfilled) > 3:
            coverage_desc += "等"
    analysis_parts.append(coverage_desc)

    # 2. 关键路径分析
    critical_desc = f"关键信息采集率{critical*100:.0f}%，已覆盖{len(critical_filled)}项关键要素"
    if critical_unfilled:
        critical_desc += f"，缺失：{', '.join(critical_unfilled)}"
    analysis_parts.append(critical_desc)

    # 3. 问诊逻辑分析
    valid_steps = [s for s in steps if s in IDEAL_ORDER]
    if valid_steps:
        cn_steps = [STEP_CN_MAP.get(s, s) for s in valid_steps]
        logic_desc = f"问诊逻辑性{logic*100:.0f}%，问诊流程依次为{'→'.join(cn_steps)}"
        if logic < 0.8:
            logic_desc += "，存在问诊顺序不够规范或跳步情况"
    else:
        logic_desc = f"问诊逻辑性{logic*100:.0f}%"
    analysis_parts.append(logic_desc)

    # 4. 问诊效率分析
    total_q = len(classifications)
    efficiency_desc = f"问诊效率{efficiency*100:.0f}%，共{total_q}个问题，其中{relevant_count}个有效"
    if redundant_count > 0:
        efficiency_desc += f"，{redundant_count}个重复"
    if irrelevant_count > 0:
        efficiency_desc += f"，{irrelevant_count}个偏离主题"
    analysis_parts.append(efficiency_desc)

    return "。".join(analysis_parts) + "。"


# ── Main Function ──

async def run_inquiry_analysis(conversation_text: str, patient_info: str) -> dict:
    """
    基于结构化建模与可计算指标的问诊过程评估

    评估维度：
    1. Coverage (30%): 临床信息 Schema 槽位填充率
    2. Critical (30%): 关键问诊路径覆盖率
    3. Logic (20%): 问诊步骤顺序合理性
    4. Efficiency (20%): 问题有效性比例
    """
    try:
        # ── Step 1 & 2: 槽位填充 + 关键路径（第一次 LLM 调用）──
        slot_messages = [
            {"role": "system", "content": SLOT_FILLING_SYSTEM_PROMPT},
            {"role": "user", "content": SLOT_FILLING_FEWSHOT_USER},
            {"role": "assistant", "content": SLOT_FILLING_FEWSHOT_ASSISTANT},
            {
                "role": "user",
                "content": f"【患者信息】\n{patient_info}\n\n【问诊对话记录】\n{conversation_text}\n\n请提取槽位填充信息。"
            },
        ]

        slot_result = await call_qwen_chat(slot_messages, temperature=0.2)
        slot_data = _extract_json(slot_result)

    except Exception as e:
        logging.error(f"槽位填充 LLM 调用失败: {e}")
        # 降级处理：使用默认空数据
        slot_data = {
            "slots": {
                "chief_complaint": {"symptom": False, "duration": False, "severity": False},
                "history": {"onset": False, "progression": False, "trigger": False},
                "past_history": {"disease": False, "surgery": False},
                "medication": {"current_drugs": False},
                "allergy": {"drug_allergy": False}
            },
            "critical_slots": {"associated_symptom": False, "risk_factor": False}
        }

    try:
        # ── Step 3 & 4: 问诊步骤 + 问题分类（第二次 LLM 调用）──
        logic_messages = [
            {"role": "system", "content": LOGIC_EFFICIENCY_SYSTEM_PROMPT},
            {"role": "user", "content": LOGIC_EFFICIENCY_FEWSHOT_USER},
            {"role": "assistant", "content": LOGIC_EFFICIENCY_FEWSHOT_ASSISTANT},
            {
                "role": "user",
                "content": f"【患者信息】\n{patient_info}\n\n【问诊对话记录】\n{conversation_text}\n\n请提取问诊步骤序列和问题分类。"
            },
        ]

        logic_result = await call_qwen_chat(logic_messages, temperature=0.2)
        logic_data = _extract_json(logic_result)

    except Exception as e:
        logging.error(f"问诊逻辑 LLM 调用失败: {e}")
        # 降级处理
        logic_data = {
            "inquiry_steps": [],
            "question_classification": []
        }

    # ── Step 5: 数学计算综合评分 ──
    coverage = _calculate_coverage(slot_data)
    critical = _calculate_critical(slot_data)
    logic = _calculate_logic(logic_data)
    efficiency = _calculate_efficiency(logic_data)

    # 加权计算最终得分
    final_score = (
        WEIGHTS["coverage"] * coverage +
        WEIGHTS["critical"] * critical +
        WEIGHTS["logic"] * logic +
        WEIGHTS["efficiency"] * efficiency
    ) * 100

    # 确保分数在 0-100 范围内并取整
    final_score = int(round(max(0.0, min(100.0, final_score))))

    # 生成分析文本
    analysis = _generate_analysis(coverage, critical, logic, efficiency, slot_data, logic_data)

    # 构建返回结果
    result = {
        "score": final_score,
        "analysis": analysis,
        "details": {
            "coverage": {"score": round(coverage * 100, 1), "weight": WEIGHTS["coverage"]},
            "critical": {"score": round(critical * 100, 1), "weight": WEIGHTS["critical"]},
            "logic": {"score": round(logic * 100, 1), "weight": WEIGHTS["logic"]},
            "efficiency": {"score": round(efficiency * 100, 1), "weight": WEIGHTS["efficiency"]},
        }
    }

    return {"raw_response": json.dumps(result, ensure_ascii=False)}
