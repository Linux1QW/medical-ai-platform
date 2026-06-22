# -*- coding: utf-8 -*-
"""综合评分智能体 — 代码计算五维加权评分，LLM生成综合评估摘要"""

import json
import re
import logging
from app.services.qwen_client import call_qwen_chat

# ── 权重配置 ──────────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "inquiry": 0.25,
    "knowledge": 0.25,
    "humanistic": 0.20,
    "diagnosis": 0.15,
    "treatment": 0.15,
}

# ── LLM Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一名临床问诊综合评估专家。你需要根据五个维度的评分和分析结果，生成一份综合评估摘要。

五个评估维度：
1. 问诊技巧评估（权重25%）- 评估医生病史采集的系统性、完整性和逻辑性
2. 医学知识评估（权重25%）- 评估医生医学知识应用和临床推理能力
3. 人文关怀评估（权重20%）- 评估医生沟通技巧、共情能力和人文关怀
4. 诊断能力评估（权重15%）- 评估医生诊断思维的准确性和全面性
5. 治疗方案评估（权重15%）- 评估医生治疗方案的合理性和规范性

请根据各维度评分和分析，撰写一份150-300字的综合评估摘要。摘要应包含：
- 整体表现概述：对医生本次问诊的整体评价
- 主要优点：医生表现突出的方面
- 主要不足：需要改进的方面
- 各维度表现排名：按得分从高到低排序

注意：
- 如果某个维度标注为"未评估"，请在摘要中说明该维度未参与评估，不计入排名
- 输出纯文本，不要使用任何Markdown格式符号（如*、#、-列表等）
- 直接输出JSON格式，包含 summary 字段
- 摘要应客观、专业、有建设性"""

# ── Few-shot 示例 ─────────────────────────────────────────────────

FEWSHOT_USER = """【维度1 - 问诊技巧评估】
评分：72分
分析：问诊过程在病史采集方面表现中等偏上。医生大致遵循了从症状了解到检查解读的流程，详细询问了大便次数、性状、排气情况、腹痛部位和性质、体重变化，并追问了进食后症状加重的特点和生活方式。但未严格按照标准问诊顺序，既往史、个人史和家族史缺失。

【维度2 - 医学知识评估】
评分：78分
分析：医学知识应用表现良好。医生正确识别了肠道功能紊乱的可能性，结合肠镜结果排除了器质性病变。对息肉切除后的随访安排合理。但未详细评估肠易激综合征的Rome IV诊断标准，未考虑食物不耐受的可能。

【维度3 - 人文关怀评估】
评分：73分
分析：沟通整体表现尚可。医生态度平和亲切，未出现不耐烦或打断患者的情况。使用了通俗语言，避免了专业术语。但共情能力中等，用药告知不充分。

【维度4 - 诊断能力评估】
评分：75分
分析：诊断思维基本合理，能结合患者症状和检查结果进行综合分析，但鉴别诊断范围不够广，未充分考虑功能性胃肠病的亚型分类。

【维度5 - 治疗方案评估】
评分：70分
分析：治疗方案大体合理，药物选择基本恰当，但缺少非药物治疗建议，如饮食调整、心理疏导和生活方式改善等具体措施。

请生成综合评估摘要。"""

FEWSHOT_ASSISTANT = """{"summary": "该医生在本次问诊中的综合表现良好，加权总分73分。各维度按权重计算：问诊技巧72分x25%=18.0，医学知识78分x25%=19.5，人文关怀73分x20%=14.6，诊断能力75分x15%=11.3，治疗方案70分x15%=10.5。整体表现概述：本次问诊医生较为系统地了解了患者症状特征和检查结果，做出了合理的临床判断，各维度表现较为均衡，医学知识是亮点。主要优点：症状采集较详细，涵盖大便频次、性状、排气、体重变化等关键指标；诊断推理清晰，能结合检查结果综合分析；态度友善，语言通俗。主要不足：病史采集缺少既往史、个人史、家族史；鉴别诊断不够全面；治疗方案缺少非药物干预建议；用药告知不充分。各维度表现排名（从高到低）：医学知识（78分）、诊断能力（75分）、人文关怀（73分）、问诊技巧（72分）、治疗方案（70分）。建议重点加强问诊系统性训练、完善鉴别诊断和丰富治疗方案。"}"""


# ── Helper Functions ──────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON"""
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


def calculate_total(scores: dict) -> int | None:
    """计算加权总分，None 维度不参与，权重重分配"""
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return None

    total_weight = sum(SCORING_WEIGHTS[k] for k in valid)
    if total_weight == 0:
        return None

    weighted_sum = sum(valid[k] * SCORING_WEIGHTS[k] for k in valid)
    return round(weighted_sum / total_weight)


# 维度中文名映射
_DIMENSION_NAMES = {
    "inquiry": "问诊技巧评估",
    "knowledge": "医学知识评估",
    "humanistic": "人文关怀评估",
    "diagnosis": "诊断能力评估",
    "treatment": "治疗方案评估",
}


def _generate_fallback_summary(
    scores: dict,
    analyses: dict,
    total_score: int | None,
) -> str:
    """当 LLM 调用失败时，生成包含五个维度的降级摘要"""
    # 分离有效维度和未评估维度
    valid_dims = []
    skipped_dims = []
    for key in SCORING_WEIGHTS:
        if scores.get(key) is not None:
            valid_dims.append((_DIMENSION_NAMES[key], scores[key]))
        else:
            skipped_dims.append(_DIMENSION_NAMES[key])

    if total_score is None:
        return "所有维度均未评估，无法生成综合摘要。"

    # 排序有效维度
    valid_dims.sort(key=lambda x: x[1], reverse=True)

    # 判断表现等级
    if total_score >= 90:
        level = "优秀"
    elif total_score >= 80:
        level = "良好"
    elif total_score >= 60:
        level = "一般"
    else:
        level = "不及格"

    # 权重明细
    weight_details = ""
    for key in SCORING_WEIGHTS:
        if scores.get(key) is not None:
            name = _DIMENSION_NAMES[key]
            s = scores[key]
            w = SCORING_WEIGHTS[key]
            weight_details += f"{name}{s}分x{int(w*100)}%={s * w:.1f}，"
    weight_details = weight_details.rstrip("，")

    # 排名
    ranking = "、".join(f"{name}（{score}分）" for name, score in valid_dims)

    # 未评估维度说明
    skipped_note = ""
    if skipped_dims:
        skipped_note = f"以下维度未评估（证据不足）：{'、'.join(skipped_dims)}。"

    # 差异判断
    if len(valid_dims) >= 2:
        diff = valid_dims[0][1] - valid_dims[-1][1]
        balance = "较为均衡" if diff < 15 else "存在差异"
    else:
        balance = "仅有单一维度参与评估"

    summary = (
        f"该医生在本次问诊中的综合表现为{level}，加权总分{total_score}分。"
        f"各维度按权重计算：{weight_details}。"
        f"整体表现概述：本次问诊各维度表现{balance}，"
        f"{valid_dims[0][0]}是亮点。"
        f"各维度表现排名（从高到低）：{ranking}。"
        f"{skipped_note}"
    )

    return summary


# ── Main Function ─────────────────────────────────────────────────

async def run_scoring(
    inquiry_score: float,
    inquiry_analysis: str,
    knowledge_score: float | None,
    knowledge_analysis: str,
    humanistic_score: float,
    humanistic_analysis: str,
    diagnosis_score: float | None = None,
    diagnosis_analysis: str = "",
    treatment_score: float | None = None,
    treatment_analysis: str = "",
) -> dict:
    """
    综合评分智能体 — 代码计算五维加权评分，LLM生成综合评估摘要

    评分维度：
    1. Inquiry (25%): 问诊技巧评估评分
    2. Knowledge (25%): 医学知识评估评分（可为 None，拒答时不参与加权）
    3. Humanistic (20%): 人文关怀评估评分
    4. Diagnosis (15%): 诊断能力评估评分（可为 None）
    5. Treatment (15%): 治疗方案评估评分（可为 None）

    Returns:
        dict: {"raw_response": json_string, "total_score": int|None, "summary": str}
    """
    # ── Step 1: 收集各维度分数，计算加权综合评分 ──
    scores = {
        "inquiry": inquiry_score,
        "knowledge": knowledge_score,
        "humanistic": humanistic_score,
        "diagnosis": diagnosis_score,
        "treatment": treatment_score,
    }
    analyses = {
        "inquiry": inquiry_analysis,
        "knowledge": knowledge_analysis,
        "humanistic": humanistic_analysis,
        "diagnosis": diagnosis_analysis,
        "treatment": treatment_analysis,
    }
    total_score = calculate_total(scores)

    # ── Step 2: 构建 LLM 用户消息（覆盖五个维度） ──
    dimension_blocks = []
    dimension_labels = [
        ("inquiry", "问诊技巧评估"),
        ("knowledge", "医学知识评估"),
        ("humanistic", "人文关怀评估"),
        ("diagnosis", "诊断能力评估"),
        ("treatment", "治疗方案评估"),
    ]
    for i, (key, label) in enumerate(dimension_labels, 1):
        s = scores[key]
        a = analyses[key]
        if s is not None:
            block = f"【维度{i} - {label}】\n评分：{s}分\n分析：{a}"
        else:
            block = f"【维度{i} - {label}】\n该维度未评估（证据不足），不计入加权总分。"
        dimension_blocks.append(block)

    user_content = "\n\n".join(dimension_blocks) + "\n\n请生成综合评估摘要。"

    # ── Step 3: 调用 LLM 生成综合评估摘要 ──
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ]

        result = await call_qwen_chat(messages, temperature=0.3)
        llm_data = _extract_json(result)
        summary = llm_data.get("summary", "")

        if not summary:
            raise ValueError("LLM 返回的 summary 为空")

    except Exception as e:
        logging.error(f"LLM 摘要生成失败: {e}，使用降级摘要")
        summary = _generate_fallback_summary(scores, analyses, total_score)

    # ── Step 4: 构建返回结果 ──
    result = {
        "total_score": total_score,
        "summary": summary,
        "weights": SCORING_WEIGHTS,
        "dimension_scores": scores,
    }

    return {"raw_response": json.dumps(result, ensure_ascii=False), "total_score": total_score, "summary": summary}
