# -*- coding: utf-8 -*-
"""综合评分智能体 — 代码计算三维加权评分，LLM生成综合评估摘要"""

import json
import re
import logging
from app.services.qwen_client import call_qwen_chat

# ── 权重配置 ──────────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "inquiry": 0.4,
    "knowledge": 0.3,
    "humanistic": 0.3,
}

# ── LLM Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一名临床问诊综合评估专家。你需要根据三个维度的评分和分析结果，生成一份综合评估摘要。

三个评估维度：
1. 问诊分析（权重40%）- 评估医生病史采集的系统性、完整性和逻辑性
2. 医学知识（权重30%）- 评估医生医学知识应用、诊断思维和治疗方案
3. 人文关怀（权重30%）- 评估医生沟通技巧、共情能力和人文关怀

请根据各维度评分和分析，撰写一份150-300字的综合评估摘要。摘要应包含：
- 整体表现概述：对医生本次问诊的整体评价
- 主要优点：医生表现突出的方面
- 主要不足：需要改进的方面
- 各维度表现排名：按得分从高到低排序

注意：
- 输出纯文本，不要使用任何Markdown格式符号（如*、#、-列表等）
- 直接输出JSON格式，包含 summary 字段
- 摘要应客观、专业、有建设性"""

# ── Few-shot 示例 ─────────────────────────────────────────────────

FEWSHOT_USER = """【维度1 - 问诊分析】
评分：72分
分析：问诊过程在病史采集方面表现中等偏上。医生大致遵循了从症状了解到检查解读的流程，详细询问了大便次数、性状、排气情况、腹痛部位和性质、体重变化，并追问了进食后症状加重的特点和生活方式。但未严格按照标准问诊顺序，既往史、个人史和家族史缺失。缺少对症状起始时间、诱因、加重/缓解因素的系统追问。

【维度2 - 医学知识】
评分：78分
分析：医学知识应用表现良好。医生正确识别了肠道功能紊乱的可能性，结合肠镜结果排除了器质性病变。对息肉切除后的随访安排合理（一年后复查肠镜）。对慢性胃炎的评估基本准确。但未详细评估肠易激综合征的Rome IV诊断标准，未考虑食物不耐受的可能，对胃炎的分型和Hp感染状态未做评估。

【维度3 - 人文关怀】
评分：73分
分析：沟通整体表现尚可。医生态度平和亲切，未出现不耐烦或打断患者的情况。对患者的提问均给予了回应。使用了通俗语言，避免了专业术语。但共情能力中等，对患者困惑回应过于简单，用药告知不充分，未说明药物名称、服用方法和注意事项。

请生成综合评估摘要。"""

FEWSHOT_ASSISTANT = """{"summary": "该医生在本次问诊中的综合表现良好，加权总分74分。各维度按权重计算：问诊分析72分x40%=28.8，医学知识78分x30%=23.4，人文关怀73分x30%=21.9。整体表现概述：本次问诊历时较长，医生较为系统地了解了患者症状特征和检查结果，做出了合理的临床判断，各维度表现均衡，医学知识是亮点。主要优点：症状采集较详细，涵盖大便频次、性状、排气、体重变化等关键指标；诊断推理清晰，能结合检查结果综合分析；医学知识扎实；态度友善，语言通俗。主要不足：病史采集缺少既往史、个人史、家族史；鉴别诊断不够全面；用药告知不充分；缺少饮食指导和短期随访计划。各维度表现排名（从高到低）：医学知识（78分）、人文关怀（73分）、问诊分析（72分）。建议重点加强问诊系统性训练和用药告知规范。"}"""


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


def _calculate_total_score(inquiry_score: int, knowledge_score: int, humanistic_score: int) -> int:
    """计算加权综合评分"""
    total = (
        SCORING_WEIGHTS["inquiry"] * inquiry_score +
        SCORING_WEIGHTS["knowledge"] * knowledge_score +
        SCORING_WEIGHTS["humanistic"] * humanistic_score
    )
    return round(total)


def _generate_fallback_summary(
    inquiry_score: int,
    inquiry_analysis: str,
    knowledge_score: int,
    knowledge_analysis: str,
    humanistic_score: int,
    humanistic_analysis: str,
    total_score: int,
) -> str:
    """当 LLM 调用失败时，生成简单的降级摘要"""
    # 排序维度
    dimensions = [
        ("问诊分析", inquiry_score),
        ("医学知识", knowledge_score),
        ("人文关怀", humanistic_score),
    ]
    dimensions.sort(key=lambda x: x[1], reverse=True)
    
    # 判断表现等级
    if total_score >= 90:
        level = "优秀"
    elif total_score >= 80:
        level = "良好"
    elif total_score >= 60:
        level = "一般"
    else:
        level = "不及格"
    
    # 生成简单摘要
    summary = (
        f"该医生在本次问诊中的综合表现为{level}，加权总分{total_score}分。"
        f"各维度按权重计算：问诊分析{inquiry_score}分x40%={inquiry_score * 0.4:.1f}，"
        f"医学知识{knowledge_score}分x30%={knowledge_score * 0.3:.1f}，"
        f"人文关怀{humanistic_score}分x30%={humanistic_score * 0.3:.1f}。"
        f"整体表现概述：本次问诊各维度表现{'较为均衡' if max(inquiry_score, knowledge_score, humanistic_score) - min(inquiry_score, knowledge_score, humanistic_score) < 15 else '存在差异'}，"
        f"{dimensions[0][0]}是亮点。"
        f"各维度表现排名（从高到低）：{dimensions[0][0]}（{dimensions[0][1]}分）、"
        f"{dimensions[1][0]}（{dimensions[1][1]}分）、{dimensions[2][0]}（{dimensions[2][1]}分）。"
    )
    
    return summary


# ── Main Function ─────────────────────────────────────────────────

async def run_scoring(
    inquiry_score: int,
    inquiry_analysis: str,
    knowledge_score: int,
    knowledge_analysis: str,
    humanistic_score: int,
    humanistic_analysis: str,
) -> dict:
    """
    综合评分智能体 — 代码计算三维加权评分，LLM生成综合评估摘要
    
    评分维度：
    1. Inquiry (40%): 问诊分析评分
    2. Knowledge (30%): 医学知识评分
    3. Humanistic (30%): 人文关怀评分
    
    Returns:
        dict: {"raw_response": json_string}
              json_string 包含 total_score 和 summary 字段
    """
    # ── Step 1: 代码计算加权综合评分 ──
    total_score = _calculate_total_score(inquiry_score, knowledge_score, humanistic_score)
    
    # ── Step 2: 调用 LLM 生成综合评估摘要 ──
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {
                "role": "user",
                "content": (
                    f"【维度1 - 问诊分析】\n"
                    f"评分：{inquiry_score}分\n"
                    f"分析：{inquiry_analysis}\n\n"
                    f"【维度2 - 医学知识】\n"
                    f"评分：{knowledge_score}分\n"
                    f"分析：{knowledge_analysis}\n\n"
                    f"【维度3 - 人文关怀】\n"
                    f"评分：{humanistic_score}分\n"
                    f"分析：{humanistic_analysis}\n\n"
                    f"请生成综合评估摘要。"
                ),
            },
        ]
        
        result = await call_qwen_chat(messages, temperature=0.3)
        llm_data = _extract_json(result)
        summary = llm_data.get("summary", "")
        
        if not summary:
            raise ValueError("LLM 返回的 summary 为空")
            
    except Exception as e:
        logging.error(f"LLM 摘要生成失败: {e}，使用降级摘要")
        # 降级处理：使用代码生成简单摘要
        summary = _generate_fallback_summary(
            inquiry_score, inquiry_analysis,
            knowledge_score, knowledge_analysis,
            humanistic_score, humanistic_analysis,
            total_score,
        )
    
    # ── Step 3: 构建返回结果 ──
    result = {
        "total_score": total_score,
        "summary": summary,
        "weights": SCORING_WEIGHTS,
        "dimension_scores": {
            "inquiry": inquiry_score,
            "knowledge": knowledge_score,
            "humanistic": humanistic_score,
        },
    }
    
    return {"raw_response": json.dumps(result, ensure_ascii=False)}
