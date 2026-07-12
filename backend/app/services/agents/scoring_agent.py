# -*- coding: utf-8 -*-
"""综合评分智能体 — 兼容 Facade，内部调用 ScoringEngine

保留旧接口签名和行为（包括 None 权重重分配），确保基线测试兼容。
新引擎 ScoreCalculator 使用更严格的策略（不重分配），后续逐步迁移。
"""

import json
import logging
from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_from_text
from app.services.scoring.policies import get_default_policy, ScoringPolicy
from app.services.scoring.calculator import ScoreCalculator, DimensionResult
from app.services.scoring.summary import SummaryGenerator

# ── 保留旧常量引用（兼容基线测试 import） ──────────────────────────
SCORING_WEIGHTS = get_default_policy().weights

# ── 维度中文名映射（保留旧引用） ──────────────────────────────────
_DIMENSION_NAMES = get_default_policy().dimension_names


# ── Helper Functions（保留旧行为） ──────────────────────────────────

def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON"""
    return extract_json_from_text(text)


def calculate_total(scores: dict) -> int | None:
    """计算加权总分，None 维度不参与，权重重分配（保留旧行为）

    注意：此函数保留旧的权重重分配逻辑以兼容基线测试。
    新引擎 ScoreCalculator 不做重分配，部分维度缺失时返回 None。
    """
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return None

    total_weight = sum(SCORING_WEIGHTS[k] for k in valid)
    if total_weight == 0:
        return None

    weighted_sum = sum(valid[k] * SCORING_WEIGHTS[k] for k in valid)
    return round(weighted_sum / total_weight)


def _generate_fallback_summary(
    scores: dict,
    analyses: dict,
    total_score: int | None,
) -> str:
    """当 LLM 调用失败时，生成包含五个维度的降级摘要（保留旧行为）"""
    valid_dims = []
    skipped_dims = []
    for key in SCORING_WEIGHTS:
        if scores.get(key) is not None:
            valid_dims.append((_DIMENSION_NAMES[key], scores[key]))
        else:
            skipped_dims.append(_DIMENSION_NAMES[key])

    if total_score is None:
        return "所有维度均未评估，无法生成综合摘要。"

    valid_dims.sort(key=lambda x: x[1], reverse=True)

    if total_score >= 90:
        level = "优秀"
    elif total_score >= 80:
        level = "良好"
    elif total_score >= 60:
        level = "一般"
    else:
        level = "不及格"

    weight_details = ""
    for key in SCORING_WEIGHTS:
        if scores.get(key) is not None:
            name = _DIMENSION_NAMES[key]
            s = scores[key]
            w = SCORING_WEIGHTS[key]
            weight_details += f"{name}{s}分x{int(w*100)}%={s * w:.1f}，"
    weight_details = weight_details.rstrip("，")

    ranking = "、".join(f"{name}（{score}分）" for name, score in valid_dims)

    skipped_note = ""
    if skipped_dims:
        skipped_note = f"以下维度未评估（证据不足）：{'、'.join(skipped_dims)}。"

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


# ── LLM Prompts（保留旧引用，供外部直接使用） ─────────────────────
# 从 scoring_agent.py 原始定义复制，保持向后兼容
from app.services.scoring.summary import SYSTEM_PROMPT, FEWSHOT_USER, FEWSHOT_ASSISTANT  # noqa: E402


# ── Main Function（保留旧签名） ────────────────────────────────────

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
