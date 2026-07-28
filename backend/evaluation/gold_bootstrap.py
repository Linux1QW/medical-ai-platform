# -*- coding: utf-8 -*-
"""gold_bootstrap — 为 dataset 转换用例自动生成 gold 检索期望建议

dataset/ 病例不含检索期望字段，人工逐例标注成本高。本模块按
主诊断/主诉关键词规则表，将病例映射到 data/ 目录指南文档的文件名
子串（gold_relevant_sources），供 source_hit_rate 指标消费——判定
口径与 scripts/eval/golden_set.json 的 relevant_source_contains 一致：
citation 的 source 文件名包含任一子串即算命中。

自动生成结果仅为"建议值"，notes 追加 gold=auto-suggested 标记，
供后续人工修正；未命中任何规则的病例保持 gold 字段为空（不参与
source_hit_rate 计算，不会拉低指标）。
"""
import logging
from typing import List, Tuple

from .datasets import RagGoldCase

logger = logging.getLogger(__name__)

# 标记串：写入 notes，供幂等判断与人工筛查
AUTO_SUGGESTED_MARK = "gold=auto-suggested"

# 规则表：(触发关键词, 期望来源文件名子串)
# 匹配对象为「主诊断 + 主诉」拼接文本；多条规则命中时取并集（保持顺序去重），
# 覆盖"慢性胃炎,慢性结肠炎"等复合诊断。子串对应 data/ 真实指南文件名
# （如"3.内科学 消化内科分册.pdf"、"诊断学 (万学红...).pdf"）。
_GOLD_SOURCE_RULES: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    # 消化系统疾病（dataset 主体：胃炎/结肠炎/反流/腹痛腹泻等）
    (
        ("胃", "肠", "腹", "反酸", "嗳气", "恶心", "呕吐", "便秘",
         "黑便", "反流", "幽门螺杆菌", "消化", "食管炎"),
        ("消化内科", "内科学"),
    ),
    # 一般体检 / 健康咨询类
    (
        ("一般性医学检查", "体检", "健康检查"),
        ("诊断学", "全科医学"),
    ),
]


def suggest_gold_sources(text: str) -> List[str]:
    """按规则表为病例文本生成期望来源子串（并集，保持规则顺序去重）"""
    if not text:
        return []
    suggested: List[str] = []
    for keywords, sources in _GOLD_SOURCE_RULES:
        if any(kw in text for kw in keywords):
            for src in sources:
                if src not in suggested:
                    suggested.append(src)
    return suggested


def bootstrap_gold_case(case: RagGoldCase) -> RagGoldCase:
    """为单个用例补充 gold_relevant_sources 建议（幂等，原对象不变）

    已有人工/既往标注（gold_relevant_sources 非空）或已带自动标记的
    用例原样返回，不覆盖。
    """
    if case.gold_relevant_sources:
        return case
    if case.notes and AUTO_SUGGESTED_MARK in case.notes:
        return case

    match_text = f"{case.doctor_diagnosis or ''} {case.chief_complaint or ''}"
    suggested = suggest_gold_sources(match_text)
    if not suggested:
        return case

    notes = f"{case.notes}; {AUTO_SUGGESTED_MARK}" if case.notes else AUTO_SUGGESTED_MARK
    return case.model_copy(update={
        "gold_relevant_sources": suggested,
        "notes": notes,
    })


def bootstrap_gold_cases(cases: List[RagGoldCase]) -> Tuple[List[RagGoldCase], int]:
    """批量补充 gold 建议

    Returns:
        (处理后的用例列表, 本次新增标注的用例数)
    """
    bootstrapped: List[RagGoldCase] = []
    annotated = 0
    for case in cases:
        updated = bootstrap_gold_case(case)
        if updated is not case:
            annotated += 1
        bootstrapped.append(updated)
    if annotated:
        logger.info(f"gold_bootstrap: 为 {annotated}/{len(cases)} 个用例生成 gold 来源建议")
    return bootstrapped, annotated
