# -*- coding: utf-8 -*-
"""五维原子 Rubric — 可核查的临床行为项评分。

将问诊分析、医学知识、人文关怀、诊断评估、治疗方案从"单分数"升级为
5~8 个可核查的原子行为项（rubric item），每项有 verdict、score、severity。

核心语义：
- unassessed 不得被聚合为 0 分
- high severity + fail → 自动 review_required
- 全部 unassessed → 维度 insufficient

用法：
    from evaluation.rubric import aggregate_rubric, load_rubric_v1

    rubric = load_rubric_v1()
    result = aggregate_rubric(items, dimension="inquiry")
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Verdict 枚举 ─────────────────────────────────────────────────────────────


class RubricVerdict(str, Enum):
    """Rubric item 判定结果。

    pass: 完全满足。
    partial: 部分满足。
    fail: 不满足。
    not_applicable: 不适用（如维度不涉及）。
    unassessed: 未评估（信息不足或跳过）。
    """

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNASSESSED = "unassessed"


# ── RubricItem 数据模型 ──────────────────────────────────────────────────────


class RubricItem(BaseModel):
    """单个原子行为项评估结果。"""

    item_id: str
    dimension: str
    description: str = ""
    verdict: RubricVerdict
    score: Optional[float] = Field(default=None, ge=0, le=100)
    severity: str = "normal"  # normal / high
    evidence_spans: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _validate_verdict(cls, v: str) -> str:
        valid = {k.value for k in RubricVerdict}
        if v not in valid:
            raise ValueError(f"非法 verdict: {v!r}，合法值: {sorted(valid)}")
        return v


# ── 维度聚合结果 ─────────────────────────────────────────────────────────────


class DimensionResult(BaseModel):
    """单维度聚合结果。"""

    dimension: str
    status: str = "scored"  # scored / insufficient
    score: Optional[float] = Field(default=None, ge=0, le=100)
    items_assessed: int = 0
    items_total: int = 0
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)


# ── RubricSet（v1 JSON 加载结构）────────────────────────────────────────────


class RubricDefinition(BaseModel):
    """Rubric 定义模板（无 verdict/score）。"""

    item_id: str
    dimension: str
    description: str = ""
    severity: str = "normal"


class RubricSet(BaseModel):
    """五维 Rubric 定义集合。"""

    version: str
    description: str = ""
    dimensions: dict[str, list[RubricDefinition]]


# ── 聚合函数 ─────────────────────────────────────────────────────────────────


def aggregate_rubric(
    items: list[RubricItem],
    dimension: str,
) -> DimensionResult:
    """聚合某维度的 rubric items 为 DimensionResult。

    规则：
    1. unassessed / not_applicable 不参与分数计算
    2. 全部 unassessed → insufficient
    3. high severity + fail → review_required

    Args:
        items: 该维度的 rubric items。
        dimension: 维度名称。

    Returns:
        DimensionResult。
    """
    if not items:
        return DimensionResult(dimension=dimension, status="insufficient", score=None)

    # 过滤可计分项（排除 unassessed 和 not_applicable）
    scored = [
        it for it in items
        if it.verdict not in (RubricVerdict.UNASSESSED, RubricVerdict.NOT_APPLICABLE)
    ]

    result = DimensionResult(dimension=dimension, items_total=len(items))

    if not scored:
        result.status = "insufficient"
        result.score = None
        return result

    # 计算均分
    total = sum(it.score for it in scored if it.score is not None)
    result.score = round(total / len(scored), 2) if scored else None
    result.items_assessed = len(scored)
    result.status = "scored"

    # high severity fail → review
    for it in scored:
        if it.severity == "high" and it.verdict == RubricVerdict.FAIL:
            result.review_required = True
            result.review_reasons.append(
                f"high severity item {it.item_id} failed: {it.description}"
            )

    return result


# ── 校验函数 ─────────────────────────────────────────────────────────────────


def validate_rubric_items(items: list[RubricItem]) -> list[str]:
    """校验 rubric items 列表，返回错误/警告（空 = 无问题）。

    检查：
    1. 重复 item_id
    2. unassessed 但带了 score
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    for it in items:
        # 重复 ID
        if it.item_id in seen_ids:
            errors.append(f"duplicate item_id: {it.item_id}")
        seen_ids.add(it.item_id)

        # unassessed 但带了 score
        if it.verdict == RubricVerdict.UNASSESSED and it.score is not None:
            errors.append(
                f"item {it.item_id} is unassessed but has score={it.score}"
            )

    return errors


# ── v1 JSON 加载 ─────────────────────────────────────────────────────────────


_RUBRICS_DIR = Path(__file__).parent / "rubrics"


def load_rubric_v1(path: Path | None = None) -> RubricSet:
    """加载 v1 rubric 定义。

    Args:
        path: 自定义路径，默认使用内置 rubrics/v1.json。

    Returns:
        RubricSet 实例。
    """
    if path is None:
        path = _RUBRICS_DIR / "v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    # 将维度下的 dict 列表转为 RubricDefinition
    dimensions: dict[str, list[RubricDefinition]] = {}
    for dim_name, items in data.get("dimensions", {}).items():
        dimensions[dim_name] = [RubricDefinition(**it) for it in items]

    return RubricSet(
        version=data.get("version", "v1"),
        description=data.get("description", ""),
        dimensions=dimensions,
    )
