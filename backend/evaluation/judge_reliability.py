# -*- coding: utf-8 -*-
"""Judge 稳定性和人工校准 — 评估 LLM-as-Judge 的可靠性。

核心指标：
- repeat_agreement: 重复评分一致率（目标 ≥ 0.85）
- position_consistency: 位置一致率（目标 ≥ 0.90）
- score_std: 评分标准差
- human_agreement: 与人工标签的一致率（可选）

用法：
    from evaluation.judge_reliability import evaluate_judge_reliability

    rel = evaluate_judge_reliability(runs, human_labels=labels)
"""
from __future__ import annotations

import statistics
from typing import Optional

from pydantic import BaseModel, Field

# ── JudgeRun 数据模型 ────────────────────────────────────────────────────────


class JudgeRun(BaseModel):
    """单次 Judge 评分记录。"""

    case_id: str
    dimension: str
    score: float = Field(ge=0, le=100)
    position: str = "original"  # original / swapped
    judge_version: str = "judge_v1"
    model_family: str = "unknown"
    seed: Optional[int] = None
    degraded: bool = False


# ── JudgeReliability 结果 ────────────────────────────────────────────────────


class JudgeReliability(BaseModel):
    """Judge 可靠性评估结果。"""

    repeat_agreement: float = Field(ge=0, le=1, description="重复一致率")
    position_consistency: float = Field(ge=0, le=1, description="位置一致率")
    score_std: float = Field(ge=0, description="评分标准差")
    score_mean: float = Field(description="评分均值")
    human_agreement: Optional[float] = Field(default=None, ge=0, le=1, description="人工标签一致率")
    needs_review: bool = False
    degraded_count: int = 0
    n_runs: int = 0


# ── 评估函数 ─────────────────────────────────────────────────────────────────


def evaluate_judge_reliability(
    runs: list[JudgeRun],
    human_labels: Optional[list[float]] = None,
    threshold: float = 0.85,
) -> JudgeReliability:
    """评估 Judge 可靠性。

    Args:
        runs: Judge 评分记录列表。
        human_labels: 人工标签列表（可选，长度须与 runs 一致）。
        threshold: 一致率阈值，低于此值触发 needs_review。

    Returns:
        JudgeReliability 实例。
    """
    if not runs:
        return JudgeReliability(
            repeat_agreement=0.0,
            position_consistency=0.0,
            score_std=0.0,
            score_mean=0.0,
        )

    scores = [r.score for r in runs]
    score_mean = statistics.mean(scores)
    score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # 重复一致率：1 - normalized_std
    # std=0 → 完全一致(1.0)；std 越大一致率越低
    max_possible_std = 50.0  # 0-100 分制，最大合理标准差
    repeat_agreement = max(0.0, 1.0 - score_std / max_possible_std)

    # 位置一致率：同位置同分的比例
    positions: dict[str, list[float]] = {}
    for r in runs:
        positions.setdefault(r.position, []).append(r.score)

    if len(positions) <= 1:
        # 只有一种位置，视为完全一致
        position_consistency = 1.0
    else:
        # 计算各位置均分的差异
        pos_means = [statistics.mean(v) for v in positions.values()]
        pos_std = statistics.stdev(pos_means) if len(pos_means) > 1 else 0.0
        position_consistency = max(0.0, 1.0 - pos_std / max_possible_std)

    # 人工标签一致率
    human_agreement: Optional[float] = None
    if human_labels is not None and len(human_labels) == len(runs):
        diffs = [abs(r.score - h) for r, h in zip(runs, human_labels, strict=True)]
        avg_diff = statistics.mean(diffs)
        # 差异 0 → 1.0；差异 50+ → 0.0
        human_agreement = max(0.0, 1.0 - avg_diff / max_possible_std)

    # 降级计数
    degraded_count = sum(1 for r in runs if r.degraded)

    # needs_review 判定
    needs_review = repeat_agreement < threshold

    return JudgeReliability(
        repeat_agreement=round(repeat_agreement, 4),
        position_consistency=round(position_consistency, 4),
        score_std=round(score_std, 4),
        score_mean=round(score_mean, 4),
        human_agreement=round(human_agreement, 4) if human_agreement is not None else None,
        needs_review=needs_review,
        degraded_count=degraded_count,
        n_runs=len(runs),
    )
