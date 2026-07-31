# -*- coding: utf-8 -*-
"""人工复核状态机和审计 — 可追踪的复核流程。

核心语义：
- 状态迁移：pending_review → in_review → approved/rejected/returned
- 高风险 approve 必须填写 reason
- 原始评估不可变（snapshot 隔离）
- 每次调整可还原

用法：
    from evaluation.review_audit import apply_review_decision, create_review_snapshot

    snapshot = create_review_snapshot(evaluation)
    result = apply_review_decision(evaluation, decision)
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── 状态枚举 ─────────────────────────────────────────────────────────────────


class ReviewStatus(str, Enum):
    """复核状态。"""

    PENDING = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETURNED = "returned"


# 合法状态迁移
_VALID_TRANSITIONS = {
    "pending_review": {"in_review"},
    "in_review": {"approved", "rejected", "returned"},
    "approved": set(),
    "rejected": set(),
    "returned": {"in_review"},  # returned 可重新进入复核
}


# ── ReviewDecision 模型 ──────────────────────────────────────────────────────


class ReviewDecision(BaseModel):
    """复核决策。"""

    reviewer_id: str
    status: str  # approved / rejected / returned
    reason_code: str = "confirmed"
    reason: str = ""
    adjusted_scores: dict[str, float] = Field(default_factory=dict)
    rubric_adjustments: list[dict] = Field(default_factory=list)


# ── 状态迁移验证 ─────────────────────────────────────────────────────────────


def validate_review_transition(from_status: str, to_status: str) -> bool:
    """验证状态迁移是否合法。

    Args:
        from_status: 当前状态。
        to_status: 目标状态。

    Returns:
        True 如果迁移合法。
    """
    allowed = _VALID_TRANSITIONS.get(from_status, set())
    return to_status in allowed


# ── 创建复核快照 ─────────────────────────────────────────────────────────────


def create_review_snapshot(evaluation: dict) -> dict:
    """创建评估的不可变快照。

    Args:
        evaluation: 评估字典。

    Returns:
        包含 original_scores 的快照字典。
    """
    score_fields = [
        "inquiry_score", "knowledge_score", "humanistic_score",
        "diagnosis_score", "treatment_score", "total_score",
    ]
    original_scores = {k: evaluation.get(k) for k in score_fields if k in evaluation}

    return {
        "evaluation_id": evaluation.get("id"),
        "consultation_id": evaluation.get("consultation_id"),
        "risk_level": evaluation.get("risk_level", "low"),
        "original_scores": copy.deepcopy(original_scores),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── 应用复核决策 ─────────────────────────────────────────────────────────────


def apply_review_decision(
    evaluation: dict,
    decision: ReviewDecision,
) -> dict:
    """应用复核决策，返回结果字典。

    Args:
        evaluation: 原始评估字典。
        decision: 复核决策。

    Returns:
        包含 original_scores、adjusted_scores、final_status 的字典。

    Raises:
        ValueError: 高风险 approve 无 reason 时。
    """
    risk_level = evaluation.get("risk_level", "low")

    # 高风险 approve 必须有 reason
    if decision.status == "approved" and risk_level == "high" and not decision.reason:
        raise ValueError("高风险 approve 必须填写 reason")

    # 创建快照
    snapshot = create_review_snapshot(evaluation)

    # 计算调整后分数
    adjusted_scores = copy.deepcopy(snapshot["original_scores"])
    for key, val in decision.adjusted_scores.items():
        if key in adjusted_scores:
            adjusted_scores[key] = val

    return {
        "evaluation_id": snapshot["evaluation_id"],
        "reviewer_id": decision.reviewer_id,
        "final_status": decision.status,
        "original_scores": snapshot["original_scores"],
        "adjusted_scores": adjusted_scores,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
