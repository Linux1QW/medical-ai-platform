# -*- coding: utf-8 -*-
"""Task 13 — 复核反馈候选样本模型与归因逻辑

将 ReviewRecord + Evaluation 转换为去标识的 ReviewFeedbackCandidate，
供人工挑选后补齐 gold annotation 生成 RagGoldCase(split="regression")。

隐私策略：
- 默认导出只含去标识元数据、分数、反馈脱敏文本和 citation IDs
- consultation_id / user_id 使用 HMAC-SHA256，不可反查
- 敏感内容导出需同时提供 --include-content --acknowledge-sensitive-data
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.observability.trace_context import sanitize_for_observability

logger = logging.getLogger(__name__)


# ── 数据模型 ──────────────────────────────────────────────────────────────────


class ReviewFeedbackCandidate(BaseModel):
    """复核反馈候选样本

    去标识化后的结构化反馈记录，供人工挑选并标注为 gold case。
    """

    candidate_id: str
    review_id: str
    run_id: str
    consultation_id_hash: str
    department: str | None = None
    review_reason: str | None = None
    original_scores: dict[str, float | None] = Field(default_factory=dict)
    adjusted_scores: dict[str, float | None] = Field(default_factory=dict)
    feedback: str = ""
    retrieval_status: str = "not_run"
    evidence_stance: str = "undetermined"
    citation_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    contains_sensitive_content: bool = False


# ── 导出配置 ──────────────────────────────────────────────────────────────────


@dataclass
class ExportConfig:
    """导出配置"""

    hmac_key: str
    include_content: bool = False
    acknowledge_sensitive_data: bool = False
    output_path: str = ""
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None

    def can_export_sensitive(self) -> bool:
        """是否允许导出受限内容（需双确认）"""
        if not self.include_content:
            return True
        return self.include_content and self.acknowledge_sensitive_data


# ── 归因常量 ──────────────────────────────────────────────────────────────────

ATTRIBUTION_KNOWLEDGE_SCORING_GAP = "knowledge_scoring_gap"
ATTRIBUTION_RETRIEVAL_INSUFFICIENT = "retrieval_insufficient"
ATTRIBUTION_CITATION_QUALITY = "citation_quality"
ATTRIBUTION_REVIEW_FEEDBACK = "review_feedback"

_KNOWLEDGE_GAP_THRESHOLD = 10.0

# review_reason 关键词到归因的映射
_REASON_ATTRIBUTION_MAP: dict[str, str] = {
    "evidence_insufficient": ATTRIBUTION_RETRIEVAL_INSUFFICIENT,
    "evidence": ATTRIBUTION_RETRIEVAL_INSUFFICIENT,
    "citation_mismatch": ATTRIBUTION_CITATION_QUALITY,
    "citation_error": ATTRIBUTION_CITATION_QUALITY,
    "citation": ATTRIBUTION_CITATION_QUALITY,
}


# ── 核心函数 ──────────────────────────────────────────────────────────────────


def make_candidate_id(review_id: str) -> str:
    """从 review_id 生成稳定的 candidate_id（HMAC-SHA256 hex）

    使用固定命名空间前缀确保与 consultation_id hash 不冲突。
    """
    namespace = b"review-feedback-candidate-v1:"
    digest = hmac.new(namespace, review_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest


def hash_consultation_id(consultation_id: int, hmac_key: str) -> str:
    """对 consultation_id 计算 HMAC-SHA256，不可反查"""
    key_bytes = hmac_key.encode("utf-8")
    id_bytes = str(consultation_id).encode("utf-8")
    return hmac.new(key_bytes, id_bytes, hashlib.sha256).hexdigest()


def sanitize_feedback_text(text: str) -> str:
    """脱敏反馈文本（复用 sanitize_for_observability）"""
    if not text:
        return text
    result = sanitize_for_observability(text)
    return result if isinstance(result, str) else str(result)


def classify_attribution(
    original_scores: dict[str, float | None],
    adjusted_scores: dict[str, float | None],
    review_reason: str | None,
) -> str:
    """按规则归因

    优先级：
    1. 知识分调整大于 10 分 → knowledge_scoring_gap
    2. review_reason 含证据不足关键词 → retrieval_insufficient
    3. review_reason 含引用不一致关键词 → citation_quality
    4. 其他 → review_feedback
    """
    # 1. 知识分差距检查
    orig_knowledge = original_scores.get("knowledge_score")
    adj_knowledge = adjusted_scores.get("knowledge_score")
    if orig_knowledge is not None and adj_knowledge is not None:
        delta = abs(adj_knowledge - orig_knowledge)
        if delta > _KNOWLEDGE_GAP_THRESHOLD:
            return ATTRIBUTION_KNOWLEDGE_SCORING_GAP

    # 2-3. review_reason 关键词匹配
    if review_reason:
        reason_lower = review_reason.lower()
        for keyword, attribution in _REASON_ATTRIBUTION_MAP.items():
            if keyword in reason_lower:
                return attribution

    # 4. 默认
    return ATTRIBUTION_REVIEW_FEEDBACK


def _extract_original_scores(review: Any, evaluation: Any) -> dict[str, float | None]:
    """提取原始五维分数（优先用 ReviewRecord 快照，回退 Evaluation）"""
    if review.original_scores:
        return dict(review.original_scores)
    return {
        "inquiry_score": evaluation.inquiry_score,
        "knowledge_score": evaluation.knowledge_score,
        "humanistic_score": evaluation.humanistic_score,
        "diagnosis_score": evaluation.diagnosis_score,
        "treatment_score": evaluation.treatment_score,
    }


def _extract_adjusted_scores(review: Any, original: dict[str, float | None]) -> dict[str, float | None]:
    """提取调整后分数（ReviewRecord.score_adjustments 覆盖原始）"""
    adjusted = dict(original)
    if review.score_adjustments:
        for key, value in review.score_adjustments.items():
            if value is not None:
                adjusted[key] = float(value)
    return adjusted


def _extract_citation_ids(evaluation: Any) -> list[str]:
    """从 evaluation.citation_data 提取 citation IDs"""
    citation_data = evaluation.citation_data or []
    ids: list[str] = []
    for item in citation_data:
        if isinstance(item, dict):
            cid = item.get("citation_id") or item.get("id")
            if cid:
                ids.append(str(cid))
    return ids


def _detect_sensitive_content(feedback: str) -> bool:
    """简单检测反馈是否可能含敏感内容（脱敏前检测）"""
    import re

    # 手机号模式
    if re.search(r"1[3-9]\d{9}", feedback):
        return True
    # 身份证模式
    if re.search(r"\d{17}[\dXx]", feedback):
        return True
    # 姓名模式
    if re.search(r"(?:姓名|患者)[：:\s]*[\u4e00-\u9fff]{2,4}", feedback):
        return True
    return False


def build_candidate(
    review: Any,
    evaluation: Any,
    hmac_key: str,
) -> ReviewFeedbackCandidate:
    """从 ReviewRecord + Evaluation 构造 ReviewFeedbackCandidate

    Args:
        review: ReviewRecord ORM 实例
        evaluation: Evaluation ORM 实例
        hmac_key: FEEDBACK_EXPORT_HMAC_KEY 环境变量值

    Returns:
        去标识化的候选样本
    """
    # 检测原始反馈是否含敏感内容
    contains_sensitive = _detect_sensitive_content(review.feedback or "")

    # 脱敏反馈文本
    sanitized_feedback = sanitize_feedback_text(review.feedback or "")

    # 分数
    original = _extract_original_scores(review, evaluation)
    adjusted = _extract_adjusted_scores(review, original)

    # 归因
    _attribution = classify_attribution(
        original, adjusted, review.review_reason
    )

    # citation IDs
    citation_ids = _extract_citation_ids(evaluation)

    return ReviewFeedbackCandidate(
        candidate_id=make_candidate_id(review.id),
        review_id=review.id,
        run_id=evaluation.run_id or "",
        consultation_id_hash=hash_consultation_id(evaluation.consultation_id, hmac_key),
        department=None,  # Evaluation 无 department 字段
        review_reason=review.review_reason,
        original_scores=original,
        adjusted_scores=adjusted,
        feedback=sanitized_feedback,
        retrieval_status=evaluation.retrieval_status or "not_run",
        evidence_stance=evaluation.evidence_stance or "undetermined",
        citation_ids=citation_ids,
        created_at=review.created_at or datetime.utcnow(),
        contains_sensitive_content=contains_sensitive,
    )
