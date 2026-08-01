# -*- coding: utf-8 -*-
"""Claim-Evidence Graph — 临床主张与证据关联。

将知识 Agent 的诊断、治疗、风险和教育结论拆为 claims，
每个 claim 绑定 supports / contradicts / insufficient 的 evidence link。

核心语义：
- unsupported 的 treatment/risk claim → needs_review
- 证据冲突不能标记为 supported
- 可统计 claim coverage 和 contradiction rate

用法：
    from evaluation.rag_claims import validate_claim_evidence, calculate_claim_metrics
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ── Claim 状态枚举 ───────────────────────────────────────────────────────────


class ClaimStatus(str, Enum):
    """Claim 证据状态。"""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


# ── EvidenceLink ─────────────────────────────────────────────────────────────


class EvidenceLink(BaseModel):
    """证据链接。"""

    citation_id: str
    link_type: str  # supports / contradicts / insufficient
    text_snippet: str = ""
    entailment_score: float = Field(default=0.5, ge=0, le=1)

    @field_validator("link_type")
    @classmethod
    def _validate_link_type(cls, v: str) -> str:
        valid = {"supports", "contradicts", "insufficient"}
        if v not in valid:
            raise ValueError(f"非法 link_type: {v!r}，合法值: {sorted(valid)}")
        return v


# ── ClinicalClaim ────────────────────────────────────────────────────────────


class ClinicalClaim(BaseModel):
    """临床主张。"""

    claim_id: str
    claim_type: str  # diagnosis / treatment / risk / education
    text: str = ""
    status: ClaimStatus = ClaimStatus.SUPPORTED
    evidence: list[EvidenceLink] = Field(default_factory=list)
    needs_review: bool = False

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        valid = {k.value for k in ClaimStatus}
        if v not in valid:
            raise ValueError(f"非法 status: {v!r}，合法值: {sorted(valid)}")
        return v


# ── 校验函数 ─────────────────────────────────────────────────────────────────


def validate_claim_evidence(claims: list[ClinicalClaim]) -> list[str]:
    """校验 claim-evidence 关联，返回错误/警告列表。

    检查：
    1. unsupported treatment/risk claim → needs_review
    2. 证据冲突（同时有 supports 和 contradicts）但标记为 supported → 错误
    """
    errors: list[str] = []

    for claim in claims:
        # unsupported treatment → review
        if claim.claim_type in ("treatment", "risk") and claim.status == ClaimStatus.UNSUPPORTED:
            errors.append(f"unsupported {claim.claim_type} claim {claim.claim_id} needs review")
            claim.needs_review = True

        # 证据冲突检查
        link_types = {e.link_type for e in claim.evidence}
        has_support = "supports" in link_types
        has_contradict = "contradicts" in link_types

        if has_support and has_contradict and claim.status == ClaimStatus.SUPPORTED:
            errors.append(
                f"claim {claim.claim_id} has conflicting evidence but marked as supported"
            )

    return errors


# ── 统计指标 ─────────────────────────────────────────────────────────────────


def calculate_claim_metrics(claims: list[ClinicalClaim]) -> dict:
    """计算 claim 统计指标。

    Returns:
        包含 total_claims、supported_count、unsupported_count、contradiction_rate 等。
    """
    total = len(claims)
    if total == 0:
        return {
            "total_claims": 0,
            "supported_count": 0,
            "partially_supported_count": 0,
            "unsupported_count": 0,
            "conflicting_count": 0,
            "contradiction_rate": 0.0,
            "coverage_rate": 0.0,
        }

    supported = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
    partial = sum(1 for c in claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED)
    unsupported = sum(1 for c in claims if c.status == ClaimStatus.UNSUPPORTED)
    conflicting = sum(1 for c in claims if c.status == ClaimStatus.CONFLICTING)

    # 有证据的 claim 比例
    with_evidence = sum(1 for c in claims if c.evidence)

    return {
        "total_claims": total,
        "supported_count": supported,
        "partially_supported_count": partial,
        "unsupported_count": unsupported,
        "conflicting_count": conflicting,
        "contradiction_rate": round(conflicting / total, 4) if total else 0.0,
        "coverage_rate": round(with_evidence / total, 4) if total else 0.0,
    }
