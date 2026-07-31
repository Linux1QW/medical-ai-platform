# -*- coding: utf-8 -*-
"""Claim-Evidence Graph 测试 — Task 8

验证：
1. ClinicalClaim 结构完整
2. EvidenceLink 绑定 supports/contradicts/insufficient
3. unsupported treatment claim → needs_review
4. 证据冲突不能标记为 supported
5. claim 指标可统计
"""
import pytest
from pydantic import ValidationError

from evaluation.rag_claims import (
    ClinicalClaim,
    ClaimStatus,
    EvidenceLink,
    calculate_claim_metrics,
    validate_claim_evidence,
)


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _claim(
    claim_id: str = "c1",
    claim_type: str = "diagnosis",
    status: str = "supported",
    evidence: list[dict] | None = None,
    **kwargs,
) -> ClinicalClaim:
    return ClinicalClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        text="测试主张",
        status=status,
        evidence=evidence or [],
        **kwargs,
    )


def _evidence(link_type: str = "supports", citation_id: str = "cit_001") -> EvidenceLink:
    return EvidenceLink(
        citation_id=citation_id,
        link_type=link_type,
        text_snippet="证据片段",
        entailment_score=0.9,
    )


# ── 1. ClinicalClaim 结构 ───────────────────────────────────────────────────


class TestClinicalClaim:
    def test_valid_claim(self):
        c = _claim()
        assert c.claim_type == "diagnosis"
        assert c.status == ClaimStatus.SUPPORTED

    def test_all_statuses(self):
        assert ClaimStatus.SUPPORTED == "supported"
        assert ClaimStatus.PARTIALLY_SUPPORTED == "partially_supported"
        assert ClaimStatus.UNSUPPORTED == "unsupported"
        assert ClaimStatus.CONFLICTING == "conflicting"


# ── 2. EvidenceLink ─────────────────────────────────────────────────────────


class TestEvidenceLink:
    def test_valid_link(self):
        e = _evidence()
        assert e.link_type == "supports"
        assert e.entailment_score == 0.9

    def test_invalid_link_type_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceLink(
                citation_id="cit_001",
                link_type="bogus",
                text_snippet="x",
                entailment_score=0.5,
            )


# ── 3. validate_claim_evidence ──────────────────────────────────────────────


class TestValidateClaimEvidence:
    def test_supported_with_evidence(self):
        """有证据支持的 claim → valid"""
        claims = [_claim(evidence=[_evidence("supports").model_dump()])]
        errors = validate_claim_evidence(claims)
        assert errors == []

    def test_unsupported_treatment_needs_review(self):
        """无证据治疗 claim → needs_review"""
        claims = [_claim(claim_type="treatment", status="unsupported", evidence=[])]
        errors = validate_claim_evidence(claims)
        assert any("review" in e.lower() or "unsupported" in e.lower() for e in errors)

    def test_conflicting_evidence_not_supported(self):
        """证据冲突不能标记为 supported"""
        claims = [_claim(
            status="supported",
            evidence=[
                _evidence("supports").model_dump(),
                _evidence("contradicts", "cit_002").model_dump(),
            ],
        )]
        errors = validate_claim_evidence(claims)
        assert any("conflict" in e.lower() for e in errors)


# ── 4. calculate_claim_metrics ──────────────────────────────────────────────


class TestClaimMetrics:
    def test_basic_metrics(self):
        claims = [
            _claim(claim_id="c1", status="supported"),
            _claim(claim_id="c2", status="unsupported"),
            _claim(claim_id="c3", status="partially_supported"),
        ]
        metrics = calculate_claim_metrics(claims)
        assert metrics["total_claims"] == 3
        assert metrics["supported_count"] == 1
        assert metrics["unsupported_count"] == 1
        assert metrics["contradiction_rate"] >= 0

    def test_empty_claims(self):
        metrics = calculate_claim_metrics([])
        assert metrics["total_claims"] == 0
