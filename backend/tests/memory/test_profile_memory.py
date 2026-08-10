"""Tests for ProfileMemoryService."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.services.memory.profile import ProfileMemoryService, VALID_DIMENSIONS


@pytest.fixture
def svc() -> ProfileMemoryService:
    return ProfileMemoryService()


def test_create_candidate(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="rapport", summary="Good rapport skills"
    )
    assert entry.status == "candidate"
    assert entry.doctor_id == 1
    assert entry.skill_dimension == "rapport"
    assert entry.summary == "Good rapport skills"
    assert entry.memory_id.startswith("mem_")


def test_create_invalid_dimension(svc: ProfileMemoryService) -> None:
    with pytest.raises(ValueError, match="Invalid skill dimension"):
        svc.create_candidate(
            doctor_id=1, skill_dimension="invalid_dim", summary="test"
        )


def test_approve_candidate(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="communication", summary="Clear communication"
    )
    approved = svc.approve(entry.memory_id, reviewer_id=10, review_comment="Looks good")
    assert approved.status == "approved"
    assert approved.reviewer_id == 10
    assert approved.review_comment == "Looks good"
    assert approved.reviewed_at is not None
    assert approved.expires_at is not None
    assert approved.expires_at > datetime.utcnow()


def test_approve_non_candidate(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="safety_awareness", summary="Safety first"
    )
    svc.approve(entry.memory_id, reviewer_id=10)
    with pytest.raises(ValueError, match="not candidate"):
        svc.approve(entry.memory_id, reviewer_id=10)


def test_reject_candidate(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="clinical_reasoning", summary="Needs work"
    )
    rejected = svc.reject(entry.memory_id, reviewer_id=10, review_comment="Insufficient evidence")
    assert rejected.status == "rejected"
    assert rejected.reviewer_id == 10
    assert rejected.review_comment == "Insufficient evidence"


def test_expire_stale(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="professionalism", summary="Professional"
    )
    approved = svc.approve(entry.memory_id, reviewer_id=10, ttl_days=0)
    # Manually set expires_at to past
    approved.expires_at = datetime.utcnow() - timedelta(days=1)
    count = svc.expire_stale()
    assert count == 1
    assert approved.status == "expired"


def test_get_approved_no_consent(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="rapport", summary="Good"
    )
    svc.approve(entry.memory_id, reviewer_id=10)
    # No consent set → default False
    result = svc.get_approved_memories(1)
    assert result == []


def test_get_approved_with_consent(svc: ProfileMemoryService) -> None:
    entry = svc.create_candidate(
        doctor_id=1, skill_dimension="rapport", summary="Good"
    )
    svc.approve(entry.memory_id, reviewer_id=10)
    svc.set_consent(1, True)
    result = svc.get_approved_memories(1)
    assert len(result) == 1
    assert result[0].status == "approved"


def test_get_approved_max_5(svc: ProfileMemoryService) -> None:
    svc.set_consent(1, True)
    for dim in VALID_DIMENSIONS:
        entry = svc.create_candidate(
            doctor_id=1, skill_dimension=dim, summary=f"Summary for {dim}"
        )
        svc.approve(entry.memory_id, reviewer_id=10)
    result = svc.get_approved_memories(1)
    assert len(result) == 5


def test_consent_default_false(svc: ProfileMemoryService) -> None:
    assert svc.get_consent(999) is False


def test_list_memories_by_status(svc: ProfileMemoryService) -> None:
    e1 = svc.create_candidate(doctor_id=1, skill_dimension="rapport", summary="A")
    e2 = svc.create_candidate(doctor_id=1, skill_dimension="communication", summary="B")
    svc.approve(e1.memory_id, reviewer_id=10)
    # e2 remains candidate
    candidates = svc.list_memories(1, status="candidate")
    approved = svc.list_memories(1, status="approved")
    assert len(candidates) == 1
    assert candidates[0].memory_id == e2.memory_id
    assert len(approved) == 1
    assert approved[0].memory_id == e1.memory_id
