"""Tests for ProfileMemoryService (async, DB-backed)."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.memory.profile import ProfileMemoryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db() -> MagicMock:
    """Create a mock AsyncSession."""
    db = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    return db


def _make_svc(db: MagicMock | None = None) -> ProfileMemoryService:
    """Create a ProfileMemoryService with mocked repo."""
    if db is None:
        db = _mock_db()
    svc = ProfileMemoryService(db)
    svc._repo = MagicMock()
    return svc


# ---------------------------------------------------------------------------
# 1. test_create_candidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_candidate() -> None:
    svc = _make_svc()
    mock_memory = MagicMock()
    mock_memory.status = "candidate"
    mock_memory.doctor_id = 1
    mock_memory.skill_dimension = "rapport"
    mock_memory.summary = "Good rapport skills"
    mock_memory.memory_id = "mem_001"
    svc._repo.create_memory = AsyncMock(return_value=mock_memory)

    entry = await svc.create_candidate(
        doctor_id=1, skill_dimension="rapport", summary="Good rapport skills"
    )
    assert entry.status == "candidate"
    assert entry.doctor_id == 1
    assert entry.skill_dimension == "rapport"
    assert entry.summary == "Good rapport skills"
    assert entry.memory_id == "mem_001"


# ---------------------------------------------------------------------------
# 2. test_create_invalid_dimension
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_invalid_dimension() -> None:
    svc = _make_svc()
    with pytest.raises(ValueError, match="Invalid skill dimension"):
        await svc.create_candidate(
            doctor_id=1, skill_dimension="invalid_dim", summary="test"
        )


# ---------------------------------------------------------------------------
# 3. test_approve_candidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_candidate() -> None:
    svc = _make_svc()
    mock_memory = MagicMock()
    mock_memory.status = "approved"
    mock_memory.reviewer_id = 10
    mock_memory.review_comment = "Looks good"
    mock_memory.reviewed_at = datetime.utcnow()
    mock_memory.expires_at = datetime.utcnow() + timedelta(days=90)
    svc._repo.update_status = AsyncMock(return_value=mock_memory)
    svc._repo.has_consent = AsyncMock(return_value=True)

    approved = await svc.approve(1, doctor_id=1, reviewer_id=10, review_comment="Looks good")
    assert approved.status == "approved"
    assert approved.reviewer_id == 10
    assert approved.review_comment == "Looks good"


# ---------------------------------------------------------------------------
# 4. test_approve_non_candidate (not found → KeyError)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_non_candidate() -> None:
    svc = _make_svc()
    svc._repo.update_status = AsyncMock(return_value=None)

    with pytest.raises(KeyError, match="not found"):
        await svc.approve(1, doctor_id=1, reviewer_id=10)


# ---------------------------------------------------------------------------
# 5. test_reject_candidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_candidate() -> None:
    svc = _make_svc()
    mock_memory = MagicMock()
    mock_memory.status = "rejected"
    mock_memory.reviewer_id = 10
    mock_memory.review_comment = "Insufficient evidence"
    svc._repo.update_status = AsyncMock(return_value=mock_memory)

    rejected = await svc.reject(1, doctor_id=1, reviewer_id=10, review_comment="Insufficient evidence")
    assert rejected.status == "rejected"
    assert rejected.reviewer_id == 10
    assert rejected.review_comment == "Insufficient evidence"


# ---------------------------------------------------------------------------
# 6. test_get_approved_no_consent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_approved_no_consent() -> None:
    svc = _make_svc()
    svc._repo.has_consent = AsyncMock(return_value=False)

    result = await svc.get_approved_memories(1)
    assert result == []


# ---------------------------------------------------------------------------
# 7. test_get_approved_with_consent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_approved_with_consent() -> None:
    svc = _make_svc()
    svc._repo.has_consent = AsyncMock(return_value=True)
    mock_memories = [MagicMock(status="approved")]
    svc._repo.list_memories = AsyncMock(return_value=mock_memories)

    result = await svc.get_approved_memories(1)
    assert len(result) == 1
    assert result[0].status == "approved"


# ---------------------------------------------------------------------------
# 8. test_get_approved_max_5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_approved_max_5() -> None:
    svc = _make_svc()
    svc._repo.has_consent = AsyncMock(return_value=True)
    mock_memories = [MagicMock(status="approved") for _ in range(6)]
    svc._repo.list_memories = AsyncMock(return_value=mock_memories)

    result = await svc.get_approved_memories(1, max_count=5)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# 9. test_consent_default_false
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consent_default_false() -> None:
    svc = _make_svc()
    svc._repo.has_consent = AsyncMock(return_value=False)

    assert await svc.get_consent(999) is False


# ---------------------------------------------------------------------------
# 10. test_list_memories_by_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_memories_by_status() -> None:
    svc = _make_svc()
    e1 = MagicMock(memory_id="mem_1", status="candidate")
    e2 = MagicMock(memory_id="mem_2", status="approved")
    svc._repo.list_memories = AsyncMock(side_effect=[
        [e1],   # candidates
        [e2],   # approved
    ])

    candidates = await svc.list_memories(1, status="candidate")
    approved = await svc.list_memories(1, status="approved")
    assert len(candidates) == 1
    assert candidates[0].memory_id == "mem_1"
    assert len(approved) == 1
    assert approved[0].memory_id == "mem_2"
