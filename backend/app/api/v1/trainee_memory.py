"""Trainee memory API endpoints."""
from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Query

from app.schemas.trainee_memory import (
    ConsentRequest,
    ConsentResponse,
    CreateMemoryRequest,
    MemoryResponse,
    ReviewMemoryRequest,
)
from app.services.memory.profile import MemoryStatus, ProfileMemoryService

router = APIRouter(prefix="/trainee-memory", tags=["trainee-memory"])

_service = ProfileMemoryService()


def get_service() -> ProfileMemoryService:
    return _service


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(request: CreateMemoryRequest) -> MemoryResponse:
    """Create a new candidate memory entry."""
    try:
        entry = _service.create_candidate(
            doctor_id=request.doctor_id,
            skill_dimension=request.skill_dimension,
            summary=request.summary,
            evidence_refs=request.evidence_refs,
        )
        return MemoryResponse(
            memory_id=entry.memory_id,
            doctor_id=entry.doctor_id,
            status=entry.status,
            skill_dimension=entry.skill_dimension,
            summary=entry.summary,
            evidence_refs=entry.evidence_refs,
            created_at=entry.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/memories/{memory_id}/review", response_model=MemoryResponse)
async def review_memory(memory_id: str, request: ReviewMemoryRequest) -> MemoryResponse:
    """Approve or reject a candidate memory."""
    try:
        if request.action == "approve":
            entry = _service.approve(
                memory_id,
                reviewer_id=request.reviewer_id,
                review_comment=request.review_comment,
                ttl_days=request.ttl_days,
            )
        else:
            entry = _service.reject(
                memory_id,
                reviewer_id=request.reviewer_id,
                review_comment=request.review_comment,
            )
        return MemoryResponse(
            memory_id=entry.memory_id,
            doctor_id=entry.doctor_id,
            status=entry.status,
            skill_dimension=entry.skill_dimension,
            summary=entry.summary,
            evidence_refs=entry.evidence_refs,
            reviewer_id=entry.reviewer_id,
            review_comment=entry.review_comment,
            reviewed_at=entry.reviewed_at,
            expires_at=entry.expires_at,
            created_at=entry.created_at,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/doctors/{doctor_id}/memories", response_model=list[MemoryResponse])
async def list_doctor_memories(
    doctor_id: int,
    status: str | None = Query(None),
) -> list[MemoryResponse]:
    """List memories for a doctor."""
    entries = _service.list_memories(doctor_id, status=cast("MemoryStatus | None", status))
    return [
        MemoryResponse(
            memory_id=e.memory_id,
            doctor_id=e.doctor_id,
            status=e.status,
            skill_dimension=e.skill_dimension,
            summary=e.summary,
            evidence_refs=e.evidence_refs,
            reviewer_id=e.reviewer_id,
            review_comment=e.review_comment,
            reviewed_at=e.reviewed_at,
            expires_at=e.expires_at,
            created_at=e.created_at,
        )
        for e in entries
    ]


@router.put("/consent", response_model=ConsentResponse)
async def set_consent(request: ConsentRequest) -> ConsentResponse:
    """Set consent for a doctor's profile memory."""
    _service.set_consent(request.doctor_id, request.consent)
    return ConsentResponse(doctor_id=request.doctor_id, consent=request.consent)


@router.get("/doctors/{doctor_id}/consent", response_model=ConsentResponse)
async def get_consent(doctor_id: int) -> ConsentResponse:
    """Get consent status for a doctor."""
    consent = _service.get_consent(doctor_id)
    return ConsentResponse(doctor_id=doctor_id, consent=consent)
