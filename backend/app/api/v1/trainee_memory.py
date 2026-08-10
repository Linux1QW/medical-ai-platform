"""Trainee memory API endpoints with authentication and authorization."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import record_audit_log
from app.core.deps import get_current_user
from app.core.permissions import require_permission
from app.db.session import get_db
from app.models.user import User
from app.schemas.trainee_memory import (
    ConsentRequest,
    ConsentResponse,
    CreateMemoryRequest,
    MemoryResponse,
    ReviewMemoryRequest,
)
from app.services.memory.profile import ProfileMemoryService

router = APIRouter(prefix="/trainee-memory", tags=["trainee-memory"])


def _memory_to_response(memory) -> MemoryResponse:
    """Convert a TraineeMemory model to response schema."""
    return MemoryResponse(
        memory_id=memory.id,
        doctor_id=memory.doctor_id,
        status=memory.status,
        skill_dimension=memory.skill_dimension,
        summary=memory.summary,
        evidence_refs=memory.evidence_refs or [],
        reviewer_id=memory.reviewer_id,
        review_comment=memory.review_comment,
        reviewed_at=memory.reviewed_at,
        expires_at=memory.expires_at,
        created_at=memory.created_at,
    )


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(
    request: CreateMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:manage-self"),
) -> MemoryResponse:
    """Create a new candidate memory entry.

    Doctor identity comes from token, not request body.
    """
    service = ProfileMemoryService(db)
    try:
        memory = await service.create_candidate(
            doctor_id=current_user.id,
            skill_dimension=request.skill_dimension,
            summary=request.summary,
            evidence_refs=request.evidence_refs,
        )
        await record_audit_log(
            db,
            user_id=current_user.id,
            action="create_trainee_memory",
            resource_id=str(memory.id),
            detail=f"Created candidate memory for doctor {current_user.id}",
        )
        return _memory_to_response(memory)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/memories/{memory_id}/review", response_model=MemoryResponse)
async def review_memory(
    memory_id: int,
    request: ReviewMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:review"),
) -> MemoryResponse:
    """Approve or reject a candidate memory.

    Requires trainee-memory:review permission (admin only).
    Reviewer identity comes from token.
    """
    service = ProfileMemoryService(db)

    # We need to get the memory first to know the doctor_id
    # Use a raw query since we're admin reviewing any doctor's memory
    from app.models.trainee_memory import TraineeMemory
    from sqlalchemy import select
    result = await db.execute(
        select(TraineeMemory).where(TraineeMemory.id == memory_id)
    )
    memory_obj = result.scalar_one_or_none()
    if memory_obj is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    try:
        if request.action == "approve":
            memory = await service.approve(
                memory_id,
                doctor_id=memory_obj.doctor_id,
                reviewer_id=current_user.id,
                review_comment=request.review_comment,
                ttl_days=request.ttl_days,
            )
        else:
            memory = await service.reject(
                memory_id,
                doctor_id=memory_obj.doctor_id,
                reviewer_id=current_user.id,
                review_comment=request.review_comment,
            )
        return _memory_to_response(memory)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/me/memories", response_model=list[MemoryResponse])
async def list_my_memories(
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:manage-self"),
) -> list[MemoryResponse]:
    """List current doctor's own memories.

    Doctor identity from token. Cannot read another doctor's memories.
    """
    service = ProfileMemoryService(db)
    entries = await service.list_memories(current_user.id, status=status)
    return [_memory_to_response(e) for e in entries]


@router.delete("/me/memories/{memory_id}", status_code=204, response_class=Response)
async def delete_my_memory(
    memory_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:manage-self"),
) -> Response:
    """Delete current doctor's own memory.

    Doctor identity from token. Cannot delete another doctor's memory.
    """
    service = ProfileMemoryService(db)
    deleted = await service.delete_memory(memory_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    await record_audit_log(
        db,
        user_id=current_user.id,
        action="delete_trainee_memory",
        resource_id=str(memory_id),
        detail=f"Doctor {current_user.id} deleted memory {memory_id}",
    )
    return Response(status_code=204)


@router.put("/me/consent", response_model=ConsentResponse)
async def set_my_consent(
    request: ConsentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:manage-self"),
) -> ConsentResponse:
    """Set consent for current doctor's profile memory.

    Doctor identity from token. Cannot set another doctor's consent.
    Consent OFF → immediately excludes approved memories from Coach context.
    """
    service = ProfileMemoryService(db)
    await service.set_consent(current_user.id, request.consent)
    await record_audit_log(
        db,
        user_id=current_user.id,
        action="set_trainee_memory_consent",
        detail=f"Doctor {current_user.id} set consent to {request.consent}",
    )
    return ConsentResponse(doctor_id=current_user.id, consent=request.consent)


@router.get("/me/consent", response_model=ConsentResponse)
async def get_my_consent(
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("trainee-memory:manage-self"),
) -> ConsentResponse:
    """Get consent status for current doctor."""
    service = ProfileMemoryService(db)
    consent = await service.get_consent(current_user.id)
    return ConsentResponse(doctor_id=current_user.id, consent=consent)
