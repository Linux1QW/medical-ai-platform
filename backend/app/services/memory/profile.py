"""Trainee profile memory service with approval lifecycle and DB persistence."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import record_audit_log
from app.repositories.trainee_memory import TraineeMemoryRepository

VALID_DIMENSIONS = [
    "rapport", "information_gathering", "clinical_reasoning",
    "communication", "safety_awareness", "professionalism",
]

# Patterns that suggest PHI or non-trainee-behavior content
_PHI_PATTERNS = re.compile(
    r"\b("
    r"\d{3}-\d{2}-\d{4}"        # SSN
    r"|\d{16,19}"                # Credit card
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
    r"|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"  # IP address
    r")",
    re.IGNORECASE,
)

# Keywords suggesting non-trainee-behavior content (patient data, not behavior)
_NON_BEHAVIOR_KEYWORDS = re.compile(
    r"\b("
    r"patient\s+name"
    r"|diagnosis\s+code"
    r"|icd-10"
    r"|prescription\s+details"
    r"|lab\s+results?\s+for\s+patient"
    r")\b",
    re.IGNORECASE,
)


def _validate_deidentification(summary: str) -> None:
    """Validate that summary contains no PHI or patient-identifiable info."""
    if _PHI_PATTERNS.search(summary):
        raise ValueError("Summary contains potential PHI or identifiable information")


def _validate_trainee_behavior_only(summary: str) -> None:
    """Validate that summary describes trainee behavior, not patient data."""
    if _NON_BEHAVIOR_KEYWORDS.search(summary):
        raise ValueError("Summary must describe trainee behavior only, not patient data")


class ProfileMemoryService:
    """Manages trainee profile memories with DB persistence.

    Lifecycle: candidate → approved/rejected/expired
    - Only approved memories are visible to the coach
    - Consent defaults to False (opt-in)
    - Max 5 approved memories loaded per context compilation
    - 6 skill dimensions
    """

    def __init__(self, db: AsyncSession) -> None:
        self._repo = TraineeMemoryRepository(db)
        self._db = db

    async def create_candidate(
        self,
        *,
        doctor_id: int,
        skill_dimension: str,
        summary: str,
        evidence_refs: Optional[list[str]] = None,
    ):
        """Create a new candidate memory entry.

        Validates:
        - skill_dimension is one of the 6 valid dimensions
        - summary is deidentified (no PHI)
        - summary describes trainee behavior only
        """
        if skill_dimension not in VALID_DIMENSIONS:
            raise ValueError(
                f"Invalid skill dimension: {skill_dimension}. "
                f"Must be one of {VALID_DIMENSIONS}"
            )

        _validate_deidentification(summary)
        _validate_trainee_behavior_only(summary)

        memory = await self._repo.create_memory(
            doctor_id=doctor_id,
            skill_dimension=skill_dimension,
            summary=summary[:500],
            evidence_refs=evidence_refs or [],
        )
        return memory

    async def approve(
        self,
        memory_id: int,
        doctor_id: int,
        *,
        reviewer_id: int,
        review_comment: Optional[str] = None,
        ttl_days: int = 90,
    ):
        """Approve a candidate memory.

        Records reviewer from token and creates audit event.
        """
        expires_at = datetime.utcnow() + timedelta(days=ttl_days)
        memory = await self._repo.update_status(
            memory_id=memory_id,
            doctor_id=doctor_id,
            status="approved",
            reviewer_id=reviewer_id,
            review_comment=review_comment,
        )
        if memory is None:
            raise KeyError(f"Memory {memory_id} not found for doctor {doctor_id}")

        # Set expires_at directly since repo doesn't handle it
        memory.expires_at = expires_at
        await self._db.flush()

        # Create audit event
        await record_audit_log(
            self._db,
            user_id=reviewer_id,
            action="approve_trainee_memory",
            resource_id=str(memory_id),
            detail=f"Approved memory {memory_id} for doctor {doctor_id}",
        )
        return memory

    async def reject(
        self,
        memory_id: int,
        doctor_id: int,
        *,
        reviewer_id: int,
        review_comment: Optional[str] = None,
    ):
        """Reject a candidate memory."""
        memory = await self._repo.update_status(
            memory_id=memory_id,
            doctor_id=doctor_id,
            status="rejected",
            reviewer_id=reviewer_id,
            review_comment=review_comment,
        )
        if memory is None:
            raise KeyError(f"Memory {memory_id} not found for doctor {doctor_id}")

        await record_audit_log(
            self._db,
            user_id=reviewer_id,
            action="reject_trainee_memory",
            resource_id=str(memory_id),
            detail=f"Rejected memory {memory_id} for doctor {doctor_id}",
        )
        return memory

    async def get_approved_memories(
        self,
        doctor_id: int,
        *,
        max_count: int = 5,
    ) -> list:
        """Get approved memories for a doctor. Max 5 by default.

        Only returns memories if consent is given.
        Consent OFF → immediately excludes approved memories from Coach context.
        """
        if not await self._repo.has_consent(doctor_id):
            return []

        memories = await self._repo.list_memories(
            doctor_id=doctor_id,
            status="approved",
            include_expired=False,
        )
        return memories[:max_count]

    async def set_consent(self, doctor_id: int, consent: bool):
        """Set consent for a doctor. Default is False.

        Consent OFF → immediately excludes approved memories from Coach context.
        """
        if consent:
            await self._repo.grant_consent(doctor_id)
        else:
            await self._repo.revoke_consent(doctor_id)

    async def get_consent(self, doctor_id: int) -> bool:
        """Get consent status for a doctor."""
        return await self._repo.has_consent(doctor_id)

    async def list_memories(
        self,
        doctor_id: int,
        status: Optional[str] = None,
    ) -> list:
        """List memories for a doctor, optionally filtered by status."""
        return await self._repo.list_memories(
            doctor_id=doctor_id,
            status=status,
            include_expired=True,
        )

    async def delete_memory(self, memory_id: int, doctor_id: int) -> bool:
        """Delete a memory scoped to doctor."""
        return await self._repo.delete_memory(memory_id, doctor_id)

    async def get_memory(self, memory_id: int, doctor_id: int):
        """Get a single memory scoped to doctor."""
        return await self._repo.get_memory(memory_id, doctor_id)
