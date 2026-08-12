"""Repository for trainee memory CRUD and consent management."""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trainee_memory import TraineeMemory, TraineeMemoryConsent


class TraineeMemoryRepository:
    """Handles CRUD for trainee memories with doctor_id authorization."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    async def create_memory(
        self,
        doctor_id: int,
        skill_dimension: str,
        summary: str,
        evidence_refs: Optional[list] = None,
        expires_at: Optional[datetime] = None,
    ) -> TraineeMemory:
        """Create a new candidate memory for a doctor."""
        memory = TraineeMemory(
            doctor_id=doctor_id,
            status="candidate",
            skill_dimension=skill_dimension,
            summary=summary,
            evidence_refs=evidence_refs,
            expires_at=expires_at,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def get_memory(self, memory_id: int, doctor_id: int) -> Optional[TraineeMemory]:
        """Get a memory by ID, scoped to doctor for authorization."""
        stmt = select(TraineeMemory).where(
            and_(
                TraineeMemory.id == memory_id,
                TraineeMemory.doctor_id == doctor_id,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_memories(
        self,
        doctor_id: int,
        status: Optional[str] = None,
        include_expired: bool = False,
    ) -> list[TraineeMemory]:
        """List memories for a doctor, optionally filtered by status."""
        conditions = [TraineeMemory.doctor_id == doctor_id]
        if status is not None:
            conditions.append(TraineeMemory.status == status)
        if not include_expired:
            conditions.append(
                and_(
                    TraineeMemory.status != "expired",
                )
            )

        stmt = (
            select(TraineeMemory)
            .where(and_(*conditions))
            .order_by(TraineeMemory.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        memory_id: int,
        doctor_id: int,
        status: str,
        reviewer_id: Optional[int] = None,
        review_comment: Optional[str] = None,
    ) -> Optional[TraineeMemory]:
        """Update memory status (e.g. approve/reject). Scoped to doctor."""
        memory = await self.get_memory(memory_id, doctor_id)
        if memory is None:
            return None
        memory.status = status
        memory.updated_at = datetime.utcnow()
        if reviewer_id is not None:
            memory.reviewer_id = reviewer_id
        if review_comment is not None:
            memory.review_comment = review_comment
        memory.reviewed_at = datetime.utcnow()
        await self.db.flush()
        return memory

    async def delete_memory(self, memory_id: int, doctor_id: int) -> bool:
        """Delete a memory scoped to doctor. Returns True if deleted."""
        memory = await self.get_memory(memory_id, doctor_id)
        if memory is None:
            return False
        await self.db.delete(memory)
        await self.db.flush()
        return True

    # ------------------------------------------------------------------
    # Consent management
    # ------------------------------------------------------------------

    async def get_consent(self, doctor_id: int) -> Optional[TraineeMemoryConsent]:
        """Get consent record for a doctor."""
        stmt = select(TraineeMemoryConsent).where(
            TraineeMemoryConsent.doctor_id == doctor_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def grant_consent(self, doctor_id: int) -> TraineeMemoryConsent:
        """Grant memory consent for a doctor."""
        consent = await self.get_consent(doctor_id)
        if consent is not None:
            consent.granted = True
            consent.granted_at = datetime.utcnow()
            consent.revoked_at = None
            consent.updated_at = datetime.utcnow()
            await self.db.flush()
            return consent

        consent = TraineeMemoryConsent(
            doctor_id=doctor_id,
            granted=True,
            granted_at=datetime.utcnow(),
        )
        self.db.add(consent)
        await self.db.flush()
        return consent

    async def revoke_consent(self, doctor_id: int) -> Optional[TraineeMemoryConsent]:
        """Revoke memory consent for a doctor."""
        consent = await self.get_consent(doctor_id)
        if consent is None:
            return None
        consent.granted = False
        consent.revoked_at = datetime.utcnow()
        consent.updated_at = datetime.utcnow()
        await self.db.flush()
        return consent

    async def has_consent(self, doctor_id: int) -> bool:
        """Check if a doctor has granted memory consent."""
        consent = await self.get_consent(doctor_id)
        return consent is not None and consent.granted is True
