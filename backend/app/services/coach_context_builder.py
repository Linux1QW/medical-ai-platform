"""Source-bound CoachContextView builder.

Constructs a CoachContextView from real consultation data,
ensuring no hidden fields (expected_diagnosis, system_prompt, gold labels)
leak into the coach agent's context.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_runtime.contracts import (
    ApprovedMemory,
    CoachContextView,
    VisibleMessage,
    VisiblePatientProfile,
)
from app.models.consultation import Consultation, ConsultationMessage
from app.models.patient import VirtualPatient
from app.models.trainee_memory import TraineeMemory, TraineeMemoryConsent

logger = logging.getLogger(__name__)

MAX_PROFILE_MEMORIES = 5


class CoachContextBuilder:
    """Builds a CoachContextView from database records.

    STRICT rules:
    - Only public VirtualPatient fields (age, gender, chief_complaint)
    - Never include expected_diagnosis, system_prompt, or gold labels
    - At most 5 approved, consented, non-expired memory entries
    - No patient names, diagnosis strings, or raw conversation in memory
    """

    async def build(
        self,
        db: AsyncSession,
        consultation: Consultation,
        doctor_id: int,
    ) -> CoachContextView:
        """Build a CoachContextView from real consultation data.

        Args:
            db: Async database session.
            consultation: The consultation ORM object (already loaded).
            doctor_id: The doctor's user ID.

        Returns:
            A CoachContextView with only visible, safe fields.
        """
        # 1. Load patient-visible profile from VirtualPatient
        visible_patient = await self._load_visible_patient(db, consultation.patient_id)

        # 2. Load consultation messages ordered by sequence
        messages = await self._load_messages(db, consultation.id)

        # 3. Load approved, consented, non-expired profile memories (max 5)
        memories = await self._load_approved_memories(db, doctor_id)

        return CoachContextView(
            consultation_id=consultation.id,
            doctor_id=doctor_id,
            visible_patient=visible_patient,
            messages=messages,
            approved_profile_memories=memories,
        )

    async def _load_visible_patient(
        self, db: AsyncSession, patient_id: int
    ) -> VisiblePatientProfile:
        """Load only public fields from VirtualPatient."""
        result = await db.execute(
            select(VirtualPatient).where(VirtualPatient.id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if patient is None:
            raise ValueError(f"VirtualPatient id={patient_id} not found")

        return VisiblePatientProfile(
            age=patient.age,
            gender=patient.gender,
            chief_complaint=patient.chief_complaint,
        )

    async def _load_messages(
        self, db: AsyncSession, consultation_id: int
    ) -> list[VisibleMessage]:
        """Load ConsultationMessages ordered by sequence."""
        result = await db.execute(
            select(ConsultationMessage)
            .where(ConsultationMessage.consultation_id == consultation_id)
            .order_by(ConsultationMessage.sequence)
        )
        rows = result.scalars().all()
        return [
            VisibleMessage(
                sequence=msg.sequence,
                role=cast(Literal["doctor", "patient", "system"], msg.role),
                content=msg.content,
            )
            for msg in rows
        ]

    async def _load_approved_memories(
        self, db: AsyncSession, doctor_id: int
    ) -> list[ApprovedMemory]:
        """Load approved, consented, non-expired profile memories (max 5).

        Only memory IDs and safe summary fields are included.
        No patient names, diagnosis strings, or raw conversation.
        """
        now = datetime.utcnow()

        # Check if doctor has granted consent
        consent_result = await db.execute(
            select(TraineeMemoryConsent).where(
                and_(
                    TraineeMemoryConsent.doctor_id == doctor_id,
                    TraineeMemoryConsent.granted == 1,
                )
            )
        )
        consent = consent_result.scalar_one_or_none()
        if consent is None:
            return []

        # Load approved, non-expired memories
        result = await db.execute(
            select(TraineeMemory)
            .where(
                and_(
                    TraineeMemory.doctor_id == doctor_id,
                    TraineeMemory.status == "approved",
                    (TraineeMemory.expires_at.is_(None))
                    | (TraineeMemory.expires_at > now),
                )
            )
            .order_by(TraineeMemory.updated_at.desc())
            .limit(MAX_PROFILE_MEMORIES)
        )
        rows = result.scalars().all()
        return [
            ApprovedMemory(
                memory_id=UUID(int=mem.id),
                skill_dimension=mem.skill_dimension,
                summary=mem.summary,
            )
            for mem in rows
        ]
