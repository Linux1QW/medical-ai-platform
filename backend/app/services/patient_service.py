from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import VirtualPatient
from app.schemas.patient import PatientCreate, PatientUpdate


async def create_patient(db: AsyncSession, data: PatientCreate) -> VirtualPatient:
    patient = VirtualPatient(**data.model_dump())
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


async def get_patient_by_id(db: AsyncSession, patient_id: int) -> Optional[VirtualPatient]:
    result = await db.execute(select(VirtualPatient).where(VirtualPatient.id == patient_id))
    return result.scalar_one_or_none()


async def list_patients(
    db: AsyncSession,
    personality_type: Optional[str] = None,
    difficulty_level: Optional[int] = None,
) -> List[VirtualPatient]:
    query = select(VirtualPatient)
    if personality_type:
        query = query.where(VirtualPatient.personality_type == personality_type)
    if difficulty_level:
        query = query.where(VirtualPatient.difficulty_level == difficulty_level)
    query = query.order_by(VirtualPatient.id.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def update_patient(
    db: AsyncSession, patient_id: int, data: PatientUpdate
) -> Optional[VirtualPatient]:
    patient = await get_patient_by_id(db, patient_id)
    if not patient:
        return None
    update_data = data.model_dump(exclude_unset=True)
    for k, v in update_data.items():
        setattr(patient, k, v)
    await db.commit()
    await db.refresh(patient)
    return patient


async def delete_patient(db: AsyncSession, patient_id: int) -> bool:
    patient = await get_patient_by_id(db, patient_id)
    if not patient:
        return False
    await db.delete(patient)
    await db.commit()
    return True
