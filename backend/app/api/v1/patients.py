from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_current_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.patient import PatientCreate, PatientUpdate, PatientOut
from app.services.patient_service import (
    create_patient,
    get_patient_by_id,
    list_patients,
    update_patient,
    delete_patient,
)

router = APIRouter()


@router.get("/", response_model=List[PatientOut])
async def get_patients(
    personality_type: Optional[str] = None,
    difficulty_level: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return await list_patients(db, personality_type, difficulty_level)


@router.get("/{patient_id}", response_model=PatientOut)
async def get_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    patient = await get_patient_by_id(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@router.post("/", response_model=PatientOut)
async def add_patient(
    data: PatientCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    return await create_patient(db, data)


@router.put("/{patient_id}", response_model=PatientOut)
async def edit_patient(
    patient_id: int,
    data: PatientUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    patient = await update_patient(db, patient_id, data)
    if not patient:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@router.delete("/{patient_id}")
async def remove_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    ok = await delete_patient(db, patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="患者不存在")
    return {"detail": "删除成功"}
