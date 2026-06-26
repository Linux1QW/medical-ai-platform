from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limiter import limiter

from app.core.deps import get_current_user, get_current_admin
from app.core.audit import record_audit_log
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
@limiter.limit("30/minute")
async def add_patient(
    request: Request,
    data: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    patient = await create_patient(db, data)
    await record_audit_log(
        db, user_id=current_user.id, action="admin_action",
        request=request, resource_id=str(patient.id),
        detail=f"创建虚拟患者: name={data.name}",
    )
    await db.commit()
    return patient


@router.put("/{patient_id}", response_model=PatientOut)
@limiter.limit("30/minute")
async def edit_patient(
    request: Request,
    patient_id: int,
    data: PatientUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    patient = await update_patient(db, patient_id, data)
    if not patient:
        raise HTTPException(status_code=404, detail="患者不存在")
    await record_audit_log(
        db, user_id=current_user.id, action="admin_action",
        request=request, resource_id=str(patient_id),
        detail=f"更新虚拟患者: patient_id={patient_id}",
    )
    await db.commit()
    return patient


@router.delete("/{patient_id}")
@limiter.limit("30/minute")
async def remove_patient(
    request: Request,
    patient_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    ok = await delete_patient(db, patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="患者不存在")
    await record_audit_log(
        db, user_id=current_user.id, action="admin_action",
        request=request, resource_id=str(patient_id),
        detail=f"删除虚拟患者: patient_id={patient_id}",
    )
    await db.commit()
    return {"detail": "删除成功"}
