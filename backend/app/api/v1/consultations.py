from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.consultation import (
    ConsultationCreate,
    ConsultationOut,
    ConsultationDetail,
    MessageCreate,
    MessageOut,
    DiagnosisSubmit,
)
from app.services.consultation_service import (
    create_consultation,
    get_consultation,
    list_consultations,
    get_messages,
    send_doctor_message,
    end_consultation,
    submit_diagnosis,
    delete_consultation,
)

router = APIRouter()


@router.post("/", response_model=ConsultationOut)
async def start_consultation(
    data: ConsultationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await create_consultation(db, current_user.id, data.patient_id)


@router.get("/", response_model=List[ConsultationOut])
async def get_my_consultations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await list_consultations(db, current_user.id)


@router.get("/all", response_model=List[ConsultationOut])
async def get_all_consultations(
    username: Optional[str] = None,
    personality: Optional[str] = None,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """管理员：获取全平台问诊记录，支持多维度筛选"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问全部问诊记录")
        
    filters = {
        "username": username,
        "personality": personality,
        "score_min": score_min,
        "score_max": score_max,
        "start_time": start_time,
        "end_time": end_time
    }
    return await list_consultations(db, doctor_id=None, filters=filters)


@router.get("/{consultation_id}", response_model=ConsultationDetail)
async def get_consultation_detail(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    messages = await get_messages(db, consultation_id)
    return ConsultationDetail(
        **ConsultationOut.model_validate(consultation).model_dump(),
        messages=[MessageOut.model_validate(m) for m in messages],
    )


@router.post("/{consultation_id}/messages", response_model=List[MessageOut])
async def send_message(
    consultation_id: int,
    data: MessageCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    if consultation.status != "in_progress":
        raise HTTPException(status_code=400, detail="该问诊已结束")
        
    # 检查轮次限制 (医生+患者各算一条，所以总消息数 / 2 为当前轮次)
    messages = await get_messages(db, consultation_id)
    current_rounds = len([m for m in messages if m.role == 'doctor'])
    if current_rounds >= consultation.max_rounds:
        raise HTTPException(status_code=403, detail="已达到最大问诊轮次，请提交评估或延长轮次")

    try:
        doctor_msg, patient_msg = await send_doctor_message(db, consultation_id, data.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问诊对话失败: {type(e).__name__}: {e}")
    return [
        MessageOut.model_validate(doctor_msg),
        MessageOut.model_validate(patient_msg),
    ]


@router.post("/{consultation_id}/extend", response_model=ConsultationOut)
async def extend_consultation_rounds(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """延长问诊轮次限制"""
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    consultation.max_rounds += 10
    await db.commit()
    await db.refresh(consultation)
    return consultation


@router.post("/{consultation_id}/submit-diagnosis", response_model=ConsultationOut)
async def submit_diagnosis_endpoint(
    consultation_id: int,
    data: DiagnosisSubmit,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """提交诊断结果和治疗方案，同时结束问诊"""
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    if consultation.status != "in_progress":
        raise HTTPException(status_code=400, detail="该问诊已结束")
    return await submit_diagnosis(db, consultation_id, data.diagnosis, data.treatment_plan)


@router.post("/{consultation_id}/end", response_model=ConsultationOut)
async def finish_consultation(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    return await end_consultation(db, consultation_id)


@router.delete("/{consultation_id}")
async def remove_consultation(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ok = await delete_consultation(db, consultation_id, current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="问诊记录不存在或无权删除")
    return {"detail": "删除成功"}
