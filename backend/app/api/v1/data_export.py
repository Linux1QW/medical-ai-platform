"""用户数据导出 API"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import require_permission
from app.db.session import get_db
from app.models.user import User
from app.models.consultation import Consultation, ConsultationMessage
from app.models.evaluation import Evaluation

router = APIRouter()


@router.get("/me/data-export")
async def export_my_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("consultation:view"),
):
    """导出当前用户的所有数据（用户信息 + 问诊记录 + 评估记录）"""

    # 查询用户所有问诊记录
    consultations_result = await db.execute(
        select(Consultation).where(Consultation.doctor_id == current_user.id)
    )
    consultations = consultations_result.scalars().all()

    export_data = {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "real_name": current_user.real_name,
            "role": current_user.role,
            "department": current_user.department,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        },
        "consultations": [],
    }

    for c in consultations:
        # 查询问诊消息
        messages_result = await db.execute(
            select(ConsultationMessage)
            .where(ConsultationMessage.consultation_id == c.id)
            .order_by(ConsultationMessage.sequence)
        )
        messages = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sequence": m.sequence,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages_result.scalars().all()
        ]

        # 查询关联评估
        eval_result = await db.execute(
            select(Evaluation).where(Evaluation.consultation_id == c.id)
        )
        evaluation = eval_result.scalar_one_or_none()
        eval_data = None
        if evaluation:
            eval_data = {
                "id": evaluation.id,
                "total_score": evaluation.total_score,
                "inquiry_score": evaluation.inquiry_score,
                "knowledge_score": evaluation.knowledge_score,
                "humanistic_score": evaluation.humanistic_score,
                "diagnosis_score": evaluation.diagnosis_score,
                "treatment_score": evaluation.treatment_score,
                "overall_summary": evaluation.overall_summary,
                "improvement_suggestions": evaluation.improvement_suggestions,
                "evaluation_status": evaluation.evaluation_status,
                "created_at": evaluation.created_at.isoformat() if evaluation.created_at else None,
            }

        export_data["consultations"].append({
            "id": c.id,
            "patient_id": c.patient_id,
            "status": c.status,
            "diagnosis": c.diagnosis,
            "treatment_plan": c.treatment_plan,
            "started_at": c.started_at.isoformat() if c.started_at else None,
            "ended_at": c.ended_at.isoformat() if c.ended_at else None,
            "messages": messages,
            "evaluation": eval_data,
        })

    return export_data
