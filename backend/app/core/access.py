"""资源访问权限校验"""

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consultation import Consultation
from app.models.user import User
from app.services.consultation_service import get_consultation


async def require_consultation_access(
    db: AsyncSession,
    consultation_id: int,
    current_user: User,
) -> Consultation:
    """校验问诊记录存在且当前用户有权访问（本人或管理员）。"""
    consultation = await get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "问诊记录不存在"},
        )
    if current_user.role != "admin" and consultation.doctor_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "FORBIDDEN", "message": "无权访问该问诊记录"},
        )
    return consultation
