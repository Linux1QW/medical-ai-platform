from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.services.evaluation_service import get_stats, get_user_stats_breakdown

router = APIRouter()


@router.get("/")
async def get_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """管理员：全平台统计；普通用户：本人统计"""
    doctor_id = None if current_user.role == "admin" else current_user.id
    result = await get_stats(db, doctor_id)
    if current_user.role == "admin":
        result["user_stats"] = await get_user_stats_breakdown(db)
    return result
