# -*- coding: utf-8 -*-
"""病例推荐 API"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.db.session import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services.case_recommender import recommend_cases
from app.services.difficulty_model import calculate_actual_difficulty

router = APIRouter()


@router.get("/recommend")
async def get_recommendations(
    department: Optional[str] = Query(None, description="科室筛选"),
    target_difficulty: Optional[float] = Query(None, description="目标难度（0-10）"),
    count: int = Query(5, ge=1, le=20, description="推荐数量"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """智能推荐病例 — 基于医生能力水平推荐合适难度的病例

    使用"最近发展区"理论：根据医生历史评估表现，推荐略高于当前能力的病例。
    如果不指定 target_difficulty，系统会根据医生历史平均分自动计算。
    """
    recommendations = await recommend_cases(
        db=db,
        doctor_id=current_user.id,
        target_difficulty=target_difficulty,
        count=count,
    )

    # 自动计算时返回实际使用的目标难度
    estimated_target = target_difficulty
    if target_difficulty is None and recommendations:
        # 从推荐理由中提取目标难度（所有条目共享同一目标）
        for r in recommendations:
            if "目标" in r.reason:
                try:
                    estimated_target = float(r.reason.split("目标")[-1].rstrip("）"))
                except (ValueError, IndexError):
                    pass
                break

    return {
        "recommendations": [
            {
                "case_id": r.case_id,
                "case_name": r.case_name,
                "static_difficulty": r.static_difficulty,
                "actual_difficulty": r.actual_difficulty,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r in recommendations
        ],
        "target_difficulty": estimated_target,
        "doctor_id": current_user.id,
    }


@router.get("/{case_id}/difficulty")
async def get_case_difficulty(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取病例的动态难度评估

    基于历史评估数据计算病例实际难度。需要该病例在数据库中有对应的虚拟患者记录。
    如果数据不足（少于 2 次评估），返回 difficulty=null。
    """
    # 尝试从虚拟患者表查找对应记录
    from sqlalchemy import select
    from app.models.patient import VirtualPatient

    stmt = select(VirtualPatient).where(VirtualPatient.name == case_id)
    result = await db.execute(stmt)
    patient = result.scalar_one_or_none()

    if patient is None:
        # 没有 DB 记录，返回静态信息
        return {
            "case_id": case_id,
            "difficulty": None,
            "confidence": 0,
            "sample_size": 0,
            "metrics": {},
            "message": "该病例暂无评估历史数据",
        }

    diff_result = await calculate_actual_difficulty(db, patient.id)

    return {
        "case_id": case_id,
        "difficulty": diff_result.difficulty,
        "confidence": diff_result.confidence,
        "sample_size": diff_result.sample_size,
        "metrics": diff_result.metrics,
    }
