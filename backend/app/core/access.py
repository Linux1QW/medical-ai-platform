"""资源访问权限校验"""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consultation import Consultation
from app.models.evaluation import Evaluation
from app.models.evaluation_run import EvaluationRun
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


# ── 内部查询 ──────────────────────────────────────────────────────────────────


async def _get_evaluation(db: AsyncSession, evaluation_id: int) -> Evaluation | None:
    result = await db.execute(
        select(Evaluation).where(Evaluation.id == evaluation_id)
    )
    return result.scalar_one_or_none()


async def _get_evaluation_run(db: AsyncSession, run_id: str) -> EvaluationRun | None:
    result = await db.execute(
        select(EvaluationRun).where(EvaluationRun.id == run_id)
    )
    return result.scalar_one_or_none()


# ── 公开 helper ──────────────────────────────────────────────────────────────


async def require_evaluation_access(
    db: AsyncSession,
    evaluation_id: int,
    current_user: User,
) -> Evaluation:
    """校验评估报告存在且当前用户有权访问（本人或管理员）。

    不存在或非管理员访问他人对象统一返回 404，避免资源枚举。
    """
    evaluation = await _get_evaluation(db, evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "评估报告不存在"},
        )
    consultation = await get_consultation(db, evaluation.consultation_id)
    if not consultation:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "关联问诊记录不存在"},
        )
    if current_user.role != "admin" and consultation.doctor_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "评估报告不存在"},
        )
    return evaluation


async def require_evaluation_run_access(
    db: AsyncSession,
    run_id: str,
    current_user: User,
) -> EvaluationRun:
    """校验评估运行记录存在且当前用户有权访问（本人或管理员）。

    不存在或非管理员访问他人对象统一返回 404，避免资源枚举。
    """
    run = await _get_evaluation_run(db, run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "评估运行记录不存在"},
        )
    consultation = await get_consultation(db, run.consultation_id)
    if not consultation:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "关联问诊记录不存在"},
        )
    if current_user.role != "admin" and consultation.doctor_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "评估运行记录不存在"},
        )
    return run
