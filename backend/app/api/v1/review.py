# -*- coding: utf-8 -*-
"""人工复核 API — Task 8: 统一 needs_review 语义，原子复核事务

端点：
- POST /reviews/{evaluation_id}/submit  — 提交复核（仅 admin）
- GET  /reviews/{evaluation_id}/status  — 获取复核状态（owner/admin）
- GET  /reviews/pending                 — 待复核列表（仅 admin）
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_admin, get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.review import (
    PendingReviewItemOut,
    ReviewSubmission,
    ReviewSubmitOut,
)
from app.services.review_service import (
    ReviewConflictError,
    ReviewSaveError,
    count_pending_evaluations,
    list_pending_evaluations,
)
from app.services.review_service import (
    submit_review as svc_submit_review,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["人工复核"])


# ── 端点 ─────────────────────────────────────────────────────────────────────


@router.post("/{evaluation_id}/submit", response_model=ReviewSubmitOut)
async def submit_review_endpoint(
    evaluation_id: int,
    review: ReviewSubmission,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    """提交复核意见（仅管理员）

    原子事务：单 commit 完成 evaluation/run/lock/record 全部更新。
    reviewer_id 从认证 token 解析，请求体不传。
    """
    try:
        result = await svc_submit_review(
            db=db,
            evaluation_id=evaluation_id,
            reviewer_id=str(current_user.id),
            feedback=review.feedback,
            score_adjustments=review.score_adjustments,
        )
    except ReviewConflictError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except ReviewSaveError:
        raise HTTPException(
            status_code=500,
            detail={"error_code": "REVIEW_SAVE_FAILED", "message": "复核记录保存失败，请稍后重试"},
        ) from None

    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return ReviewSubmitOut(**result)


@router.get("/{evaluation_id}/status")
async def get_review_status(
    evaluation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取评估的复核状态（需登录）"""
    from sqlalchemy import select

    from app.models.evaluation import Evaluation

    result = await db.execute(select(Evaluation).where(Evaluation.id == evaluation_id))
    ev = result.scalar_one_or_none()

    if ev is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # 权限检查：资源 owner 或 admin
    if current_user.role != "admin" and ev.consultation_id:
        # 检查是否为资源 owner（consultation 的 doctor）
        from sqlalchemy import select as sa_select

        from app.models.consultation import Consultation

        consult_result = await db.execute(
            sa_select(Consultation.doctor_id).where(Consultation.id == ev.consultation_id)
        )
        doctor_id = consult_result.scalar_one_or_none()
        if doctor_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "AUTH_FORBIDDEN", "message": "权限不足"},
            )

    return {
        "evaluation_id": ev.id,
        "status": ev.evaluation_status,
        "needs_review": ev.human_review_needed,
        "review_reason": ev.review_reason,
        "review_completed_by": ev.review_completed_by,
        "review_completed_at": str(ev.review_completed_at) if ev.review_completed_at else None,
    }


@router.get("/pending")
async def list_pending_reviews(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    """列出所有待复核的评估（仅管理员）

    只查 needs_review 状态，按 created_at 倒序。
    """
    try:
        pending = await list_pending_evaluations(db, limit=limit, offset=offset)
        total = await count_pending_evaluations(db)
    except Exception as e:
        # DB 异常不得吞掉后伪装空列表
        logger.error(f"Failed to list pending evaluations: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "DB_ERROR", "message": "查询待复核列表失败"},
        ) from e

    items = [PendingReviewItemOut(**item) for item in pending]
    return {"pending_reviews": items, "total": total}
