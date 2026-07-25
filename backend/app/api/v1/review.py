# -*- coding: utf-8 -*-
"""人工复核 API — 教师复核评估结果并恢复图执行"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_admin, get_current_user
from app.db.session import get_db
from app.models.user import User
from app.services.review_service import (
    ReviewSaveError,
    finalize_review_state,
    list_pending_evaluations,
    load_evaluation_state,
    save_review_record,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["人工复核"])


# ── 请求/响应模型 ────────────────────────────────────────────────────────────


class ReviewSubmission(BaseModel):
    """教师提交的复核意见

    reviewer_id 由服务端从认证凭据解析，请求体中传入的值将被忽略。
    """
    reviewer_id: Optional[str] = Field(None, description="复核教师 ID（服务端自动填充，忽略传入值）")
    feedback: str = Field(..., max_length=5000, description="复核意见")
    score_adjustments: Optional[dict] = Field(None, description="评分调整")
    override_decision: Optional[bool] = Field(None, description="是否覆盖原决策")


class ReviewRecord(BaseModel):
    """复核记录"""
    id: str
    evaluation_id: str
    reviewer_id: str
    original_scores: Optional[dict] = None
    adjusted_scores: Optional[dict] = None
    feedback: str
    review_reason: Optional[str] = None
    created_at: datetime


# ── 端点 ─────────────────────────────────────────────────────────────────────


@router.post("/{evaluation_id}/submit")
async def submit_review(
    evaluation_id: str,
    review: ReviewSubmission,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    """提交复核意见（仅管理员）

    1. 从 Redis checkpoint 加载暂停的评估状态
    2. 注入复核意见
    3. 恢复图执行（或更新评估结果）
    4. 保存复核记录
    """
    # 0. 复核人身份以认证凭据为准，防止伪造
    review.reviewer_id = str(current_user.id)

    # 1. 加载评估状态
    state = await load_evaluation_state(db, evaluation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Evaluation not found or not pending review")

    if state.get("evaluation_status") != "pending_review":
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation is not pending review (status={state.get('evaluation_status')})",
        )

    # 2. 注入复核意见
    state["review_feedback"] = review.feedback
    state["review_completed_by"] = review.reviewer_id
    state["review_completed_at"] = datetime.now().isoformat()
    state["needs_review"] = False
    state["evaluation_status"] = "review_completed"

    # 3. 应用评分调整（如果有）
    if review.score_adjustments:
        for key, value in review.score_adjustments.items():
            if key in state:
                state[f"original_{key}"] = state[key]
                state[key] = value

    # 4. 保存复核记录
    review_id = str(uuid.uuid4())
    try:
        await save_review_record(
            db,
            review_id=review_id,
            evaluation_id=evaluation_id,
            reviewer_id=review.reviewer_id,
            feedback=review.feedback,
            review_reason=state.get("review_reason"),
            score_adjustments=review.score_adjustments,
        )
    except ReviewSaveError:
        # 复核记录写入失败必须显式失败，避免调用方误认为复核已存档
        raise HTTPException(
            status_code=500,
            detail={"error_code": "REVIEW_SAVE_FAILED", "message": "复核记录保存失败，请稍后重试"},
        ) from None

    # 5. 恢复图执行或更新最终结果
    result = await finalize_review_state(evaluation_id, state)

    return {
        "review_id": review_id,
        "status": "review_completed",
        "result": result,
    }


@router.get("/{evaluation_id}/status")
async def get_review_status(
    evaluation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取评估的复核状态（需登录）"""
    state = await load_evaluation_state(db, evaluation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return {
        "evaluation_id": evaluation_id,
        "status": state.get("evaluation_status"),
        "needs_review": state.get("needs_review", False),
        "review_reason": state.get("review_reason"),
        "review_feedback": state.get("review_feedback"),
        "review_completed_by": state.get("review_completed_by"),
    }


@router.get("/pending")
async def list_pending_reviews(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin),
):
    """列出所有待复核的评估（仅管理员）"""
    pending = await list_pending_evaluations(db)
    return {"pending_reviews": pending, "total": len(pending)}
