# -*- coding: utf-8 -*-
"""人工复核 API — 教师复核评估结果并恢复图执行"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["人工复核"])


# ── 请求/响应模型 ────────────────────────────────────────────────────────────


class ReviewSubmission(BaseModel):
    """教师提交的复核意见"""
    reviewer_id: str = Field(..., description="复核教师 ID")
    feedback: str = Field(..., description="复核意见")
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
async def submit_review(evaluation_id: str, review: ReviewSubmission):
    """提交复核意见

    1. 从 Redis checkpoint 加载暂停的评估状态
    2. 注入复核意见
    3. 恢复图执行（或更新评估结果）
    4. 保存复核记录
    """
    # 1. 加载评估状态
    state = await _load_evaluation_state(evaluation_id)
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
    await _save_review_record(review_id, evaluation_id, review, state)

    # 5. 恢复图执行或更新最终结果
    result = await _resume_or_finalize(evaluation_id, state)

    return {
        "review_id": review_id,
        "status": "review_completed",
        "result": result,
    }


@router.get("/{evaluation_id}/status")
async def get_review_status(evaluation_id: str):
    """获取评估的复核状态"""
    state = await _load_evaluation_state(evaluation_id)
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
async def list_pending_reviews():
    """列出所有待复核的评估"""
    pending = await _list_pending_evaluations()
    return {"pending_reviews": pending, "total": len(pending)}


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


async def _load_evaluation_state(evaluation_id: str) -> Optional[dict]:
    """从 Redis checkpoint 或数据库加载评估状态"""
    # 尝试从 Redis checkpoint 加载
    try:
        from app.services.llm_cache import _get_redis

        redis = await _get_redis()
        if redis:
            key = f"eval_checkpoint:{evaluation_id}"
            data = await redis.get(key)
            if data:
                return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from Redis: {e}")

    # Fallback: 从数据库加载
    try:
        from app.db.session import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    "SELECT state_json FROM evaluation_checkpoints WHERE evaluation_id = :eid"
                ),
                {"eid": evaluation_id},
            )
            row = result.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from DB: {e}")

    return None


async def _save_review_record(
    review_id: str, evaluation_id: str, review: ReviewSubmission, state: dict
):
    """保存复核记录到数据库"""
    try:
        from app.db.session import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO review_records (id, evaluation_id, reviewer_id, feedback,
                                              review_reason, score_adjustments, created_at)
                    VALUES (:id, :eval_id, :reviewer, :feedback, :reason, :adjustments, :created_at)
                """
                ),
                {
                    "id": review_id,
                    "eval_id": evaluation_id,
                    "reviewer": review.reviewer_id,
                    "feedback": review.feedback,
                    "reason": state.get("review_reason"),
                    "adjustments": (
                        json.dumps(review.score_adjustments, ensure_ascii=False)
                        if review.score_adjustments
                        else None
                    ),
                    "created_at": datetime.now(),
                },
            )
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to save review record: {e}")


async def _resume_or_finalize(evaluation_id: str, state: dict) -> dict:
    """恢复图执行或生成最终结果

    如果 LangGraph 支持从 checkpoint 恢复，则继续执行；
    否则直接标记为完成并返回当前状态。
    """
    state["evaluation_status"] = "completed"

    # 保存更新后的状态到 Redis
    try:
        from app.services.llm_cache import _get_redis

        redis = await _get_redis()
        if redis:
            key = f"eval_checkpoint:{evaluation_id}"
            await redis.set(
                key,
                json.dumps(state, ensure_ascii=False, default=str),
                ex=86400,
            )
    except Exception as e:
        logger.warning(f"Failed to save updated state: {e}")

    return {
        "evaluation_id": evaluation_id,
        "status": "completed",
        "review_completed": True,
    }


async def _list_pending_evaluations() -> list:
    """列出所有待复核的评估"""
    try:
        from app.db.session import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, consultation_id, review_reason, created_at
                    FROM evaluations
                    WHERE evaluation_status = 'pending_review'
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                )
            )
            rows = result.fetchall()
            return [
                {
                    "evaluation_id": str(row[0]),
                    "consultation_id": row[1],
                    "review_reason": row[2],
                    "created_at": str(row[3]),
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning(f"Failed to list pending evaluations: {e}")
        return []
