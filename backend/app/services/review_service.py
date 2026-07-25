# -*- coding: utf-8 -*-
"""人工复核服务层 — 评估状态加载、复核记录持久化、待复核列表

将原先散落在 review 路由层的 raw SQL 收敛到 service 层，统一使用 ORM。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.evaluation_checkpoint import EvaluationCheckpoint
from app.models.review_record import ReviewRecord

logger = logging.getLogger(__name__)


class ReviewSaveError(Exception):
    """复核记录保存失败（由路由层转换为 HTTP 500）"""


async def load_evaluation_state(
    db: AsyncSession, evaluation_id: str
) -> Optional[Dict[str, Any]]:
    """从 Redis checkpoint 或数据库加载评估状态"""
    # 优先从 Redis checkpoint 加载
    try:
        from app.services.llm_cache import _get_redis

        redis = await _get_redis()
        if redis:
            data = await redis.get(f"eval_checkpoint:{evaluation_id}")
            if data:
                return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from Redis: {e}")

    # Fallback: 从数据库加载
    try:
        result = await db.execute(
            select(EvaluationCheckpoint.state_json)
            .where(EvaluationCheckpoint.evaluation_id == evaluation_id)
            .order_by(EvaluationCheckpoint.updated_at.desc())
            .limit(1)
        )
        state_json = result.scalar_one_or_none()
        if state_json is not None:
            # JSON 列通常已反序列化为 dict；兼容历史字符串存储
            if isinstance(state_json, str):
                return json.loads(state_json)
            return state_json
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from DB: {e}")

    return None


async def save_review_record(
    db: AsyncSession,
    review_id: str,
    evaluation_id: str,
    reviewer_id: str,
    feedback: str,
    review_reason: Optional[str],
    score_adjustments: Optional[dict],
) -> None:
    """保存复核记录；失败时抛出 ReviewSaveError，避免调用方误认为已存档"""
    try:
        db.add(
            ReviewRecord(
                id=review_id,
                evaluation_id=evaluation_id,
                reviewer_id=reviewer_id,
                feedback=feedback,
                review_reason=review_reason,
                score_adjustments=score_adjustments,
                created_at=datetime.now(),
            )
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to save review record: {e}")
        await db.rollback()
        raise ReviewSaveError(str(e)) from e


async def finalize_review_state(evaluation_id: str, state: dict) -> Dict[str, Any]:
    """标记评估完成并将更新后的状态写回 Redis checkpoint

    如果 LangGraph 支持从 checkpoint 恢复，则继续执行；
    否则直接标记为完成并返回当前状态。
    """
    state["evaluation_status"] = "completed"

    try:
        from app.services.llm_cache import _get_redis

        redis = await _get_redis()
        if redis:
            await redis.set(
                f"eval_checkpoint:{evaluation_id}",
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


async def list_pending_evaluations(db: AsyncSession, limit: int = 50) -> List[Dict[str, Any]]:
    """列出所有待复核的评估"""
    try:
        result = await db.execute(
            select(
                Evaluation.id,
                Evaluation.consultation_id,
                Evaluation.review_reason,
                Evaluation.created_at,
            )
            .where(Evaluation.evaluation_status == "pending_review")
            .order_by(Evaluation.created_at.desc())
            .limit(limit)
        )
        return [
            {
                "evaluation_id": str(row.id),
                "consultation_id": row.consultation_id,
                "review_reason": row.review_reason,
                "created_at": str(row.created_at),
            }
            for row in result.all()
        ]
    except Exception as e:
        logger.warning(f"Failed to list pending evaluations: {e}")
        return []
