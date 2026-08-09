# -*- coding: utf-8 -*-
"""人工复核服务层 — 原子复核事务、待复核列表、复核记录持久化

Task 8: 统一 needs_review 语义，实现原子人工复核。

核心算法：
1. SELECT Evaluation FOR UPDATE
2. 读取关联 consultation/run 并校验一致性
3. 快照五维原始分数，验证并应用调整值，重新计算 total score
4. 插入 ReviewRecord，只 flush 不 commit
5. 更新 Evaluation / EvaluationRun / EvaluationLock
6. 单次 db.commit()
7. commit 后 best-effort 更新 Redis checkpoint
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.evaluation_checkpoint import EvaluationCheckpoint
from app.models.evaluation_lock import EvaluationLock
from app.models.evaluation_run import EvaluationRun
from app.models.review_record import ReviewRecord
from app.schemas.review import ScoreAdjustments

logger = logging.getLogger(__name__)


# ── 异常类 ────────────────────────────────────────────────────────────────────


class ReviewSaveError(Exception):
    """复核记录保存失败（由路由层转换为 HTTP 500）"""


class ReviewConflictError(Exception):
    """复核状态冲突（由路由层转换为 HTTP 409）"""

    def __init__(self, message: str, status_code: int = 409):
        self.status_code = status_code
        super().__init__(message)


class ReviewNotFoundError(Exception):
    """评估不存在（由路由层转换为 HTTP 404）"""


# ── 统一评分函数引用 ───────────────────────────────────────────────────────────


def _calculate_total(scores: dict) -> Optional[int]:
    """复用项目现有统一评分函数计算总分

    使用与 scoring_agent.calculate_total 相同的逻辑（None 维度不参与，权重重分配）。
    """
    from app.services.agents.scoring_agent import SCORING_WEIGHTS, calculate_total

    # 将 _score 字段名映射为权重 key
    key_map = {
        "inquiry_score": "inquiry",
        "knowledge_score": "knowledge",
        "humanistic_score": "humanistic",
        "diagnosis_score": "diagnosis",
        "treatment_score": "treatment",
    }
    mapped = {}
    for score_key, weight_key in key_map.items():
        val = scores.get(score_key)
        mapped[weight_key] = val

    return calculate_total(mapped)


# ── 原子复核提交 ────────────────────────────────────────────────────────────────


async def submit_review(
    db: AsyncSession,
    evaluation_id: int,
    reviewer_id: str,
    feedback: str,
    score_adjustments: Optional[ScoreAdjustments | dict] = None,
) -> Optional[Dict[str, Any]]:
    """原子复核提交 — 单事务完成所有更新

    Returns:
        成功时返回 {"review_id", "evaluation_id", "run_id", "status", "reviewed_at"}
        evaluation 不存在时返回 None（路由层转 404）

    Raises:
        ReviewConflictError: 状态不是 needs_review（409）
        RuntimeError: 数据库异常（rollback 后抛出）
    """
    try:
        # Step 1: SELECT Evaluation FOR UPDATE
        result = await db.execute(
            select(Evaluation).where(Evaluation.id == evaluation_id).with_for_update()
        )
        ev = result.scalar_one_or_none()

        if ev is None:
            return None

        # 状态校验：只允许 needs_review
        if ev.evaluation_status != "needs_review":
            raise ReviewConflictError(
                f"Evaluation status is '{ev.evaluation_status}', expected 'needs_review'",
                status_code=409,
            )

        # Step 2: 读取关联 run 并校验一致性
        run = None
        if ev.run_id:
            run_result = await db.execute(
                select(EvaluationRun).where(EvaluationRun.id == ev.run_id)
            )
            run = run_result.scalar_one_or_none()

        # 读取 lock
        lock_result = await db.execute(
            select(EvaluationLock).where(EvaluationLock.consultation_id == ev.consultation_id)
        )
        lock = lock_result.scalar_one_or_none()

        # Step 3: 快照五维原始分数
        original_scores = {
            "inquiry_score": ev.inquiry_score,
            "knowledge_score": ev.knowledge_score,
            "humanistic_score": ev.humanistic_score,
            "diagnosis_score": ev.diagnosis_score,
            "treatment_score": ev.treatment_score,
        }

        # 解析并应用评分调整
        adjustments_dict = None
        if score_adjustments is not None:
            if isinstance(score_adjustments, ScoreAdjustments):
                adjustments_dict = score_adjustments.model_dump(exclude_none=True)
            elif isinstance(score_adjustments, dict):
                # 过滤 None 值
                adjustments_dict = {k: v for k, v in score_adjustments.items() if v is not None}
            else:
                adjustments_dict = {}

            # 应用非 None 调整值
            for key, value in adjustments_dict.items():
                if hasattr(ev, key):
                    setattr(ev, key, value)

        # 重新计算 total score
        new_scores = {
            "inquiry_score": ev.inquiry_score,
            "knowledge_score": ev.knowledge_score,
            "humanistic_score": ev.humanistic_score,
            "diagnosis_score": ev.diagnosis_score,
            "treatment_score": ev.treatment_score,
        }
        new_total = _calculate_total(new_scores)
        if new_total is not None:
            ev.total_score = float(new_total)

        # Step 4: 插入 ReviewRecord，只 flush 不 commit
        review_id = str(uuid.uuid4())
        review_record = ReviewRecord(
            id=review_id,
            evaluation_id=str(ev.id),
            reviewer_id=reviewer_id,
            feedback=feedback,
            review_reason=ev.review_reason,
            score_adjustments=adjustments_dict,
            original_scores=original_scores,
            created_at=datetime.utcnow(),
        )
        db.add(review_record)
        await db.flush()

        # Step 5: 更新 Evaluation
        ev.human_review_needed = False
        ev.evaluation_status = "reviewed"
        ev.review_completed_by = reviewer_id
        ev.review_completed_at = datetime.utcnow()

        # Step 6: 更新 EvaluationRun
        if run is not None:
            run.status = "reviewed"
            run.updated_at = datetime.utcnow()

        # 更新 EvaluationLock
        if lock is not None and lock.can_transition_to("reviewed"):
            lock.status = "reviewed"

        # Step 7: 单次 commit
        await db.commit()

        # Step 8: commit 后 best-effort 更新 Redis checkpoint
        await _update_redis_checkpoint(ev)

        return {
            "review_id": uuid.UUID(review_id),
            "evaluation_id": ev.id,
            "run_id": uuid.UUID(ev.run_id) if ev.run_id else None,
            "status": "reviewed",
            "reviewed_at": ev.review_completed_at,
        }

    except ReviewConflictError:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to submit review for evaluation {evaluation_id}: {e}")
        raise


async def _update_redis_checkpoint(ev: Evaluation) -> None:
    """Best-effort 更新 Redis checkpoint；失败仅记录告警"""
    try:
        from app.services.llm_cache import _get_redis

        redis = await _get_redis()
        if redis:
            state = {
                "evaluation_id": ev.id,
                "evaluation_status": ev.evaluation_status,
                "needs_review": False,
                "review_completed": True,
                "review_completed_by": ev.review_completed_by,
                "review_completed_at": str(ev.review_completed_at),
            }
            await redis.set(
                f"eval_checkpoint:{ev.id}",
                json.dumps(state, ensure_ascii=False, default=str),
                ex=86400,
            )
    except Exception as e:
        logger.warning(f"Failed to update Redis checkpoint for evaluation {ev.id}: {e}")


# ── 待复核列表 ────────────────────────────────────────────────────────────────


async def list_pending_evaluations(
    db: AsyncSession, limit: int = 50, offset: int = 0
) -> List[Dict[str, Any]]:
    """列出所有待复核的评估 — 只查 needs_review，按 created_at 倒序

    Raises:
        RuntimeError: DB 异常不吞掉
    """
    # 不再吞掉异常 — DB 错误必须传播
    result = await db.execute(
        select(
            Evaluation.id,
            Evaluation.consultation_id,
            Evaluation.run_id,
            Evaluation.review_reason,
            Evaluation.total_score,
            Evaluation.retrieval_status,
            Evaluation.evidence_stance,
            Evaluation.created_at,
        )
        .where(Evaluation.evaluation_status == "needs_review")
        .order_by(Evaluation.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = result.all()
    return [
        {
            "evaluation_id": row.id,
            "consultation_id": row.consultation_id,
            "run_id": row.run_id,
            "review_reason": row.review_reason,
            "total_score": row.total_score,
            "retrieval_status": row.retrieval_status,
            "evidence_stance": row.evidence_stance,
            "created_at": row.created_at,
        }
        for row in rows
    ]


async def count_pending_evaluations(db: AsyncSession) -> int:
    """统计待复核总数"""
    result = await db.execute(
        select(Evaluation.id).where(Evaluation.evaluation_status == "needs_review")
    )
    return len(result.all())


# ── 兼容旧接口（保留但标记 deprecated）──────────────────────────────────────


async def load_evaluation_state(
    db: AsyncSession, evaluation_id: str
) -> Optional[Dict[str, Any]]:
    """从 Redis checkpoint 或数据库加载评估状态（保留旧接口）"""
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
    """保存复核记录（保留旧接口）"""
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
    """标记评估完成并将更新后的状态写回 Redis checkpoint（保留旧接口）"""
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
