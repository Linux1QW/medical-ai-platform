# -*- coding: utf-8 -*-
"""病例难度评估模型 — 基于历史评估数据动态计算实际难度"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consultation import Consultation
from app.models.evaluation import Evaluation

logger = logging.getLogger(__name__)


@dataclass
class DifficultyResult:
    difficulty: Optional[float]  # 0-10, None 表示数据不足
    confidence: float  # 0-1, 样本越多越可信
    sample_size: int
    metrics: dict = field(default_factory=dict)


async def calculate_actual_difficulty(
    db: AsyncSession,
    patient_id: int,
) -> DifficultyResult:
    """根据历史评估结果计算病例实际难度

    难度计算逻辑：
    - 医生普遍得分低 → 难度高
    - 医生普遍需要复核 → 难度高
    - 诊断准确率越低 → 难度高

    Args:
        db: 异步数据库会话
        patient_id: 虚拟患者 ID（对应 consultations.patient_id）

    Returns:
        DifficultyResult 包含难度评分和置信度
    """
    history = await _get_evaluation_history(db, patient_id)

    if not history or len(history) < 2:
        return DifficultyResult(
            difficulty=None,
            confidence=0,
            sample_size=len(history) if history else 0,
        )

    total = len(history)

    # 1. 知识分越低 → 越难 (权重 0.35)
    avg_knowledge = sum(
        (h.knowledge_score if h.knowledge_score is not None else 50) for h in history
    ) / total
    knowledge_difficulty = (100 - avg_knowledge) / 10  # 0-10

    # 2. 复核率越高 → 越难 (权重 0.25)
    review_rate = sum(1 for h in history if h.human_review_needed) / total
    review_difficulty = review_rate * 10  # 0-10

    # 3. 诊断准确率越低 → 越难 (权重 0.25)
    avg_diagnosis = sum(
        (h.diagnosis_score if h.diagnosis_score is not None else 50) for h in history
    ) / total
    diagnosis_difficulty = (100 - avg_diagnosis) / 10  # 0-10

    # 4. 检索状态非 ok 的比例越高 → 越难 (权重 0.15)
    bad_retrieval_rate = sum(
        1 for h in history
        if h.retrieval_status not in ("ok", "not_run")
    ) / total
    retrieval_difficulty = bad_retrieval_rate * 10  # 0-10

    # 综合难度
    actual_difficulty = (
        knowledge_difficulty * 0.35
        + review_difficulty * 0.25
        + diagnosis_difficulty * 0.25
        + retrieval_difficulty * 0.15
    )

    # 置信度：样本越多越可信，最多 30 个样本达到完全置信
    confidence = min(total / 30, 1.0)

    metrics = {
        "avg_knowledge_score": round(avg_knowledge, 1),
        "avg_diagnosis_score": round(avg_diagnosis, 1),
        "needs_review_rate": round(review_rate, 2),
        "bad_retrieval_rate": round(bad_retrieval_rate, 2),
        "knowledge_difficulty": round(knowledge_difficulty, 1),
        "review_difficulty": round(review_difficulty, 1),
        "diagnosis_difficulty": round(diagnosis_difficulty, 1),
        "retrieval_difficulty": round(retrieval_difficulty, 1),
    }

    return DifficultyResult(
        difficulty=round(actual_difficulty, 1),
        confidence=round(confidence, 2),
        sample_size=total,
        metrics=metrics,
    )


async def _get_evaluation_history(
    db: AsyncSession,
    patient_id: int,
    limit: int = 50,
) -> list[Evaluation]:
    """查询该虚拟患者关联的所有已完成评估记录"""
    try:
        stmt = (
            select(Evaluation)
            .join(Consultation, Evaluation.consultation_id == Consultation.id)
            .where(
                Consultation.patient_id == patient_id,
                Evaluation.evaluation_status == "completed",
            )
            .order_by(Evaluation.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())
    except Exception as e:
        logger.warning(f"Failed to get evaluation history for patient {patient_id}: {e}")
        return []
