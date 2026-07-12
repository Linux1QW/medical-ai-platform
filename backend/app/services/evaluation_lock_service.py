import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation_lock import EvaluationLock

logger = logging.getLogger(__name__)

EVALUATION_TIMEOUT = 300  # 5 分钟


async def try_acquire_lock(
    db: AsyncSession, consultation_id: int
) -> tuple[bool, Optional[EvaluationLock]]:
    """尝试获取评估锁（SELECT FOR UPDATE）"""
    result = await db.execute(
        select(EvaluationLock)
        .where(EvaluationLock.consultation_id == consultation_id)
        .with_for_update()
    )
    existing = result.scalar_one_or_none()
    now = datetime.utcnow()

    if existing:
        if existing.expires_at < now:
            logger.info(f"Expired lock for consultation {consultation_id}")
            await db.execute(
                delete(EvaluationLock).where(
                    EvaluationLock.consultation_id == consultation_id
                )
            )
            await db.flush()
        elif existing.status in ("pending", "running"):
            return False, existing
        else:
            # 终态或 failed → 清理
            await db.execute(
                delete(EvaluationLock).where(
                    EvaluationLock.consultation_id == consultation_id
                )
            )
            await db.flush()

    run_id = str(uuid.uuid4())
    lock = EvaluationLock(
        consultation_id=consultation_id,
        status="pending",
        run_id=run_id,
        locked_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(seconds=EVALUATION_TIMEOUT),
    )
    db.add(lock)
    await db.flush()
    return True, lock


async def update_lock_status(
    db: AsyncSession,
    consultation_id: int,
    new_status: str,
    error_message: str = None,
) -> bool:
    """更新锁状态（带状态机校验）"""
    result = await db.execute(
        select(EvaluationLock).where(
            EvaluationLock.consultation_id == consultation_id
        )
    )
    lock = result.scalar_one_or_none()
    if not lock:
        return False

    if not lock.can_transition_to(new_status):
        logger.warning(f"Invalid transition: {lock.status} → {new_status}")
        return False

    lock.status = new_status
    lock.heartbeat_at = datetime.utcnow()
    if error_message:
        lock.error_message = error_message[:500]
    if new_status in ("completed", "needs_review", "failed"):
        lock.expires_at = datetime.utcnow() + timedelta(hours=24)

    await db.flush()
    return True


async def get_lock_status(
    db: AsyncSession, consultation_id: int
) -> Optional[dict]:
    """查询当前锁状态"""
    result = await db.execute(
        select(EvaluationLock).where(
            EvaluationLock.consultation_id == consultation_id
        )
    )
    lock = result.scalar_one_or_none()
    if not lock:
        return None

    return {
        "consultation_id": lock.consultation_id,
        "status": lock.status,
        "run_id": lock.run_id,
        "locked_at": lock.locked_at.isoformat() if lock.locked_at else None,
        "expires_at": lock.expires_at.isoformat() if lock.expires_at else None,
        "is_active": lock.status in ("pending", "running"),
    }
