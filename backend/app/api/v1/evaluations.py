from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import require_consultation_access
from app.core.audit import record_audit_log
from app.core.config import settings
from app.core.deps import get_current_user
from app.core.limiter import limiter
from app.core.permissions import require_permission
from app.core.security import decode_access_token
from app.core.websocket import manager
from app.db.session import AsyncSessionLocal, get_db
from app.models.user import User
from app.schemas.evaluation import EvaluationOut, EvaluationRequest
from app.services.evaluation_lock_service import (
    get_lock_status,
    try_acquire_lock,
    update_lock_status,
)
from app.services.evaluation_service import get_evaluation_by_consultation, run_evaluation
from app.services.user_service import get_user_by_id

router = APIRouter()


@router.websocket("/ws/{consultation_id}")
async def evaluation_progress_ws(
    websocket: WebSocket,
    consultation_id: int,
    token: str = Query(...),
):
    """评估进度推送 WebSocket（需 JWT 鉴权）"""
    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=1008, reason="无效的认证凭据")
        return

    user_id_str = payload.get("sub")
    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        await websocket.close(code=1008, reason="无效的认证凭据")
        return

    async with AsyncSessionLocal() as db:
        user = await get_user_by_id(db, user_id)
        if user is None:
            await websocket.close(code=1008, reason="用户不存在")
            return
        try:
            await require_consultation_access(db, consultation_id, user)
        except HTTPException:
            await websocket.close(code=1008, reason="无权访问该问诊记录")
            return

    await manager.connect(websocket, consultation_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, consultation_id)


@router.post("/", response_model=EvaluationOut)
@limiter.limit("5/hour")
async def create_evaluation(
    request: Request,
    data: EvaluationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("evaluation:create"),
):
    await require_consultation_access(db, data.consultation_id, current_user)

    # 1. 快速检查已有评估
    existing = await get_evaluation_by_consultation(db, data.consultation_id)
    if existing and existing.evaluation_status in ("completed", "needs_review", "reviewed"):
        raise HTTPException(status_code=400, detail="该问诊已有评估记录")

    # 2. 获取评估锁（防并发竞态）
    acquired, lock = await try_acquire_lock(db, data.consultation_id)
    if not acquired:
        if lock and lock.status in ("pending", "running"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "EVALUATION_IN_PROGRESS",
                    "message": "评估正在进行中，请勿重复提交",
                    "status": lock.status,
                    "locked_at": lock.locked_at.isoformat() if lock.locked_at else None,
                },
            )

    # 3. 执行评估（根据配置选择同步或 Celery 异步）
    try:
        await update_lock_status(db, data.consultation_id, "running")
        await db.commit()

        if settings.TESTING:
            # 测试模式：同步执行，不经过 Celery
            result = await run_evaluation(db, data.consultation_id)

            final_status = result.evaluation_status
            if final_status == "needs_review":
                await update_lock_status(db, data.consultation_id, "needs_review")
            else:
                await update_lock_status(db, data.consultation_id, "completed")
            await db.commit()

            await record_audit_log(
                db, user_id=current_user.id, action="trigger_evaluation",
                request=request, resource_id=str(data.consultation_id),
                detail=f"触发评估: consultation_id={data.consultation_id}",
            )
            await db.commit()
            return result
        else:
            # 生产模式：通过 Celery 异步提交
            from app.tasks.evaluation_task import run_evaluation_task

            task = run_evaluation_task.delay(
                consultation_id=data.consultation_id,
                run_id=lock.run_id,
            )

            await record_audit_log(
                db, user_id=current_user.id, action="trigger_evaluation",
                request=request, resource_id=str(data.consultation_id),
                detail=f"异步提交评估任务: task_id={task.id}",
            )
            await db.commit()

            return {"task_id": task.id, "status": "submitted"}

    except Exception as e:
        try:
            await update_lock_status(
                db, data.consultation_id, "failed",
                error_message=str(e)[:500],
            )
            await db.commit()
        except Exception:
            await db.rollback()
        raise


@router.get("/{consultation_id}/lock-status")
async def get_evaluation_lock_status(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询评估任务状态（前端轮询用）"""
    await require_consultation_access(db, consultation_id, current_user)
    status = await get_lock_status(db, consultation_id)
    if not status:
        return {"is_active": False, "status": None}
    return status


@router.get("/{consultation_id}", response_model=EvaluationOut)
async def get_evaluation(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await require_consultation_access(db, consultation_id, current_user)
    evaluation = await get_evaluation_by_consultation(db, consultation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="评估记录不存在")
    return evaluation


@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """查询 Celery 异步评估任务状态"""
    from celery.result import AsyncResult

    from app.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response
