import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.access import require_consultation_access, require_evaluation_run_access
from app.core.audit import record_audit_log
from app.core.authentication import AuthenticationError, authenticate_access_token
from app.core.deps import get_current_user
from app.core.permissions import require_permission
from app.core.websocket import get_manager
from app.db.session import AsyncSessionLocal, get_db
from app.models.evaluation import Evaluation
from app.models.evaluation_lock import EvaluationLock
from app.models.evaluation_run import EvaluationRun
from app.models.user import User
from app.schemas.evaluation import (
    EvaluationCancelOut,
    EvaluationOut,
    EvaluationRequest,
    EvaluationRunStatusOut,
    EvaluationSubmitOut,
)
from app.services.evaluation_cancel import publish_cancel_nudge
from app.services.evaluation_dispatch_service import cancel_dispatch, enqueue_dispatch
from app.services.evaluation_lock_service import (
    get_lock_status,
    update_lock_status,
)
from app.services.evaluation_run_service import (
    CancelDisposition,
    create_queued_run,
    get_run,
    request_run_cancel,
)
from app.services.evaluation_service import get_evaluation_by_consultation

logger = logging.getLogger(__name__)

router = APIRouter()


# 连接建立后等待首条鉴权消息的最长时间（秒）
WS_AUTH_TIMEOUT = 5


# ── WebSocket (deprecated consultation-scoped + run-scoped) ──────────────────


@router.websocket("/ws/{consultation_id}")
async def evaluation_progress_ws(
    websocket: WebSocket,
    consultation_id: int,
):
    """评估进度推 WebSocket（需 JWT 鉴权）— 已废弃，请使用 /ws/runs/{run_id}"""
    await websocket.close(code=1008, reason="请使用 /evaluations/ws/runs/{run_id}")


@router.websocket("/ws/runs/{run_id}")
async def evaluation_run_progress_ws(
    websocket: WebSocket,
    run_id: str,
):
    """评估 Run 进度推送 WebSocket（需 JWT 鉴权）

    鉴权方式：连接建立后客户端须在 WS_AUTH_TIMEOUT 秒内发送首条消息
    {"type": "auth", "token": "<JWT>"}，避免 token 暴露在 URL / 访问日志中。
    鉴权成功后服务端回复 {"type": "auth_ok"} 并重放最新进度。
    """
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=WS_AUTH_TIMEOUT)
    except asyncio.TimeoutError:
        await websocket.close(code=1008, reason="鉴权超时")
        return
    except WebSocketDisconnect:
        return

    try:
        auth_msg = json.loads(raw)
        token = auth_msg.get("token") if isinstance(auth_msg, dict) else None
    except json.JSONDecodeError:
        token = None

    if not token:
        await websocket.close(code=1008, reason="无效的认证凭据")
        return

    async with AsyncSessionLocal() as db:
        try:
            user = await authenticate_access_token(db, token)
        except AuthenticationError as e:
            close_code = 1013 if e.status_code == 503 else 1008
            await websocket.close(code=close_code, reason=e.message)
            return

        try:
            await require_evaluation_run_access(db, run_id, user)
        except HTTPException:
            await websocket.close(code=1008, reason="无权访问该评估运行记录")
            return

    mgr = get_manager()
    await websocket.send_json({"type": "auth_ok"})
    mgr.register(websocket, run_id)
    # 重放最新进度
    await mgr.replay_latest(websocket, run_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(websocket, run_id)


# ── Submission Transaction ────────────────────────────────────────────────────


async def _submit_evaluation_transaction(
    db: AsyncSession,
    *,
    consultation_id: int,
    current_user: User,
    request: Request,
) -> EvaluationRun:
    """在同一事务内创建 run、lock、audit、outbox，然后 commit

    全部成功或全部 rollback。
    """
    # 1. 校验已有评估报告
    existing = await get_evaluation_by_consultation(db, consultation_id)
    if existing and existing.evaluation_status in ("completed", "needs_review", "reviewed"):
        raise HTTPException(status_code=409, detail={"error_code": "EVALUATION_EXISTS", "message": "该问诊已有评估记录"})

    # 2. 检查是否有活跃的 run（queued/running/retrying）
    active_run_stmt = (
        select(EvaluationRun)
        .where(
            EvaluationRun.consultation_id == consultation_id,
            EvaluationRun.status.in_(["queued", "running", "retrying"]),
        )
        .limit(1)
    )
    active_result = await db.execute(active_run_stmt)
    active_run = active_result.scalar_one_or_none()
    if active_run:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "EVALUATION_IN_PROGRESS",
                "message": "评估正在进行中，请勿重复提交",
                "run_id": active_run.id,
                "status": active_run.status,
            },
        )

    # 3. 生成 run UUID
    run_id = str(uuid.uuid4())

    # 4. 创建 EvaluationRun(status="queued")
    run = await create_queued_run(db, run_id=run_id, consultation_id=consultation_id)

    # 5. 创建/更新 EvaluationLock
    lock_stmt = (
        select(EvaluationLock)
        .where(EvaluationLock.consultation_id == consultation_id)
        .with_for_update()
    )
    lock_result = await db.execute(lock_stmt)
    existing_lock = lock_result.scalar_one_or_none()

    now = datetime.utcnow()
    if existing_lock:
        existing_lock.status = "pending"
        existing_lock.run_id = run_id
        existing_lock.locked_at = now
        existing_lock.heartbeat_at = now
        existing_lock.expires_at = now + timedelta(seconds=300)
        existing_lock.error_message = None
    else:
        lock = EvaluationLock(
            consultation_id=consultation_id,
            status="pending",
            run_id=run_id,
            locked_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=300),
        )
        db.add(lock)

    # 6. 写审计记录（strict=True）
    await record_audit_log(
        db,
        user_id=current_user.id,
        action="trigger_evaluation",
        request=request,
        resource_id=run_id,
        detail=f"consultation_id={consultation_id}, status=queued",
        strict=True,
    )

    # 7. 创建 TraceContext + enqueue_dispatch（只 flush，不 commit）
    trace_context = {
        "trace_id": str(uuid.uuid4()),
        "run_id": run_id,
        "consultation_id": consultation_id,
    }
    await enqueue_dispatch(
        db,
        run_id=run_id,
        consultation_id=consultation_id,
        trace_context=trace_context,
    )

    # 8. 一次 commit — run/lock/audit/outbox 全部成功或全部 rollback
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"评估提交事务失败: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error_code": "SUBMISSION_FAILED", "message": "评估提交失败，请重试"},
        ) from e

    return run


# ── Cancel Transaction ────────────────────────────────────────────────────────


async def _cancel_evaluation_transaction(
    db: AsyncSession,
    *,
    run_id: str,
    current_user: User,
    request: Request,
) -> tuple[str, EvaluationRun]:
    """在一个事务内调用 request_run_cancel、cancel_dispatch、同步 EvaluationLock

    返回 (disposition_str, run)。
    """
    # 1. SELECT FOR UPDATE 获取 run
    run = await get_run(db, run_id, for_update=True)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NOT_FOUND", "message": "评估运行记录不存在"},
        )

    # 2. 调用 request_run_cancel
    disposition, run = await request_run_cancel(
        db,
        run_id=run_id,
        requested_by=current_user.id,
        now=datetime.utcnow(),
    )

    # 3. 取消 outbox
    await cancel_dispatch(db, run_id=run_id)

    # 4. 同步 EvaluationLock
    if disposition in (CancelDisposition.CANCELLED_BEFORE_START, CancelDisposition.ALREADY_TERMINAL):
        await update_lock_status(db, run.consultation_id, "cancelled")
    elif disposition == CancelDisposition.REQUESTED_RUNNING:
        # running 状态不抢先写 cancelled，只记录取消意图
        pass

    # 5. 写审计记录（strict=True）
    await record_audit_log(
        db,
        user_id=current_user.id,
        action="trigger_evaluation",
        request=request,
        resource_id=run_id,
        detail=f"event=cancel_requested, disposition={disposition.value}",
        strict=True,
    )

    # 6. commit
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"评估取消事务失败: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error_code": "CANCEL_FAILED", "message": "取消操作失败，请重试"},
        ) from e

    return disposition.value, run


# ── Best-effort cancel nudge (post-commit) ────────────────────────────────────


async def _best_effort_cancel_nudge(run_id: str, execution_task_id: str | None = None) -> None:
    """commit 后 best-effort publish cancel nudge + revoke(terminate=False)"""
    # 1. Redis nudge
    try:
        await publish_cancel_nudge(run_id)
    except Exception as e:
        logger.warning(f"cancel nudge 失败 (non-fatal): {e}")

    # 2. Celery revoke
    if execution_task_id:
        try:
            from app.celery_app import celery_app
            celery_app.control.revoke(execution_task_id, terminate=False)
        except Exception as e:
            logger.warning(f"Celery revoke 失败 (non-fatal): {e}")


# ── Progress helper ───────────────────────────────────────────────────────────


async def _get_progress_latest(run_id: str) -> dict | None:
    """从 ProgressBus 获取最新进度（best-effort）"""
    try:
        mgr = get_manager()
        if mgr and mgr._bus:
            event = await mgr._bus.get_latest(run_id)
            if event:
                return {"progress": event.progress, "message": event.message}
    except Exception as e:
        logger.debug(f"获取进度失败 (non-fatal): {e}")
    return None


# ── API Routes ────────────────────────────────────────────────────────────────


@router.post("/", response_model=EvaluationSubmitOut, status_code=202)
async def create_evaluation(
    request: Request,
    data: EvaluationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("evaluation:create"),
):
    """提交评估 — 统一 202 异步契约

    在同一事务内创建 run、lock、audit、outbox，然后返回 202。
    API 不导入或调用 run_evaluation_task.apply_async()。
    """
    await require_consultation_access(db, data.consultation_id, current_user)

    run = await _submit_evaluation_transaction(
        db,
        consultation_id=data.consultation_id,
        current_user=current_user,
        request=request,
    )

    return EvaluationSubmitOut(
        run_id=run.id,
        consultation_id=data.consultation_id,
        status="queued",
        status_url=f"/api/v1/evaluations/runs/{run.id}/status",
        websocket_url=f"/api/v1/evaluations/ws/runs/{run.id}",
    )


@router.get("/runs/{run_id}/status", response_model=EvaluationRunStatusOut)
async def get_run_status(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询评估 run 状态

    status/evaluation_id/cancel_requested 来自 DB，progress 只补充。
    Redis 不可用时仍返回 200 且 progress=None。
    """
    run = await require_evaluation_run_access(db, run_id, current_user)

    # 获取 progress（best-effort）
    progress_data = await _get_progress_latest(run_id)

    # DB 终态优先 — progress 不能把终态改回 running
    db_status = run.status
    progress = None
    message = None
    if progress_data and db_status not in ("completed", "needs_review", "reviewed", "failed", "cancelled"):
        progress = progress_data.get("progress")
        message = progress_data.get("message")

    return EvaluationRunStatusOut(
        run_id=run.id,
        consultation_id=run.consultation_id,
        status=db_status,
        progress=progress,
        message=message,
        evaluation_id=run.evaluation_id,
        error_code=run.error_type,
        attempt=run.attempt,
        cancel_requested=run.cancel_requested_at is not None,
        cancel_requested_at=run.cancel_requested_at,
        submitted_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.post("/runs/{run_id}/cancel", response_model=EvaluationCancelOut)
async def cancel_evaluation(
    run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("evaluation:create"),
):
    """取消评估 run

    - queued/retrying → 200 cancelled
    - running → 202 cancel_requested=true
    - 终态 → 200 幂等
    - needs_review → 409
    """
    run = await require_evaluation_run_access(db, run_id, current_user)

    disposition_str, run = await _cancel_evaluation_transaction(
        db,
        run_id=run_id,
        current_user=current_user,
        request=request,
    )

    # commit 后 best-effort nudge + revoke
    if disposition_str == "requested_running":
        await _best_effort_cancel_nudge(run_id, run.execution_task_id)
        return JSONResponse(
            status_code=202,
            content=EvaluationCancelOut(
                run_id=run.id,
                status="running",
                cancel_requested=True,
                requested_at=run.cancel_requested_at,
            ).model_dump(mode="json"),
        )

    if disposition_str == "not_cancellable":
        raise HTTPException(
            status_code=409,
            detail={"error_code": "NOT_CANCELLABLE", "message": "该评估状态不可取消"},
        )

    if disposition_str == "already_terminal":
        return EvaluationCancelOut(
            run_id=run.id,
            status=run.status,
            cancel_requested=False,
        )

    # cancelled_before_start
    return EvaluationCancelOut(
        run_id=run.id,
        status="cancelled",
        cancel_requested=True,
        requested_at=run.cancel_requested_at,
    )


@router.get("/{consultation_id}/lock-status")
async def get_evaluation_lock_status(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询评估任务状态（前端轮询用）— 兼容映射 pending→queued"""
    await require_consultation_access(db, consultation_id, current_user)
    status = await get_lock_status(db, consultation_id)
    if not status:
        return {"is_active": False, "status": None}
    # 对外映射 pending → queued
    if status.get("status") == "pending":
        status["status"] = "queued"
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
