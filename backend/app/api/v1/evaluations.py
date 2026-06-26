from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limiter import limiter

from app.core.deps import get_current_user
from app.core.audit import record_audit_log
from app.db.session import get_db
from app.models.user import User
from app.schemas.evaluation import EvaluationOut, EvaluationRequest
from app.services.evaluation_service import run_evaluation, get_evaluation_by_consultation
from app.core.websocket import manager

router = APIRouter()


@router.websocket("/ws/{consultation_id}")
async def evaluation_progress_ws(websocket: WebSocket, consultation_id: int):
    """评估进度推送 WebSocket"""
    await manager.connect(websocket, consultation_id)
    try:
        while True:
            # 保持连接，等待客户端消息或断开
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, consultation_id)


@router.post("/", response_model=EvaluationOut)
@limiter.limit("5/hour")
async def create_evaluation(
    request: Request,
    data: EvaluationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = await get_evaluation_by_consultation(db, data.consultation_id)
    if existing:
        raise HTTPException(status_code=400, detail="该问诊已有评估记录")
    result = await run_evaluation(db, data.consultation_id)

    await record_audit_log(
        db, user_id=current_user.id, action="trigger_evaluation",
        request=request, resource_id=str(data.consultation_id),
        detail=f"触发评估: consultation_id={data.consultation_id}",
    )
    await db.commit()
    return result


@router.get("/{consultation_id}", response_model=EvaluationOut)
async def get_evaluation(
    consultation_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    evaluation = await get_evaluation_by_consultation(db, consultation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="评估记录不存在")
    return evaluation
