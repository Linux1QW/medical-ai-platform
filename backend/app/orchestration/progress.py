"""WebSocket 进度映射器 — 通过 ProgressBus 跨进程广播"""

import logging

from app.core.websocket import get_manager
from app.orchestration.state import ProgressEvent

logger = logging.getLogger(__name__)


def _infer_status(progress: int) -> str:
    """根据进度推断状态"""
    if progress <= 0:
        return "running"
    if progress >= 100:
        return "completed"
    return "running"


async def send_progress(run_id: str, consultation_id: int, event: ProgressEvent):
    """将 ProgressEvent 通过 bus 广播到所有 WebSocket"""
    try:
        mgr = get_manager()
        await mgr.send_progress(
            run_id=run_id,
            consultation_id=consultation_id,
            status=_infer_status(event.progress),
            progress=event.progress,
            message=event.message,
        )
    except Exception as e:
        logger.warning(f"进度广播失败 (non-fatal): {e}")


async def send_progress_events(run_id: str, consultation_id: int, events: list[ProgressEvent]):
    """批量广播进度事件"""
    for event in events:
        await send_progress(run_id, consultation_id, event)
