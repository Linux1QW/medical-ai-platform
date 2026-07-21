"""WebSocket 进度映射器"""

import logging

from app.core.websocket import manager
from app.orchestration.state import ProgressEvent

logger = logging.getLogger(__name__)


async def send_progress(consultation_id: int, event: ProgressEvent):
    """将 ProgressEvent 发送到 WebSocket"""
    try:
        await manager.send_progress(consultation_id, event.progress, event.message)
    except Exception as e:
        logger.warning(f"WebSocket 进度发送失败: {e}")


async def send_progress_events(consultation_id: int, events: list[ProgressEvent]):
    """批量发送进度事件"""
    for event in events:
        await send_progress(consultation_id, event)
