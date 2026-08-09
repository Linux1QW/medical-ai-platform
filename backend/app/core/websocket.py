"""WebSocket 连接管理器 — run_id 分组 + Redis Progress Bus 跨进程广播"""

import logging
from typing import Dict, List, Optional

from fastapi import WebSocket

from app.services.progress_bus import EvaluationProgressEvent, ProgressBus

logger = logging.getLogger(__name__)


class ConnectionManager:
    """以 run_id 分组的 WebSocket 连接管理器。

    - register/disconnect 管理本进程连接
    - start() 启动 bus listener，将跨进程事件 fan-out 到本地 socket
    - replay_latest() 在新连接注册后重放最新进度
    - send_progress() 发布到 bus（跨进程广播）
    - _deliver_local() 仅向本进程 socket fan-out
    """

    def __init__(self, bus: Optional[ProgressBus] = None):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._bus = bus
        self._listener_task: Optional[object] = None

    def register(self, websocket: WebSocket, run_id: str):
        """注册已 accept（且已鉴权）的连接"""
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        logger.info(f"WebSocket connected for run {run_id}")

    def disconnect(self, websocket: WebSocket, run_id: str):
        """断开连接（对不存在的连接幂等）"""
        if run_id in self.active_connections:
            conns = self.active_connections[run_id]
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                del self.active_connections[run_id]
        logger.info(f"WebSocket disconnected for run {run_id}")

    async def start(self):
        """启动 bus listener（跨进程事件 → 本地 fan-out）"""
        if self._bus is None:
            return
        import asyncio
        self._listener_task = asyncio.create_task(
            self._bus.listen(self._on_bus_event)
        )

    async def _on_bus_event(self, event: EvaluationProgressEvent):
        """Bus 回调：将事件 fan-out 到本进程匹配的 socket"""
        await self._deliver_local(str(event.run_id), event)

    async def _deliver_local(self, run_id: str, event: EvaluationProgressEvent):
        """向本进程注册的 socket fan-out"""
        if run_id not in self.active_connections:
            return
        data = {
            "type": event.type,
            "run_id": str(event.run_id),
            "consultation_id": event.consultation_id,
            "status": event.status,
            "progress": event.progress,
            "message": event.message,
            "sequence": event.sequence,
            "created_at": event.created_at.isoformat(),
        }
        dead: list[WebSocket] = []
        for connection in self.active_connections.get(run_id, []):
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"WebSocket send failed, marking dead: {e}")
                dead.append(connection)
        for ws in dead:
            self.disconnect(ws, run_id)

    async def replay_latest(self, websocket: WebSocket, run_id: str):
        """重放最新进度事件给新连接的客户端"""
        if self._bus is None:
            return
        latest = await self._bus.get_latest(run_id)
        if latest is None:
            return
        data = {
            "type": latest.type,
            "run_id": str(latest.run_id),
            "consultation_id": latest.consultation_id,
            "status": latest.status,
            "progress": latest.progress,
            "message": latest.message,
            "sequence": latest.sequence,
            "created_at": latest.created_at.isoformat(),
        }
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"WebSocket replay failed: {e}")

    async def send_progress(
        self,
        *,
        run_id: str,
        consultation_id: int,
        status: str,
        progress: int,
        message: str,
    ):
        """发布进度到 bus（跨进程广播；publish 失败只记日志，不让评估失败）"""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                run_id=run_id,
                consultation_id=consultation_id,
                status=status,
                progress=progress,
                message=message,
            )
        except Exception as e:
            logger.warning(f"ProgressBus publish failed (non-fatal): {e}")

    async def close(self):
        """关闭 listener 和 bus 连接"""
        if self._bus:
            await self._bus.close()
        if self._listener_task is not None:
            import asyncio
            if isinstance(self._listener_task, asyncio.Task) and not self._listener_task.done():
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass


manager: Optional[ConnectionManager] = None


def get_manager() -> ConnectionManager:
    """获取全局 ConnectionManager 实例（需先通过 init_manager 初始化）"""
    global manager
    if manager is None:
        # 延迟初始化（无 bus 降级，兼容测试环境）
        manager = ConnectionManager(bus=None)
    return manager


def init_manager(bus: Optional[ProgressBus] = None) -> ConnectionManager:
    """初始化全局 ConnectionManager（在 lifespan 中调用）"""
    global manager
    manager = ConnectionManager(bus=bus)
    return manager
