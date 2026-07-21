import logging
from typing import Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # 存储 consultation_id 对应的 WebSocket 连接
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, consultation_id: int):
        await websocket.accept()
        if consultation_id not in self.active_connections:
            self.active_connections[consultation_id] = []
        self.active_connections[consultation_id].append(websocket)
        logger.info(f"WebSocket connected for consultation {consultation_id}")

    def disconnect(self, websocket: WebSocket, consultation_id: int):
        if consultation_id in self.active_connections:
            self.active_connections[consultation_id].remove(websocket)
            if not self.active_connections[consultation_id]:
                del self.active_connections[consultation_id]
        logger.info(f"WebSocket disconnected for consultation {consultation_id}")

    async def send_progress(self, consultation_id: int, progress: int, message: str):
        if consultation_id in self.active_connections:
            data = {"progress": progress, "message": message}
            for connection in self.active_connections[consultation_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending progress via WebSocket: {e}")

manager = ConnectionManager()
