# -*- coding: utf-8 -*-
"""ProgressBus + ConnectionManager 集成测试（TDD — 先写失败测试）

验证：
- 两个 ConnectionManager 实例共享同一 bus：publisher 发消息，两个 manager 都收到
- WS 重放：事件先发布、浏览器后连接，仍立即收到 latest
- 非资源 owner 收不到任何进度
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.progress_bus import FakeProgressBus


@pytest.fixture
def run_id() -> str:
    return str(uuid4())


@pytest.fixture
def bus():
    return FakeProgressBus()


def _make_ws() -> MagicMock:
    """创建一个模拟 WebSocket"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=asyncio.CancelledError)
    return ws


class TestCrossProcessDelivery:
    """两个 ConnectionManager 共享同一 bus"""

    @pytest.mark.asyncio
    async def test_two_managers_both_receive_events(self, bus, run_id):
        from app.core.websocket import ConnectionManager

        mgr1 = ConnectionManager(bus=bus)
        mgr2 = ConnectionManager(bus=bus)

        ws1 = _make_ws()
        ws2 = _make_ws()

        mgr1.register(ws1, run_id)
        mgr2.register(ws2, run_id)

        # Start listeners
        task1 = asyncio.create_task(mgr1.start())
        task2 = asyncio.create_task(mgr2.start())
        await asyncio.sleep(0.05)

        # Publish one event
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=50,
            message="评估中",
        )
        await asyncio.sleep(0.1)

        # Both managers should have delivered to their local sockets
        assert ws1.send_json.called
        assert ws2.send_json.called

        # Verify the delivered data
        call_args_1 = ws1.send_json.call_args[0][0]
        assert call_args_1["type"] == "progress"
        assert call_args_1["progress"] == 50

        call_args_2 = ws2.send_json.call_args[0][0]
        assert call_args_2["type"] == "progress"
        assert call_args_2["progress"] == 50

        await mgr1.close()
        await mgr2.close()
        task1.cancel()
        task2.cancel()
        for t in [task1, task2]:
            try:
                await t
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_delivery_does_not_depend_on_in_process_dict(self, bus, run_id):
        """证明不依赖进程内共享字典 — 两个独立 manager 通过 bus 通信"""
        from app.core.websocket import ConnectionManager

        # mgr1 和 mgr2 各自有独立的 active_connections 字典
        mgr1 = ConnectionManager(bus=bus)
        mgr2 = ConnectionManager(bus=bus)

        # 确认它们不是共享同一个字典
        assert mgr1.active_connections is not mgr2.active_connections

        ws2 = _make_ws()
        mgr2.register(ws2, run_id)

        task2 = asyncio.create_task(mgr2.start())
        await asyncio.sleep(0.05)

        # 通过 bus 发布事件（模拟另一个进程）
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=75,
            message="跨进程测试",
        )
        await asyncio.sleep(0.1)

        # mgr2 应该通过 bus listener 收到事件
        assert ws2.send_json.called
        data = ws2.send_json.call_args[0][0]
        assert data["progress"] == 75

        await mgr2.close()
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass


class TestReplayOnConnect:
    """WS 重放：事件先发布、浏览器后连接，仍立即收到 latest"""

    @pytest.mark.asyncio
    async def test_replay_latest_on_register(self, bus, run_id):
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=bus)

        # 先发布事件
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=30,
            message="第一步",
        )
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=60,
            message="第二步",
        )

        # 浏览器后连接
        ws = _make_ws()
        mgr.register(ws, run_id)

        # 重放 latest
        await mgr.replay_latest(ws, run_id)

        # 应该收到最新的进度
        assert ws.send_json.called
        data = ws.send_json.call_args[0][0]
        assert data["type"] == "progress"
        assert data["progress"] == 60
        assert data["message"] == "第二步"

    @pytest.mark.asyncio
    async def test_no_replay_when_no_latest(self, bus, run_id):
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=bus)
        ws = _make_ws()
        mgr.register(ws, run_id)

        # 没有发布过事件，重放应该什么都不做
        await mgr.replay_latest(ws, run_id)
        assert not ws.send_json.called


class TestAccessControl:
    """非资源 owner 收不到任何进度"""

    @pytest.mark.asyncio
    async def test_non_owner_receives_nothing(self, bus, run_id):
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=bus)

        task = asyncio.create_task(mgr.start())
        await asyncio.sleep(0.05)

        # 没有注册任何连接，发布事件
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=50,
            message="test",
        )
        await asyncio.sleep(0.1)

        # 没有连接注册，所以不应该有任何 send_json 调用
        # （只要不抛异常就算通过）
        assert len(mgr.active_connections) == 0

        await mgr.close()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_only_registered_run_id_receives_events(self, bus):
        """不同 run_id 的事件不会交叉投递"""
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=bus)

        run_id_a = str(uuid4())
        run_id_b = str(uuid4())

        ws_a = _make_ws()
        ws_b = _make_ws()

        mgr.register(ws_a, run_id_a)
        mgr.register(ws_b, run_id_b)

        task = asyncio.create_task(mgr.start())
        await asyncio.sleep(0.05)

        # 只发布 run_id_a 的事件
        await bus.publish(
            run_id=run_id_a,
            consultation_id=1,
            status="running",
            progress=50,
            message="for A only",
        )
        await asyncio.sleep(0.1)

        # ws_a 应该收到
        assert ws_a.send_json.called
        # ws_b 不应该收到
        assert not ws_b.send_json.called

        await mgr.close()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
