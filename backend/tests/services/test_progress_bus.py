# -*- coding: utf-8 -*-
"""ProgressBus 单元测试（TDD — 先写失败测试）

使用 FakeProgressBus（内存实现）验证事件契约与 bus 行为。
"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.services.progress_bus import EvaluationProgressEvent, FakeProgressBus


@pytest.fixture
def run_id() -> str:
    return str(uuid4())


@pytest.fixture
def bus():
    return FakeProgressBus()


class TestEvaluationProgressEvent:
    """事件字段正确性"""

    def test_event_fields_complete(self, run_id):
        event = EvaluationProgressEvent(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=50,
            message="评估中",
            sequence=1,
            created_at=datetime.now(timezone.utc),
        )
        assert event.type == "progress"
        assert str(event.run_id) == run_id
        assert event.consultation_id == 1
        assert event.status == "running"
        assert event.progress == 50
        assert event.message == "评估中"
        assert event.sequence == 1
        assert event.created_at is not None

    def test_event_type_default_is_progress(self, run_id):
        event = EvaluationProgressEvent(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=0,
            message="开始",
            sequence=1,
            created_at=datetime.now(timezone.utc),
        )
        assert event.type == "progress"


class TestFakeProgressBusPublish:
    """publish 行为"""

    @pytest.mark.asyncio
    async def test_publish_returns_event_with_all_fields(self, bus, run_id):
        event = await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=50,
            message="评估中",
        )
        assert event.type == "progress"
        assert str(event.run_id) == run_id
        assert event.consultation_id == 1
        assert event.status == "running"
        assert event.progress == 50
        assert event.message == "评估中"
        assert event.sequence >= 1
        assert event.created_at is not None

    @pytest.mark.asyncio
    async def test_sequence_monotonically_increasing(self, bus, run_id):
        events = []
        for i in range(5):
            event = await bus.publish(
                run_id=run_id,
                consultation_id=1,
                status="running",
                progress=i * 20,
                message=f"step {i}",
            )
            events.append(event)

        sequences = [e.sequence for e in events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)  # all unique
        for i in range(1, len(sequences)):
            assert sequences[i] > sequences[i - 1]

    @pytest.mark.asyncio
    async def test_latest_ttl_set(self, bus, run_id):
        await bus.publish(
            run_id=run_id,
            consultation_id=1,
            status="running",
            progress=50,
            message="评估中",
        )
        latest = await bus.get_latest(run_id)
        assert latest is not None
        assert latest.progress == 50
        # FakeProgressBus 记录 TTL
        ttl = bus.get_latest_ttl(run_id)
        assert ttl is not None
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_get_latest_returns_none_for_unknown(self, bus):
        latest = await bus.get_latest("nonexistent-run-id")
        assert latest is None

    @pytest.mark.asyncio
    async def test_get_latest_returns_most_recent(self, bus, run_id):
        await bus.publish(
            run_id=run_id, consultation_id=1, status="running",
            progress=30, message="step1",
        )
        await bus.publish(
            run_id=run_id, consultation_id=1, status="running",
            progress=60, message="step2",
        )
        latest = await bus.get_latest(run_id)
        assert latest is not None
        assert latest.progress == 60
        assert latest.message == "step2"


class TestFakeProgressBusListen:
    """listen / subscriber 行为"""

    @pytest.mark.asyncio
    async def test_listener_receives_published_events(self, bus, run_id):
        received = []

        async def on_event(event):
            received.append(event)

        # Start listener in background
        listen_task = asyncio.create_task(bus.listen(on_event))
        await asyncio.sleep(0.05)  # let listener start

        await bus.publish(
            run_id=run_id, consultation_id=1, status="running",
            progress=50, message="test",
        )
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].progress == 50

        await bus.close()
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass


class TestCorruptJsonHandling:
    """损坏 JSON 被忽略"""

    @pytest.mark.asyncio
    async def test_corrupt_json_ignored(self, bus, run_id):
        """Simulate corrupt data in store; get_latest should handle gracefully"""
        # Directly inject corrupt data
        bus._latest[run_id] = "not-valid-json"
        latest = await bus.get_latest(run_id)
        # Should return None when data is corrupt
        assert latest is None


class TestConnectionManagerDisconnect:
    """disconnect 对不存在连接幂等"""

    def test_disconnect_nonexistent_connection_is_idempotent(self):
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=None)
        ws = object()
        # Should not raise
        mgr.disconnect(ws, "nonexistent-run-id")

    def test_disconnect_same_connection_twice_is_safe(self):
        from app.core.websocket import ConnectionManager

        mgr = ConnectionManager(bus=None)

        class FakeWS:
            pass

        ws = FakeWS()
        run_id = "test-run"
        mgr.active_connections[run_id] = [ws]
        mgr.disconnect(ws, run_id)
        # Second disconnect should not raise
        mgr.disconnect(ws, run_id)
        assert run_id not in mgr.active_connections
