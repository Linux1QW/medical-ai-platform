"""Tests for WorkerAsyncRuntime — prefork-owned persistent event loop.

TDD: these tests must FAIL before implementation (ModuleNotFoundError).
"""

import asyncio
import os
import threading

import pytest

from app.tasks.async_runtime import (
    WorkerAsyncRuntime,
    WorkerRuntimeOwnershipError,
    get_worker_runtime,
)


@pytest.fixture()
def runtime():
    """Fresh runtime per test; stopped after use."""
    rt = WorkerAsyncRuntime()
    rt.start()
    yield rt
    if rt._loop is not None and not rt._loop.is_closed():
        rt.stop(close_resources=None)


# ── start() ────────────────────────────────────────────────────────────────────


class TestStart:
    def test_start_creates_loop(self):
        rt = WorkerAsyncRuntime()
        rt.start()
        try:
            assert rt._loop is not None
            assert not rt._loop.is_closed()
            assert rt._owner_pid == os.getpid()
            assert rt._owner_thread == threading.get_ident()
        finally:
            rt.stop(close_resources=None)

    def test_start_idempotent(self):
        """Calling start() twice must not replace the loop."""
        rt = WorkerAsyncRuntime()
        rt.start()
        loop_first = rt._loop
        rt.start()  # second call is no-op
        try:
            assert rt._loop is loop_first
        finally:
            rt.stop(close_resources=None)


# ── run() ──────────────────────────────────────────────────────────────────────


class TestRun:
    def test_run_returns_result(self, runtime):
        async def coro():
            return 42

        result = runtime.run(coro())
        assert result == 42

    def test_sequential_runs_same_loop(self, runtime):
        """Two consecutive run() calls must succeed on the same loop."""
        loop_ids = []

        async def record_loop():
            loop_ids.append(id(asyncio.get_running_loop()))
            return "ok"

        runtime.run(record_loop())
        runtime.run(record_loop())

        assert len(loop_ids) == 2
        assert loop_ids[0] == loop_ids[1]
        assert loop_ids[0] == id(runtime._loop)

    def test_exception_propagates(self, runtime):
        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            runtime.run(failing())

    def test_timeout_cancels_task(self, runtime):
        async def slow():
            await asyncio.sleep(10)
            return "should not reach"

        with pytest.raises(asyncio.TimeoutError):
            runtime.run(slow(), timeout=0.05)

        # After timeout, runtime must still be usable
        async def quick():
            return "recovered"

        assert runtime.run(quick()) == "recovered"


# ── stop() ─────────────────────────────────────────────────────────────────────


class TestStop:
    def test_stop_closes_loop(self):
        rt = WorkerAsyncRuntime()
        rt.start()
        loop = rt._loop
        rt.stop(close_resources=None)
        assert loop.is_closed()
        assert rt._loop is None

    def test_stop_calls_close_resources(self):
        rt = WorkerAsyncRuntime()
        rt.start()
        closed = False

        async def closer():
            nonlocal closed
            closed = True

        rt.stop(close_resources=closer)
        assert closed

    def test_run_after_stop_rejected(self):
        rt = WorkerAsyncRuntime()
        rt.start()
        rt.stop(close_resources=None)

        async def coro():
            return 1

        with pytest.raises(WorkerRuntimeOwnershipError):
            rt.run(coro())


# ── Ownership checks ──────────────────────────────────────────────────────────


class TestOwnership:
    def test_different_pid_rejected(self, runtime):
        """Simulate PID mismatch by directly mutating _owner_pid."""
        original_pid = runtime._owner_pid
        runtime._owner_pid = -1  # simulate different process

        async def coro():
            return 1

        try:
            with pytest.raises(WorkerRuntimeOwnershipError):
                runtime.run(coro())
        finally:
            runtime._owner_pid = original_pid

    def test_different_thread_rejected(self, runtime):
        """Simulate thread mismatch."""
        original_tid = runtime._owner_thread
        runtime._owner_thread = -1

        async def coro():
            return 1

        try:
            with pytest.raises(WorkerRuntimeOwnershipError):
                runtime.run(coro())
        finally:
            runtime._owner_thread = original_tid


# ── Post-test cleanup ─────────────────────────────────────────────────────────


class TestCleanup:
    def test_loop_closed_after_stop_no_residual(self):
        rt = WorkerAsyncRuntime()
        rt.start()
        loop = rt._loop
        rt.stop(close_resources=None)
        assert loop.is_closed()
        assert rt._loop is None

    def test_get_worker_runtime_singleton(self):
        """get_worker_runtime() returns the same instance."""
        r1 = get_worker_runtime()
        r2 = get_worker_runtime()
        assert r1 is r2
