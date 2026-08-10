"""Worker sequential runs — same loop ID across consecutive task executions.

TDD: With the current asyncio.run() implementation, each task gets a NEW loop,
so this test must FAIL until WorkerAsyncRuntime is wired in.
"""

import asyncio


def _make_fake_task_module():
    """Build fake task functions that record the loop ID they ran on."""
    loop_ids: list[int] = []

    async def fake_evaluation_coro(consultation_id: int, run_id: str, resume: bool = False):
        loop_ids.append(id(asyncio.get_running_loop()))
        return {"evaluation_id": 1, "status": "completed", "consultation_id": consultation_id}

    async def fake_cleanup_coro():
        loop_ids.append(id(asyncio.get_running_loop()))
        return {"audit_logs_deleted": 0, "outbox_deleted": 0, "evaluation_runs_deleted": 0}

    return loop_ids, fake_evaluation_coro, fake_cleanup_coro


class TestSequentialRunsShareLoop:
    """Simulate two consecutive Celery task invocations in the same process.

    With the old asyncio.run() approach, each call creates and destroys a loop.
    With WorkerAsyncRuntime, both calls share the same loop.
    """

    def test_two_evaluation_runs_same_loop(self):
        """Two consecutive evaluation task runs must use the same event loop."""
        from app.tasks.async_runtime import WorkerAsyncRuntime

        rt = WorkerAsyncRuntime()
        rt.start()

        loop_ids: list[int] = []

        async def fake_eval(cid: int, rid: str, resume: bool = False):
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"evaluation_id": cid, "status": "completed", "consultation_id": cid}

        try:
            # Simulate two consecutive task runs
            rt.run(fake_eval(1, "run-1"))
            rt.run(fake_eval(2, "run-2"))

            assert len(loop_ids) == 2
            assert loop_ids[0] == loop_ids[1], (
                "Two consecutive runs must share the same event loop, "
                f"got loop_ids {loop_ids}"
            )
        finally:
            rt.stop(close_resources=None)

    def test_evaluation_then_cleanup_same_loop(self):
        """Evaluation and cleanup tasks must share the same loop."""
        from app.tasks.async_runtime import WorkerAsyncRuntime

        rt = WorkerAsyncRuntime()
        rt.start()

        loop_ids: list[int] = []

        async def fake_eval(cid: int, rid: str, resume: bool = False):
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"status": "completed"}

        async def fake_cleanup():
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"ok": True}

        try:
            rt.run(fake_eval(1, "run-1"))
            rt.run(fake_cleanup())

            assert len(loop_ids) == 2
            assert loop_ids[0] == loop_ids[1]
        finally:
            rt.stop(close_resources=None)


class TestCurrentImplementationBreaksLoopIdentity:
    """Verify that the OLD asyncio.run() pattern creates and closes loops.

    This documents the problem we're fixing: each asyncio.run() call
    creates a new loop and closes it, invalidating any process-level
    async resources (DB engines, Redis clients, etc.).
    """

    def test_asyncio_run_closes_loop_after_each_call(self):
        """asyncio.run() closes the loop after each call — resources are lost."""
        loops_closed: list[bool] = []

        async def record():
            loop = asyncio.get_running_loop()
            # After asyncio.run returns, this loop will be closed
            loops_closed.append(loop)

        asyncio.run(record())
        first_loop = loops_closed[0]
        asyncio.run(record())

        # The first loop was closed by asyncio.run() when it returned
        assert first_loop.is_closed(), (
            "asyncio.run() should close the loop after returning"
        )
