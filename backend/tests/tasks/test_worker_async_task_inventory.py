"""Inventory test — production Celery task files must not use asyncio.run().

TDD: Before implementation, evaluation_task.py / data_cleanup.py / rag_index_task.py
all call asyncio.run(). This test must FAIL until they are migrated to WorkerAsyncRuntime.
"""

import asyncio
import re
from pathlib import Path

import pytest

TASKS_DIR = Path(__file__).resolve().parents[2] / "app" / "tasks"

# Production Celery task files that bridge sync Celery → async coroutines
PROD_TASK_FILES = [
    "evaluation_task.py",
    "data_cleanup.py",
    "rag_index_task.py",
]


class TestNoAsyncioRunInProductionTasks:
    """Scan production Celery task files and assert no asyncio.run() calls."""

    @pytest.mark.parametrize("filename", PROD_TASK_FILES)
    def test_no_asyncio_run(self, filename):
        filepath = TASKS_DIR / filename
        assert filepath.exists(), f"required production task is missing: {filename}"

        content = filepath.read_text(encoding="utf-8")
        # Match asyncio.run( but not inside comments
        matches = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if re.search(r"asyncio\.run\s*\(", line):
                matches.append((lineno, line.strip()))

        assert not matches, (
            f"{filename} must not use asyncio.run() — "
            f"use get_worker_runtime().run(coro) instead.\n"
            f"Found at: {matches}"
        )


class TestTaskInventorySequentialRuns:
    """Fake sequential task execution — all must share the same loop ID."""

    def test_evaluation_cleanup_ragindex_same_loop(self):
        """Simulate evaluation → cleanup → rag-index; all must use same loop."""
        from app.tasks.async_runtime import WorkerAsyncRuntime

        rt = WorkerAsyncRuntime()
        rt.start()

        loop_ids: list[int] = []

        async def fake_eval():
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"status": "completed"}

        async def fake_cleanup():
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"ok": True}

        async def fake_rag_index():
            loop_ids.append(id(asyncio.get_running_loop()))
            return {"status": "completed"}

        try:
            rt.run(fake_eval())
            rt.run(fake_cleanup())
            rt.run(fake_rag_index())

            assert len(loop_ids) == 3
            assert loop_ids[0] == loop_ids[1] == loop_ids[2], (
                "All three task types must share the same event loop, "
                f"got loop_ids {[hex(lid) for lid in loop_ids]}"
            )
        finally:
            rt.stop(close_resources=None)
