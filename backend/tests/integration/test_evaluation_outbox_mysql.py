# -*- coding: utf-8 -*-
"""MySQL integration test for concurrent outbox claim with FOR UPDATE SKIP LOCKED.

验证：
- 两个 dispatcher 并发 claim 时获得不相交的 event_id 集合
- FOR UPDATE SKIP LOCKED 保证并发安全

运行条件：
- 设置环境变量 MYSQL_INTEGRATION_DATABASE_URL
- 未设置时自动 skip
"""

import asyncio
import os

import pytest

MYSQL_URL = os.environ.get("MYSQL_INTEGRATION_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MYSQL_URL, reason="MYSQL_INTEGRATION_DATABASE_URL not set"),
]


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _create_test_table(engine):
    """Create a fresh test table for outbox rows."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS _test_dispatch_outbox"))
        await conn.execute(text("""
            CREATE TABLE _test_dispatch_outbox (
                event_id VARCHAR(36) PRIMARY KEY,
                run_id VARCHAR(36) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                task_name VARCHAR(100) NOT NULL,
                payload JSON NOT NULL,
                attempt INT NOT NULL DEFAULT 0,
                next_attempt_at DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                lease_owner VARCHAR(160) NULL,
                lease_expires_at DATETIME NULL,
                last_error_code VARCHAR(100) NULL,
                last_task_id VARCHAR(36) NULL,
                published_at DATETIME NULL,
                INDEX ix_status_next (status, next_attempt_at),
                INDEX ix_lease_expires (lease_expires_at)
            )
        """))


async def _seed_rows(engine, n: int):
    """Seed n pending outbox rows."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        for i in range(n):
            await conn.execute(
                text(
                    "INSERT INTO _test_dispatch_outbox "
                    "(event_id, run_id, status, task_name, payload, next_attempt_at) "
                    "VALUES (:eid, :rid, 'pending', 'test.task', '{}', NOW())"
                ),
                {"eid": f"event-{i:04d}", "rid": f"run-{i:04d}"},
            )


async def _claim_rows(engine, worker_id: str, batch_size: int) -> list[str]:
    """Execute a FOR UPDATE SKIP LOCKED claim and return claimed event_ids."""
    from sqlalchemy import text

    claimed = []
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT event_id FROM _test_dispatch_outbox "
                "WHERE status = 'pending' "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= NOW()) "
                "ORDER BY next_attempt_at ASC, created_at ASC, event_id ASC "
                "LIMIT :lim "
                "FOR UPDATE SKIP LOCKED"
            ),
            {"lim": batch_size},
        )
        rows = result.fetchall()
        for row in rows:
            event_id = row[0]
            await conn.execute(
                text(
                    "UPDATE _test_dispatch_outbox "
                    "SET status = 'leased', lease_owner = :worker "
                    "WHERE event_id = :eid"
                ),
                {"worker": worker_id, "eid": event_id},
            )
            claimed.append(event_id)
    return claimed


@pytest.mark.asyncio
async def test_concurrent_claim_disjoint_event_ids():
    """Two dispatchers claiming concurrently must get disjoint event_id sets."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    # Own every connection opened by this test. This prevents pooled MySQL
    # sockets surviving the test event loop as unraisable ResourceWarnings.
    engine = create_async_engine(MYSQL_URL, echo=False, poolclass=NullPool)

    try:
        await _create_test_table(engine)
        await _seed_rows(engine, n=20)

        # Two concurrent claims
        results = await asyncio.gather(
            _claim_rows(engine, worker_id="worker-a", batch_size=10),
            _claim_rows(engine, worker_id="worker-b", batch_size=10),
        )

        worker_a_ids = set(results[0])
        worker_b_ids = set(results[1])

        # Core assertion: disjoint sets
        assert worker_a_ids.isdisjoint(worker_b_ids), (
            f"Concurrent claims overlapped: {worker_a_ids & worker_b_ids}"
        )

        # SKIP LOCKED guarantees non-overlap and non-blocking, but InnoDB may
        # return an undersized batch while the leading range is locked. A later
        # dispatcher tick must claim every row once those locks are released.
        assert worker_a_ids or worker_b_ids
        recovery_ids = set(
            await _claim_rows(engine, worker_id="worker-recovery", batch_size=20)
        )
        assert worker_a_ids.isdisjoint(recovery_ids)
        assert worker_b_ids.isdisjoint(recovery_ids)
        assert len(worker_a_ids | worker_b_ids | recovery_ids) == 20
    finally:
        async with engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("DROP TABLE IF EXISTS _test_dispatch_outbox"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_skip_locked_prevents_blocking():
    """A second claim should not block when first holds FOR UPDATE locks."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(MYSQL_URL, echo=False, poolclass=NullPool)

    try:
        await _create_test_table(engine)
        await _seed_rows(engine, n=10)

        # First claim takes 5 rows
        first_batch = await _claim_rows(engine, worker_id="worker-1", batch_size=5)
        assert len(first_batch) == 5

        # Second claim should immediately get 5 more (not blocked)
        second_batch = await _claim_rows(engine, worker_id="worker-2", batch_size=5)
        assert len(second_batch) == 5

        # No overlap
        assert set(first_batch).isdisjoint(set(second_batch))
    finally:
        async with engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("DROP TABLE IF EXISTS _test_dispatch_outbox"))
        await engine.dispose()
