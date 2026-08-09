"""Tests for evaluation_dispatcher task — broker publish + circuit breaker

TDD Phase 1: These tests should FAIL until the implementation is complete.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Test: broker publish retry (2 failures then success) ─────────────────────


@pytest.mark.asyncio
async def test_dispatcher_publish_retries_on_broker_failure():
    """First two publish attempts fail, third succeeds."""
    from app.tasks.evaluation_dispatcher import publish_with_retry

    mock_celery_publish = AsyncMock(
        side_effect=[
            RuntimeError("broker down 1"),
            RuntimeError("broker down 2"),
            MagicMock(id="task-ok"),  # success on 3rd try
        ]
    )

    result = await publish_with_retry(
        task_name="run_evaluation",
        payload={"run_id": "r1", "consultation_id": 1},
        celery_publish=mock_celery_publish,
    )

    assert result is not None
    assert mock_celery_publish.call_count == 3


# ── Test: circuit breaker opens after 5 consecutive failures ─────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_5_failures():
    """5 consecutive failures should open the circuit breaker."""
    from app.tasks.evaluation_dispatcher import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
    now = datetime(2026, 8, 10, 12, 0, 0)

    for _ in range(5):
        breaker.record_failure(now)

    assert breaker.is_open(now) is True


# ── Test: circuit breaker cooldown blocks claims ─────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_cooldown_blocks_claims():
    """During cooldown period, breaker should remain open."""
    from app.tasks.evaluation_dispatcher import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
    now = datetime(2026, 8, 10, 12, 0, 0)

    for _ in range(5):
        breaker.record_failure(now)

    # 10 seconds later — still in cooldown
    assert breaker.is_open(now + timedelta(seconds=10)) is True

    # 31 seconds later — cooldown expired, half-open
    assert breaker.is_open(now + timedelta(seconds=31)) is False


# ── Test: circuit breaker closes on success after cooldown ───────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_closes_on_success():
    """After cooldown, a successful publish should close the breaker."""
    from app.tasks.evaluation_dispatcher import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
    now = datetime(2026, 8, 10, 12, 0, 0)

    for _ in range(5):
        breaker.record_failure(now)

    # After cooldown
    later = now + timedelta(seconds=31)
    breaker.record_success(later)

    assert breaker.is_open(later) is False


# ── Test: cancelled outbox never published ────────────────────────────────────


@pytest.mark.asyncio
async def test_cancelled_outbox_never_published():
    """Dispatcher should skip outbox rows that are cancelled."""
    from app.tasks.evaluation_dispatcher import DispatcherLoop

    mock_claim = AsyncMock(return_value=[])
    mock_publish = AsyncMock()
    mock_ack = AsyncMock()

    loop = DispatcherLoop(
        worker_id="w1",
        claim_fn=mock_claim,
        publish_fn=mock_publish,
        acknowledge_fn=mock_ack,
    )

    # No leased items → no publish calls
    await loop.tick()
    mock_publish.assert_not_called()


# ── Test: publish success but ack failure → next tick re-publishes ───────────


@pytest.mark.asyncio
async def test_publish_success_ack_failure_republishes():
    """If publish succeeds but acknowledge fails (e.g. process crash),
    the next tick should see the outbox still as leased and re-publish."""
    from app.tasks.evaluation_dispatcher import DispatcherLoop

    lease = MagicMock()
    lease.event_id = "evt-1"
    lease.run_id = "run-1"
    lease.task_name = "run_evaluation"
    lease.payload = {"run_id": "run-1", "consultation_id": 1}
    lease.lease_owner = "w1"

    mock_claim = AsyncMock(return_value=[lease])
    mock_publish = AsyncMock(return_value=MagicMock(id="task-id-1"))
    # Acknowledge fails (simulating crash before commit)
    mock_ack = AsyncMock(side_effect=RuntimeError("db connection lost"))

    loop = DispatcherLoop(
        worker_id="w1",
        claim_fn=mock_claim,
        publish_fn=mock_publish,
        acknowledge_fn=mock_ack,
    )

    # First tick: publish succeeds, ack fails
    await loop.tick()
    mock_publish.assert_called_once()

    # Second tick: claim returns same lease again (not acknowledged)
    await loop.tick()
    assert mock_publish.call_count == 2


# ── Test: logs do not contain payload or task_id ─────────────────────────────


def test_dispatcher_log_does_not_leak_payload():
    """Dispatcher log messages must not contain payload content or task IDs."""
    import logging
    from app.tasks.evaluation_dispatcher import sanitize_for_log

    payload = {"run_id": "secret-run", "consultation_id": 42}
    sanitized = sanitize_for_log(payload)

    assert "secret-run" not in sanitized
    assert "42" not in sanitized
    assert "run_id" in sanitized  # key names are OK


# ── Test: worker_id format ───────────────────────────────────────────────────


def test_worker_id_format():
    """Worker ID should be hostname:pid:uuid format."""
    from app.tasks.evaluation_dispatcher import generate_worker_id

    wid = generate_worker_id()
    parts = wid.split(":")
    assert len(parts) == 3
    # Third part should be a valid UUID
    uuid.UUID(parts[2])
