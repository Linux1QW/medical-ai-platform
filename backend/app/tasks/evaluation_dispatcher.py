# -*- coding: utf-8 -*-
"""Evaluation Dispatcher — Outbox → Celery broker 派发循环

独立进程/线程运行，每秒 claim outbox 行并通过 Celery publish。
包含 Circuit Breaker 和日志脱敏。
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from app.services.observability.metrics import (
    EVALUATION_DISPATCH_BREAKER_OPEN,
    EVALUATION_DISPATCH_DURATION,
)

logger = logging.getLogger(__name__)


# ── Worker ID generation ──────────────────────────────────────────────────────


def generate_worker_id() -> str:
    """生成 worker_id: hostname:pid:uuid"""
    hostname = socket.gethostname()
    pid = os.getpid()
    invocation_uuid = uuid.uuid4()
    return f"{hostname}:{pid}:{invocation_uuid}"


# ── Log sanitization ─────────────────────────────────────────────────────────


def sanitize_for_log(payload: dict[str, Any]) -> str:
    """脱敏 payload 用于日志输出 — 只保留 key 名，隐藏值"""
    safe_keys = list(payload.keys())
    return f"keys={safe_keys}"


# ── Circuit Breaker ──────────────────────────────────────────────────────────


class CircuitBreaker:
    """简单 Circuit Breaker: N 次连续失败 → 冷却期 → half-open

    - failure_threshold: 连续失败次数阈值
    - cooldown_seconds: 冷却期秒数
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 30):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: datetime | None = None

    def record_failure(self, now: datetime) -> None:
        """记录一次失败"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = now

    def record_success(self, now: datetime) -> None:
        """记录一次成功，重置计数器"""
        self._consecutive_failures = 0
        self._opened_at = None

    def is_open(self, now: datetime) -> bool:
        """判断 breaker 是否处于 open 状态

        - 未达到阈值 → closed (False)
        - 达到阈值但冷却期已过 → half-open (False)
        - 达到阈值且冷却期未过 → open (True)
        """
        if self._opened_at is None:
            return False
        elapsed = (now - self._opened_at).total_seconds()
        if elapsed >= self._cooldown_seconds:
            # half-open: 允许尝试
            return False
        return True


# ── Publish with retry ────────────────────────────────────────────────────────


async def publish_with_retry(
    task_name: str,
    payload: dict[str, Any],
    celery_publish: Callable[..., Any],
    max_retries: int = 3,
) -> Any | None:
    """尝试 publish 到 Celery broker，最多重试 max_retries 次

    使用 asyncio.to_thread 调用同步 Celery publish。
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = celery_publish(**payload)
            # Handle both sync and async callables
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            last_error = e
            logger.warning(
                f"Celery publish attempt {attempt + 1}/{max_retries} failed: "
                f"{type(e).__name__}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

    logger.error(f"Celery publish exhausted all {max_retries} retries")
    return None


# ── Dispatcher Loop ──────────────────────────────────────────────────────────


class DispatcherLoop:
    """Dispatcher 主循环 — claim → publish → acknowledge

    可注入依赖以便测试。
    """

    def __init__(
        self,
        worker_id: str,
        claim_fn: Callable[..., Awaitable[list]],
        publish_fn: Callable[..., Awaitable[Any]],
        acknowledge_fn: Callable[..., Awaitable[None]],
        reject_fn: Callable[..., Awaitable[str]] | None = None,
        poll_interval: float = 1.0,
        breaker: CircuitBreaker | None = None,
    ):
        self._worker_id = worker_id
        self._claim_fn = claim_fn
        self._publish_fn = publish_fn
        self._acknowledge_fn = acknowledge_fn
        self._reject_fn = reject_fn
        self._poll_interval = poll_interval
        self._breaker = breaker or CircuitBreaker()
        self._running = False

    async def tick(self) -> None:
        """执行一次 claim → publish → acknowledge 循环"""
        now = datetime.utcnow()
        t0 = time.monotonic()

        # Breaker open 时跳过 claim
        if self._breaker.is_open(now):
            EVALUATION_DISPATCH_BREAKER_OPEN.set(1)
            EVALUATION_DISPATCH_DURATION.labels(result="breaker_open").observe(
                time.monotonic() - t0
            )
            logger.debug("Circuit breaker open, skipping claim")
            return
        EVALUATION_DISPATCH_BREAKER_OPEN.set(0)

        leases = await self._claim_fn()

        for lease in leases:
            task_id = str(uuid.uuid4())
            try:
                result = await self._publish_fn(
                    task_name=lease.task_name,
                    payload=lease.payload,
                )
                if result is not None:
                    # publish 成功 → acknowledge
                    actual_task_id = getattr(result, "id", task_id)
                    await self._acknowledge_fn(
                        event_id=lease.event_id,
                        lease_owner=lease.lease_owner,
                        celery_task_id=actual_task_id,
                        published_at=datetime.utcnow(),
                    )
                    self._breaker.record_success(datetime.utcnow())
                    EVALUATION_DISPATCH_DURATION.labels(result="success").observe(
                        time.monotonic() - t0
                    )
                    logger.info(
                        f"Dispatched event={lease.event_id} "
                        f"run={lease.run_id} {sanitize_for_log(lease.payload)}"
                    )
                else:
                    # publish 失败 → reject
                    if self._reject_fn:
                        await self._reject_fn(
                            event_id=lease.event_id,
                            lease_owner=lease.lease_owner,
                            error_code="broker_unavailable",
                            now=datetime.utcnow(),
                        )
                    self._breaker.record_failure(datetime.utcnow())
                    EVALUATION_DISPATCH_DURATION.labels(result="publish_failed").observe(
                        time.monotonic() - t0
                    )
            except Exception as e:
                logger.warning(
                    f"Dispatch tick error for event={lease.event_id}: "
                    f"{type(e).__name__}: {sanitize_for_log(lease.payload)}"
                )
                # acknowledge 失败 → 不记录 breaker failure（可能是 DB 问题）
                if self._reject_fn:
                    try:
                        await self._reject_fn(
                            event_id=lease.event_id,
                            lease_owner=lease.lease_owner,
                            error_code="ack_failed",
                            now=datetime.utcnow(),
                        )
                    except Exception:
                        pass

    async def run(self) -> None:
        """主循环 — 持续 tick 直到停止"""
        self._running = True
        logger.info(f"Dispatcher started: worker_id={self._worker_id}")

        while self._running:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"Dispatcher tick failed: {type(e).__name__}: {e}")

            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """停止 dispatcher（SIGTERM 处理）"""
        logger.info(f"Dispatcher stopping: worker_id={self._worker_id}")
        self._running = False
