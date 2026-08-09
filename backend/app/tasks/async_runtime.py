"""WorkerAsyncRuntime — 每个 Celery prefork 子进程持有一个持久事件循环。

设计：
- 在 worker_process_init 信号中调用 .start()，保存 owner PID / thread ID。
- 同步 Celery task 通过 .run(coro) 在同一线程调用 loop.run_until_complete()。
- prefork 子进程一次只执行一个 task，无需额外 daemon thread。
- worker_process_shutdown 中调用 .stop(close_resources) 关闭资源后关闭 loop。

约束：
- 禁止 run_coroutine_threadsafe()
- 检测 PID/线程不匹配立即抛 WorkerRuntimeOwnershipError
- .run() 先 loop.create_task(coro)，异常/timeout 退出时 cancel + gather
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Awaitable, Callable, Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WorkerRuntimeOwnershipError(RuntimeError):
    """PID / 线程不匹配，或 runtime 已停止时抛出的异常。"""


class WorkerAsyncRuntime:
    """每个 Celery prefork 子进程持有一个持久事件循环。"""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._owner_pid: Optional[int] = None
        self._owner_thread: Optional[int] = None

    def start(self) -> None:
        """在 worker_process_init 中调用，创建 new_event_loop。

        幂等：如果 loop 已存在且未关闭，直接返回。
        """
        if self._loop is not None and not self._loop.is_closed():
            return

        if self._loop is not None and self._loop.is_running():
            raise WorkerRuntimeOwnershipError(
                "Cannot start(): an event loop is already running"
            )

        self._loop = asyncio.new_event_loop()
        self._owner_pid = os.getpid()
        self._owner_thread = threading.get_ident()
        logger.info(
            f"WorkerAsyncRuntime started: pid={self._owner_pid}, "
            f"thread={self._owner_thread}, loop_id={id(self._loop)}"
        )

    def _check_ownership(self) -> None:
        """校验当前 PID / 线程与 owner 匹配，且 runtime 未停止。"""
        if self._loop is None or self._loop.is_closed():
            raise WorkerRuntimeOwnershipError(
                "WorkerAsyncRuntime is not running (loop is closed or None)"
            )
        if os.getpid() != self._owner_pid:
            raise WorkerRuntimeOwnershipError(
                f"PID mismatch: owner={self._owner_pid}, current={os.getpid()}"
            )
        if threading.get_ident() != self._owner_thread:
            raise WorkerRuntimeOwnershipError(
                f"Thread mismatch: owner={self._owner_thread}, "
                f"current={threading.get_ident()}"
            )

    def run(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """在同一线程执行 coroutine，返回结果。

        先 loop.create_task(coro)，再用 run_until_complete(asyncio.wait_for(task, timeout))。
        异常/timeout 退出时 cancel task + gather，确保无悬挂 coroutine。
        """
        self._check_ownership()
        assert self._loop is not None

        task = self._loop.create_task(coro)
        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(task, timeout=timeout)
            )
        except (asyncio.TimeoutError, asyncio.CancelledError, BaseException) as exc:
            # Cancel the task if not done
            if not task.done():
                task.cancel()
                # Wait for cancellation to complete
                try:
                    self._loop.run_until_complete(
                        asyncio.gather(task, return_exceptions=True)
                    )
                except Exception:
                    pass
            raise exc

    def stop(self, close_resources: Optional[Callable[[], Awaitable[None]]] = None) -> None:
        """关闭资源后关闭 loop。在 worker_process_shutdown 中调用。"""
        if self._loop is None or self._loop.is_closed():
            self._loop = None
            return

        # 校验 ownership（允许不同 PID 调用 stop，因为可能是 shutdown 阶段）
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._check_ownership()
            except WorkerRuntimeOwnershipError:
                # shutdown 阶段可能 PID 已变化，强制关闭
                logger.warning(
                    "WorkerAsyncRuntime.stop() called from different PID/thread, "
                    "forcing loop close"
                )
                self._loop.close()
                self._loop = None
                return

            # 在同一 loop 中关闭资源
            if close_resources is not None:
                try:
                    self._loop.run_until_complete(close_resources())
                except Exception:
                    logger.exception("Error closing worker resources")

            self._loop.close()

        self._loop = None
        logger.info("WorkerAsyncRuntime stopped")


# ── 模块级单例 ──────────────────────────────────────────────────────────────

_worker_runtime: Optional[WorkerAsyncRuntime] = None


def get_worker_runtime() -> WorkerAsyncRuntime:
    """获取全局 WorkerAsyncRuntime 单例。"""
    global _worker_runtime
    if _worker_runtime is None:
        _worker_runtime = WorkerAsyncRuntime()
    return _worker_runtime


def run_worker_coroutine(
    coro: Coroutine[Any, Any, T], timeout: Optional[float] = None
) -> T:
    """便捷函数：使用全局 runtime 执行 coroutine。"""
    return get_worker_runtime().run(coro, timeout=timeout)
