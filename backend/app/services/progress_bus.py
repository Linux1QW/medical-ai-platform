"""Redis-backed Progress Bus for cross-process evaluation progress broadcasting.

提供 `ProgressBus` Protocol、`RedisProgressBus` 实现和 `FakeProgressBus` 测试替身。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.schemas.evaluation import EvaluationJobStatus

logger = logging.getLogger(__name__)


# ── Event Schema ──────────────────────────────────────────────────────────────


class EvaluationProgressEvent(BaseModel):
    """跨进程进度事件"""
    type: str = "progress"
    run_id: UUID
    consultation_id: int
    status: EvaluationJobStatus
    progress: int = Field(ge=0, le=100)
    message: str = Field(max_length=200)
    sequence: int = Field(ge=1)
    created_at: datetime


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ProgressBus(Protocol):
    async def publish(
        self,
        *,
        run_id: str,
        consultation_id: int,
        status: str,
        progress: int,
        message: str,
    ) -> EvaluationProgressEvent: ...

    async def get_latest(self, run_id: str) -> Optional[EvaluationProgressEvent]: ...

    async def listen(
        self,
        on_event: Callable[[EvaluationProgressEvent], Awaitable[None]],
    ) -> None: ...

    async def close(self) -> None: ...


# ── Redis Keys ────────────────────────────────────────────────────────────────

_SEQ_KEY = "progress:seq:{run_id}"
_LATEST_KEY = "progress:latest:{run_id}"
_CHANNEL = "progress:events"


# ── Lua Script ────────────────────────────────────────────────────────────────
# 原子操作：INCR sequence → EXPIRE → 写入 event JSON → SET latest → PUBLISH
# KEYS[1] = seq key, KEYS[2] = latest key
# ARGV[1] = TTL seconds, ARGV[2] = event JSON, ARGV[3] = channel

_PUBLISH_SCRIPT = """
local seq = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[1])
local payload = ARGV[2]
local event_json = string.gsub(payload, '"sequence":0', '"sequence":' .. seq)
redis.call('SET', KEYS[2], event_json, 'EX', ARGV[1])
redis.call('PUBLISH', ARGV[3], event_json)
return seq
"""


# ── RedisProgressBus ─────────────────────────────────────────────────────────


class RedisProgressBus:
    """Redis Pub/Sub 实现的进度总线。

    - publish 使用 Lua script 原子执行
    - listen 带指数退避重连（1/2/4/8/10 + jitter）
    - close 取消 listener 并关闭连接
    """

    def __init__(self, redis_client, ttl: int = 3600):
        self._redis = redis_client
        self._ttl = ttl
        self._script_sha: Optional[str] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._closed = False

    async def _ensure_script(self) -> str:
        """注册 Lua script 并缓存 SHA"""
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(_PUBLISH_SCRIPT)
        return self._script_sha

    async def publish(
        self,
        *,
        run_id: str,
        consultation_id: int,
        status: str,
        progress: int,
        message: str,
    ) -> EvaluationProgressEvent:
        """发布进度事件（原子操作）"""
        now = datetime.now(timezone.utc)
        # 构建事件模板（sequence=0 由 Lua 替换）
        event_template = {
            "type": "progress",
            "run_id": run_id,
            "consultation_id": consultation_id,
            "status": status,
            "progress": progress,
            "message": message[:200],
            "sequence": 0,
            "created_at": now.isoformat(),
        }
        event_json = json.dumps(event_template, ensure_ascii=False)

        seq_key = _SEQ_KEY.format(run_id=run_id)
        latest_key = _LATEST_KEY.format(run_id=run_id)

        try:
            sha = await self._ensure_script()
            seq = await self._redis.evalsha(
                sha, 2, seq_key, latest_key,
                self._ttl, event_json, _CHANNEL,
            )
        except Exception:
            # Script cache miss → reload and retry once
            try:
                self._script_sha = await self._ensure_script()
                seq = await self._redis.evalsha(
                    self._script_sha, 2, seq_key, latest_key,
                    self._ttl, event_json, _CHANNEL,
                )
            except Exception as e:
                logger.error(f"ProgressBus publish failed: {e}")
                # 返回一个本地构造的事件，不让评估失败
                seq = 1

        event = EvaluationProgressEvent(
            run_id=run_id,
            consultation_id=consultation_id,
            status=status,  # type: ignore[arg-type]
            progress=progress,
            message=message[:200],
            sequence=seq,
            created_at=now,
        )
        return event

    async def get_latest(self, run_id: str) -> Optional[EvaluationProgressEvent]:
        """获取最新进度事件（失败返回 None）"""
        try:
            latest_key = _LATEST_KEY.format(run_id=run_id)
            data = await self._redis.get(latest_key)
            if data is None:
                return None
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return EvaluationProgressEvent.model_validate_json(data)
        except Exception as e:
            logger.warning(f"ProgressBus get_latest failed: {e}")
            return None

    async def listen(
        self,
        on_event: Callable[[EvaluationProgressEvent], Awaitable[None]],
    ) -> None:
        """订阅 Redis Pub/Sub 并持续监听，异常时退避重连"""
        backoff_schedule = [1, 2, 4, 8, 10]
        attempt = 0

        while not self._closed:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(_CHANNEL)
                attempt = 0  # 连接成功重置退避

                async for message in pubsub.listen():
                    if self._closed:
                        break
                    if message["type"] != "message":
                        continue
                    raw = message["data"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        event = EvaluationProgressEvent.model_validate_json(raw)
                        await on_event(event)
                    except Exception as e:
                        logger.warning(f"ProgressBus listener: corrupt message ignored: {e}")
                        continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._closed:
                    break
                delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                jitter = random.uniform(0, delay * 0.3)
                wait = delay + jitter
                logger.warning(f"ProgressBus listener error: {e}, reconnecting in {wait:.1f}s")
                await asyncio.sleep(wait)
                attempt += 1

    async def close(self) -> None:
        """取消 listener 并关闭连接"""
        self._closed = True
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass


# ── FakeProgressBus（测试替身）────────────────────────────────────────────────


class FakeProgressBus:
    """内存实现的 ProgressBus，用于单元测试和集成测试。"""

    def __init__(self, ttl: int = 3600):
        self._ttl = ttl
        self._latest: dict[str, str] = {}  # run_id -> event JSON
        self._sequences: dict[str, int] = {}  # run_id -> current seq
        self._subscribers: list[Callable] = []
        self._closed = False

    async def publish(
        self,
        *,
        run_id: str,
        consultation_id: int,
        status: str,
        progress: int,
        message: str,
    ) -> EvaluationProgressEvent:
        seq = self._sequences.get(run_id, 0) + 1
        self._sequences[run_id] = seq

        now = datetime.now(timezone.utc)
        event = EvaluationProgressEvent(
            run_id=run_id,
            consultation_id=consultation_id,
            status=status,  # type: ignore[arg-type]
            progress=progress,
            message=message[:200],
            sequence=seq,
            created_at=now,
        )
        self._latest[run_id] = event.model_dump_json()

        # Fan-out to subscribers
        for sub in self._subscribers:
            try:
                await sub(event)
            except Exception as e:
                logger.warning(f"FakeProgressBus subscriber error: {e}")

        return event

    async def get_latest(self, run_id: str) -> Optional[EvaluationProgressEvent]:
        data = self._latest.get(run_id)
        if data is None:
            return None
        try:
            return EvaluationProgressEvent.model_validate_json(data)
        except Exception:
            return None

    def get_latest_ttl(self, run_id: str) -> Optional[int]:
        """测试用：返回 latest key 的 TTL（FakeProgressBus 固定返回配置 TTL）"""
        if run_id in self._latest:
            return self._ttl
        return None

    async def listen(
        self,
        on_event: Callable[[EvaluationProgressEvent], Awaitable[None]],
    ) -> None:
        self._subscribers.append(on_event)
        try:
            while not self._closed:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            if on_event in self._subscribers:
                self._subscribers.remove(on_event)

    async def close(self) -> None:
        self._closed = True
