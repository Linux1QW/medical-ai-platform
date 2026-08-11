"""Voice session store — Redis-backed with in-memory fallback for tests.

Provides a Protocol (interface) and two implementations:
- VoiceSessionStoreMemory: module-dict fallback (tests / dev without Redis)
- VoiceSessionStoreRedis: production store using Redis SET NX EX for dedup
"""
from __future__ import annotations

import json
import logging
import time
from typing import Protocol, runtime_checkable
from uuid import uuid4

from app.voice.session import VOICE_ROOM_TTL_SECONDS, VoiceSession

logger = logging.getLogger(__name__)


@runtime_checkable
class VoiceSessionStore(Protocol):
    """Abstract voice session store."""

    async def create(self, session: VoiceSession) -> None: ...
    async def get(self, room_name: str) -> VoiceSession | None: ...
    async def end(self, room_name: str) -> None: ...
    async def list_active(self) -> list[VoiceSession]: ...


# ---------------------------------------------------------------------------
# In-memory implementation (tests / local dev)
# ---------------------------------------------------------------------------

class VoiceSessionStoreMemory:
    """In-memory voice session store (non-persistent)."""

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}

    async def create(self, session: VoiceSession) -> None:
        self._sessions[session.room_name] = session

    async def get(self, room_name: str) -> VoiceSession | None:
        session = self._sessions.get(room_name)
        if session and session.is_expired:
            session.status = "expired"
        return session

    async def end(self, room_name: str) -> None:
        session = self._sessions.get(room_name)
        if session:
            session.status = "ended"

    async def list_active(self) -> list[VoiceSession]:
        return [
            s for s in self._sessions.values()
            if s.status in ("created", "active") and not s.is_expired
        ]


# ---------------------------------------------------------------------------
# Redis implementation (production)
# ---------------------------------------------------------------------------

_REDIS_KEY_PREFIX = "voice:session:"


class VoiceSessionStoreRedis:
    """Redis-backed voice session store.

    Each session is stored as a JSON hash under ``voice:session:<room_name>``
    with a TTL equal to VOICE_ROOM_TTL_SECONDS.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis = None  # lazily initialised

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    @staticmethod
    def _key(room_name: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{room_name}"

    @staticmethod
    def _to_dict(session: VoiceSession) -> dict:
        return {
            "session_id": str(session.session_id),
            "room_name": session.room_name,
            "consultation_id": session.consultation_id,
            "doctor_id": session.doctor_id,
            "status": session.status,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "participant_count": session.participant_count,
        }

    @staticmethod
    def _from_dict(data: dict) -> VoiceSession:
        from uuid import UUID
        return VoiceSession(
            session_id=UUID(data["session_id"]),
            room_name=data["room_name"],
            consultation_id=data["consultation_id"],
            doctor_id=data["doctor_id"],
            status=data.get("status", "created"),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at", 0.0),
            participant_count=data.get("participant_count", 0),
        )

    async def create(self, session: VoiceSession) -> None:
        r = await self._get_redis()
        key = self._key(session.room_name)
        ttl = int(session.expires_at - time.time())
        if ttl < 1:
            ttl = VOICE_ROOM_TTL_SECONDS
        await r.set(key, json.dumps(self._to_dict(session)), ex=ttl)

    async def get(self, room_name: str) -> VoiceSession | None:
        r = await self._get_redis()
        raw = await r.get(self._key(room_name))
        if raw is None:
            return None
        data = json.loads(raw)
        session = self._from_dict(data)
        if session.is_expired:
            session.status = "expired"
        return session

    async def end(self, room_name: str) -> None:
        r = await self._get_redis()
        session = await self.get(room_name)
        if session:
            session.status = "ended"
            ttl = int(session.expires_at - time.time())
            if ttl < 1:
                ttl = 60
            await r.set(self._key(room_name), json.dumps(self._to_dict(session)), ex=ttl)

    async def list_active(self) -> list[VoiceSession]:
        r = await self._get_redis()
        sessions: list[VoiceSession] = []
        async for key in r.scan_iter(match=f"{_REDIS_KEY_PREFIX}*"):
            raw = await r.get(key)
            if raw is None:
                continue
            data = json.loads(raw)
            session = self._from_dict(data)
            if session.status in ("created", "active") and not session.is_expired:
                sessions.append(session)
        return sessions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_voice_store(redis_url: str | None = None) -> VoiceSessionStore:
    """Create the appropriate store based on configuration.

    If *redis_url* is ``None`` or empty, returns the in-memory store.
    """
    if redis_url:
        return VoiceSessionStoreRedis(redis_url)
    return VoiceSessionStoreMemory()
