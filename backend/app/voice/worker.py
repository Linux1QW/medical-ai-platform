"""Voice Worker — connects to a LiveKit room and processes transcripts.

Responsibilities:
- Connect to LiveKit Room using an agent token
- Receive Final Transcripts from the doctor participant
- Deduplicate final transcripts via Redis SET NX EX (key = room_sid:participant_sid:turn_id)
- Send doctor text through the existing Consultation message pipeline
- Synthesize patient reply and publish it back to the Room
- Partial transcripts: kept in a bounded in-memory cache only (never persisted)
- Raw audio: NEVER recorded (privacy requirement)

The worker is designed to be spawned as an asyncio task when a voice session
is created and cancelled when the session ends.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Type alias for the callback that sends a doctor message and returns the patient reply
DoctorMessageHandler = Callable[[int, str], Awaitable[str]]


class VoiceWorker:
    """Processes voice transcripts from a LiveKit room.

    Parameters
    ----------
    room_name:
        LiveKit room identifier.
    consultation_id:
        The consultation this worker belongs to.
    doctor_id:
        The doctor participating in the room.
    message_handler:
        Async callable ``(consultation_id, doctor_text) -> patient_reply_text``.
    dedup_redis_url:
        Redis URL for transcript deduplication.  When *None*, dedup falls
        back to an in-memory set (tests / single-process dev).
    """

    # Partial transcript cache bounded to this many entries
    _PARTIAL_CACHE_MAX = 200

    def __init__(
        self,
        *,
        room_name: str,
        consultation_id: int,
        doctor_id: int,
        message_handler: DoctorMessageHandler | None = None,
        dedup_redis_url: str | None = None,
    ) -> None:
        self.room_name = room_name
        self.consultation_id = consultation_id
        self.doctor_id = doctor_id
        self._message_handler = message_handler
        self._dedup_redis_url = dedup_redis_url

        self._is_running = False
        self._task: asyncio.Task[None] | None = None

        # In-memory dedup fallback (used when Redis is not configured)
        self._seen_turns: set[str] = set()
        # In-memory partial transcript ring buffer (never persisted)
        self._partial_cache: list[dict[str, Any]] = []

        # Redis client for dedup (lazily initialised)
        self._dedup_redis = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the worker (idempotent)."""
        if self._is_running:
            return
        self._is_running = True
        logger.info("VoiceWorker started room=%s consultation=%s", self.room_name, self.consultation_id)

    async def stop(self) -> None:
        """Stop the worker and clean up resources."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        # Close Redis connection if any
        if self._dedup_redis is not None:
            await self._dedup_redis.close()
            self._dedup_redis = None
        logger.info("VoiceWorker stopped room=%s", self.room_name)

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ------------------------------------------------------------------
    # Transcript handling
    # ------------------------------------------------------------------

    async def handle_final_transcript(
        self,
        *,
        text: str,
        speaker: str,
        room_sid: str,
        participant_sid: str,
        turn_id: str,
    ) -> dict[str, Any]:
        """Process a final transcript segment.

        Returns a dict with at least ``{"accepted": bool}``.
        """
        if not self._is_running:
            return {"accepted": False, "reason": "worker_not_running"}

        dedup_key = f"{room_sid}:{participant_sid}:{turn_id}"

        # --- Deduplication ---
        is_dup = await self._check_dedup(dedup_key)
        if is_dup:
            logger.debug("Duplicate final transcript skipped: %s", dedup_key)
            return {"accepted": False, "reason": "duplicate"}

        result: dict[str, Any] = {"accepted": True, "speaker": speaker, "text": text}

        # --- Doctor message → send through consultation pipeline ---
        if speaker == "doctor" and self._message_handler and text.strip():
            try:
                patient_reply = await self._message_handler(
                    self.consultation_id, text.strip()
                )
                result["patient_response"] = patient_reply
            except Exception:
                logger.exception("Failed to process doctor message")
                result["patient_response"] = None

        return result

    def buffer_partial_transcript(self, *, text: str, speaker: str) -> None:
        """Buffer a partial transcript in memory (never persisted).

        The cache is bounded; oldest entries are evicted when full.
        """
        self._partial_cache.append({"text": text, "speaker": speaker})
        if len(self._partial_cache) > self._PARTIAL_CACHE_MAX:
            # Evict oldest half
            self._partial_cache = self._partial_cache[len(self._partial_cache) // 2:]

    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------

    async def _check_dedup(self, key: str) -> bool:
        """Return True if *key* was already seen (duplicate).

        Uses Redis SET NX EX when available; falls back to in-memory set.
        TTL for dedup keys: 600 seconds (matches room TTL).
        """
        if self._dedup_redis_url:
            return await self._check_dedup_redis(key)
        # In-memory fallback
        if key in self._seen_turns:
            return True
        self._seen_turns.add(key)
        return False

    async def _check_dedup_redis(self, key: str) -> bool:
        """Redis-based dedup using SET NX EX."""
        if self._dedup_redis is None:
            import redis.asyncio as aioredis
            self._dedup_redis = aioredis.from_url(
                self._dedup_redis_url, decode_responses=True
            )
        conn = self._dedup_redis
        assert conn is not None
        redis_key = f"voice:dedup:{self.room_name}:{key}"
        # SET NX EX: set only if not exists, with 600s TTL
        was_set = await conn.set(redis_key, "1", nx=True, ex=600)
        return was_set is None  # None means key already existed → duplicate
