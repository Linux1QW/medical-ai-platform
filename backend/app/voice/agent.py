"""Voice agent for LiveKit integration (Beta).

Connects to a LiveKit room, processes final transcripts,
deduplicates by (room_sid, participant_sid, turn_id), and sends
final doctor transcripts through the existing Consultation service.

Key constraints:
- Partial transcripts remain memory-only (not persisted)
- Raw audio recording stays disabled (privacy requirement)
- Deduplication key: (room_sid, participant_sid, turn_id)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VoiceTranscript:
    """A segment of voice transcript."""

    text: str
    speaker: str  # "doctor" or "patient"
    start_time: float
    end_time: float
    is_final: bool = True
    # Deduplication fields
    room_sid: str = ""
    participant_sid: str = ""
    turn_id: str = ""


@dataclass
class VoiceAgentConfig:
    """Configuration for the voice agent."""

    room_name: str
    consultation_id: int
    doctor_id: int
    language: str = "zh-CN"
    max_duration_seconds: int = 1800  # 30 minutes
    save_raw_audio: bool = False  # NEVER save raw audio (privacy)


class VoiceAgent:
    """Voice agent that processes final transcripts from LiveKit room.

    - Deduplicates final transcripts by (room_sid, participant_sid, turn_id)
    - Sends final doctor transcript through existing Consultation service
    - Synthesizes returned patient text
    - Partial transcripts remain memory-only
    - Raw audio recording stays disabled
    """

    def __init__(self, config: VoiceAgentConfig) -> None:
        self.config = config
        self.transcripts: list[VoiceTranscript] = []
        self._is_running = False
        # Deduplication set: stores (room_sid, participant_sid, turn_id) tuples
        self._seen_turns: set[tuple[str, str, str]] = set()
        # Patient response synthesis callback (set externally)
        self._patient_synth_callback: Any = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_patient_synth_callback(self, callback: Any) -> None:
        """Set callback for synthesizing patient responses."""
        self._patient_synth_callback = callback

    async def start(self) -> None:
        """Start the voice agent."""
        self._is_running = True
        logger.info("VoiceAgent started for room=%s", self.config.room_name)

    async def stop(self) -> None:
        """Stop the voice agent."""
        self._is_running = False
        logger.info("VoiceAgent stopped for room=%s", self.config.room_name)

    async def process_transcript(self, transcript: VoiceTranscript) -> dict[str, Any]:
        """Process a transcript segment.

        - Partial transcripts: kept in memory only, not persisted
        - Final transcripts: deduplicated by (room_sid, participant_sid, turn_id)
        - Doctor final transcripts: sent through Consultation service
        - Returns synthesized patient text if applicable
        """
        # Partial transcripts: memory only
        if not transcript.is_final:
            self.transcripts.append(transcript)
            return {
                "text": transcript.text,
                "speaker": transcript.speaker,
                "processed": True,
                "persisted": False,
            }

        # Final transcript: deduplicate
        dedup_key = (
            transcript.room_sid,
            transcript.participant_sid,
            transcript.turn_id,
        )
        if dedup_key in self._seen_turns:
            logger.debug("Duplicate final transcript skipped: %s", dedup_key)
            return {
                "text": transcript.text,
                "speaker": transcript.speaker,
                "processed": False,
                "duplicated": True,
            }

        self._seen_turns.add(dedup_key)
        self.transcripts.append(transcript)

        result: dict[str, Any] = {
            "text": transcript.text,
            "speaker": transcript.speaker,
            "processed": True,
            "persisted": True,
            "duplicated": False,
        }

        # If doctor spoke, send through Consultation service and synthesize patient reply
        if transcript.speaker == "doctor" and self._patient_synth_callback:
            try:
                patient_reply = await self._patient_synth_callback(
                    consultation_id=self.config.consultation_id,
                    doctor_text=transcript.text,
                )
                result["patient_response"] = patient_reply
            except Exception:
                logger.exception("Failed to synthesize patient response")
                result["patient_response"] = None

        return result

    def get_transcript_summary(self) -> str:
        """Get a summary of all final transcripts."""
        return "\n".join(
            f"[{t.speaker}] {t.text}"
            for t in self.transcripts
            if t.is_final
        )
