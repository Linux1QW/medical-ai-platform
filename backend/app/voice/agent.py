"""Voice agent for LiveKit integration (Beta).

This module defines the voice agent interface. The actual LiveKit
agent implementation requires the livekit-agents package which is
not installed by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VoiceTranscript:
    """A segment of voice transcript."""
    text: str
    speaker: str  # "doctor" or "patient"
    start_time: float
    end_time: float
    is_final: bool = True


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
    """Voice agent placeholder.

    In production, this would integrate with LiveKit's Agent Framework.
    For the beta, it provides the interface contract.

    Key constraints:
    - Does NOT save raw audio (privacy requirement)
    - Room token TTL: 10 minutes
    - HMAC-based identity
    """

    def __init__(self, config: VoiceAgentConfig) -> None:
        self.config = config
        self.transcripts: list[VoiceTranscript] = []
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self) -> None:
        """Start the voice agent (placeholder)."""
        self._is_running = True

    async def stop(self) -> None:
        """Stop the voice agent (placeholder)."""
        self._is_running = False

    async def process_transcript(self, transcript: VoiceTranscript) -> dict[str, Any]:
        """Process a transcript segment (placeholder).

        In production, this would feed the transcript into the coach graph.
        """
        self.transcripts.append(transcript)
        return {
            "text": transcript.text,
            "speaker": transcript.speaker,
            "processed": True,
        }

    def get_transcript_summary(self) -> str:
        """Get a summary of all transcripts."""
        return "\n".join(
            f"[{t.speaker}] {t.text}" for t in self.transcripts if t.is_final
        )
