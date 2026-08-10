"""Episodic memory: summaries of past consultations for context."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EpisodeSummary:
    """A summary of a past consultation episode."""
    consultation_id: int
    summary: str
    key_findings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EpisodicMemory:
    """Collection of episode summaries."""
    episodes: list[EpisodeSummary] = field(default_factory=list)

    def add_episode(self, episode: EpisodeSummary) -> None:
        self.episodes.append(episode)

    def get_recent(self, limit: int = 3) -> list[EpisodeSummary]:
        return sorted(self.episodes, key=lambda e: e.created_at, reverse=True)[:limit]
