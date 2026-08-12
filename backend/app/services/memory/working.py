"""Working memory for coach agent: tracks slots, asked dimensions, repeated questions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


class MemoryConflict(Exception):
    """Raised when a memory observation conflicts with existing state."""


class HiddenContextViolation(Exception):
    """Raised when working memory contains non-visible sources."""


@dataclass
class SlotObservation:
    """An observation about a clinical slot."""
    key: str
    value: str
    polarity: Literal["positive", "negative", "uncertain"]
    turn: int
    source_sequence: int
    confidence: float = 0.8


@dataclass
class SlotState:
    """Current state of a clinical slot."""
    key: str
    value: str
    polarity: Literal["positive", "negative", "uncertain"]
    source_sequence: int
    turn: int
    confidence: float
    history: list[SlotObservation] = field(default_factory=list)


@dataclass
class WorkingMemoryState:
    """Working memory tracking asked dimensions, slots, and repeated questions."""
    slots: dict[str, SlotState] = field(default_factory=dict)
    asked_dimensions: dict[str, int] = field(default_factory=dict)  # dimension -> last_asked_turn
    stage_history: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)

    def apply(self, observation: SlotObservation) -> None:
        """Apply a slot observation, checking for conflicts."""
        existing = self.slots.get(observation.key)
        if existing is not None:
            # Conflict: denied (negative) cannot flip to positive without correction
            if existing.polarity == "negative" and observation.polarity == "positive":
                raise MemoryConflict(
                    f"Slot '{observation.key}' was denied (negative) but now "
                    f"observed as positive without explicit correction"
                )
            # Update with new observation
            existing.history.append(observation)
            existing.value = observation.value
            existing.polarity = observation.polarity
            existing.source_sequence = observation.source_sequence
            existing.turn = observation.turn
            existing.confidence = observation.confidence
        else:
            self.slots[observation.key] = SlotState(
                key=observation.key,
                value=observation.value,
                polarity=observation.polarity,
                source_sequence=observation.source_sequence,
                turn=observation.turn,
                confidence=observation.confidence,
                history=[observation],
            )

    def get_slot(self, key: str) -> SlotState | None:
        """Get current state of a slot."""
        return self.slots.get(key)

    def mark_asked(self, dimension: str, turn: int) -> None:
        """Mark a dimension as asked at a given turn."""
        self.asked_dimensions[dimension] = turn

    def is_repeated(self, dimension: str, *, within_turns: int, current_turn: int) -> bool:
        """Check if a dimension was asked within the last N turns."""
        last_asked = self.asked_dimensions.get(dimension)
        if last_asked is None:
            return False
        return (current_turn - last_asked) <= within_turns


def validate_memory_sources(
    memory: WorkingMemoryState, visible_sequences: set[int]
) -> None:
    """Validate that all memory sources are in visible message sequences."""
    invalid = [
        slot for slot in memory.slots.values()
        if slot.source_sequence not in visible_sequences
    ]
    if invalid:
        raise HiddenContextViolation(
            f"Working memory contains {len(invalid)} non-visible sources: "
            f"{[s.key for s in invalid]}"
        )
