"""Tests for working memory."""
import pytest
from app.services.memory.working import (
    WorkingMemoryState,
    SlotObservation,
    MemoryConflict,
    HiddenContextViolation,
    validate_memory_sources,
)


def test_denied_fact_cannot_become_positive_without_correction():
    """A denied slot cannot flip to positive without explicit correction."""
    state = WorkingMemoryState()
    state.apply(SlotObservation(
        key="drug_allergy", value="penicillin",
        polarity="negative", turn=2, source_sequence=3,
    ))
    with pytest.raises(MemoryConflict):
        state.apply(SlotObservation(
            key="drug_allergy", value="penicillin",
            polarity="positive", turn=4, source_sequence=7,
        ))


def test_accepted_fact_can_be_reinforced():
    """An accepted slot can be reinforced with same polarity."""
    state = WorkingMemoryState()
    state.apply(SlotObservation(
        key="onset", value="3天前",
        polarity="positive", turn=1, source_sequence=2,
    ))
    state.apply(SlotObservation(
        key="onset", value="3天前开始",
        polarity="positive", turn=3, source_sequence=6,
    ))
    assert state.get_slot("onset").polarity == "positive"


def test_repeated_question_detection():
    """Working memory tracks asked dimensions and detects repeats."""
    state = WorkingMemoryState()
    state.mark_asked("hpi_onset", turn=1)
    assert state.is_repeated("hpi_onset", within_turns=3, current_turn=2)
    assert not state.is_repeated("hpi_onset", within_turns=3, current_turn=5)


def test_validate_memory_sources_rejects_hidden():
    """Memory sources must all be in visible message sequences."""
    state = WorkingMemoryState()
    state.apply(SlotObservation(
        key="hidden_fact", value="secret",
        polarity="positive", turn=1, source_sequence=99,
    ))
    visible_sequences = {1, 2, 3}
    with pytest.raises(HiddenContextViolation):
        validate_memory_sources(state, visible_sequences)


def test_validate_memory_sources_accepts_visible():
    """Memory sources that are all visible pass validation."""
    state = WorkingMemoryState()
    state.apply(SlotObservation(
        key="onset", value="3天前",
        polarity="positive", turn=1, source_sequence=2,
    ))
    visible_sequences = {1, 2, 3}
    validate_memory_sources(state, visible_sequences)  # no exception
