"""Tests for 72-case coach benchmark dataset."""
from __future__ import annotations

import pytest

from evaluation.coach_cases.coach_dataset import (
    VALID_INTENTS,
    VALID_ROLES,
    CoachCase,
    load_cases,
    validate_dataset,
)


@pytest.fixture(scope="module")
def cases() -> list[CoachCase]:
    """Load all benchmark cases once per module."""
    return load_cases()


def test_load_72_cases(cases: list[CoachCase]) -> None:
    """Load returns exactly 72 cases."""
    assert len(cases) == 72


def test_all_case_ids_unique(cases: list[CoachCase]) -> None:
    """No duplicate case_ids."""
    case_ids = [c.case_id for c in cases]
    assert len(set(case_ids)) == len(case_ids), "Duplicate case_ids found"


def test_matrix_complete(cases: list[CoachCase]) -> None:
    """All 6×3×4 combinations present."""
    combos = {(c.specialty, c.difficulty, c.personality) for c in cases}
    expected_count = 6 * 3 * 4  # 72
    assert len(combos) == expected_count, f"Expected {expected_count} unique combos, got {len(combos)}"


def test_validate_dataset_passes(cases: list[CoachCase]) -> None:
    """validate_dataset returns no errors."""
    errors = validate_dataset(cases)
    assert errors == [], f"Validation errors: {errors}"


def test_no_hidden_facts_in_visible(cases: list[CoachCase]) -> None:
    """forbidden_hidden_facts values don't appear in dialogue_prefix or visible_context."""
    for case in cases:
        # Collect all visible text
        visible_text = case.visible_context.get("chief_complaint", "")
        for msg in case.dialogue_prefix:
            visible_text += " " + msg["content"]

        for fact in case.forbidden_hidden_facts:
            # Extract the value part after the colon
            if ":" in fact:
                fact_value = fact.split(":", 1)[1]
            else:
                fact_value = fact
            assert fact_value not in visible_text, (
                f"{case.case_id}: hidden fact '{fact_value}' found in visible context"
            )


def test_all_intents_valid(cases: list[CoachCase]) -> None:
    """All expected_intent values are in VALID_INTENTS."""
    for case in cases:
        assert case.expected_intent in VALID_INTENTS, (
            f"{case.case_id}: invalid intent '{case.expected_intent}'"
        )


def test_each_specialty_has_12_cases(cases: list[CoachCase]) -> None:
    """Each specialty has exactly 12 cases."""
    from collections import Counter
    specialty_counts = Counter(c.specialty for c in cases)
    for specialty, count in specialty_counts.items():
        assert count == 12, f"Specialty '{specialty}' has {count} cases, expected 12"
    assert len(specialty_counts) == 6, f"Expected 6 specialties, got {len(specialty_counts)}"


def test_dialogue_prefix_has_roles(cases: list[CoachCase]) -> None:
    """All dialogue messages have valid roles."""
    for case in cases:
        for i, msg in enumerate(case.dialogue_prefix):
            assert "role" in msg, f"{case.case_id} msg[{i}]: missing 'role'"
            assert "content" in msg, f"{case.case_id} msg[{i}]: missing 'content'"
            assert msg["role"] in VALID_ROLES, (
                f"{case.case_id} msg[{i}]: invalid role '{msg['role']}'"
            )
