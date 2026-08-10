"""72-case coach benchmark dataset loader and validator."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

CASES_FILE = os.path.join(os.path.dirname(__file__), "coach_v1.jsonl")

VALID_SPECIALTIES = {"internal", "surgery", "pediatrics", "obstetrics_gynecology", "emergency", "psychiatry"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_PERSONALITIES = {"cooperative", "anxious", "evasive", "elderly_confused"}
VALID_INTENTS = {
    "rapport", "chief_complaint", "hpi_onset", "hpi_progression", "hpi_severity",
    "hpi_timing", "hpi_associated", "past_history", "medication", "allergy",
    "family_social", "examination", "assessment_communication", "treatment_communication",
    "closing", "off_topic", "unsafe",
}

VALID_ROLES = {"doctor", "patient"}


@dataclass
class CoachCase:
    """A single benchmark case."""
    case_id: str
    specialty: str
    difficulty: str
    personality: str
    visible_context: dict[str, Any]
    dialogue_prefix: list[dict[str, str]]
    expected_intent: str
    expected_stage: str
    forbidden_hidden_facts: list[str]
    target_slots: list[str]


def load_cases(path: str | None = None) -> list[CoachCase]:
    """Load all benchmark cases from JSONL."""
    filepath = path or CASES_FILE
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(CoachCase(
                case_id=data["case_id"],
                specialty=data["specialty"],
                difficulty=data["difficulty"],
                personality=data["personality"],
                visible_context=data["visible_context"],
                dialogue_prefix=data["dialogue_prefix"],
                expected_intent=data["expected_intent"],
                expected_stage=data["expected_stage"],
                forbidden_hidden_facts=data["forbidden_hidden_facts"],
                target_slots=data["target_slots"],
            ))
    return cases


def validate_dataset(cases: list[CoachCase]) -> list[str]:
    """Validate the dataset. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    # Check total count
    if len(cases) != 72:
        errors.append(f"Expected 72 cases, got {len(cases)}")

    # Check uniqueness
    case_ids = [c.case_id for c in cases]
    if len(set(case_ids)) != len(case_ids):
        errors.append("Duplicate case_ids found")

    # Check matrix completeness
    combos = {(c.specialty, c.difficulty, c.personality) for c in cases}
    expected = {(s, d, p) for s in VALID_SPECIALTIES for d in VALID_DIFFICULTIES for p in VALID_PERSONALITIES}
    missing = expected - combos
    if missing:
        errors.append(f"Missing {len(missing)} combinations: {missing}")

    # Validate each case
    for case in cases:
        if case.specialty not in VALID_SPECIALTIES:
            errors.append(f"{case.case_id}: invalid specialty '{case.specialty}'")
        if case.difficulty not in VALID_DIFFICULTIES:
            errors.append(f"{case.case_id}: invalid difficulty '{case.difficulty}'")
        if case.personality not in VALID_PERSONALITIES:
            errors.append(f"{case.case_id}: invalid personality '{case.personality}'")
        if case.expected_intent not in VALID_INTENTS:
            errors.append(f"{case.case_id}: invalid intent '{case.expected_intent}'")
        if not case.dialogue_prefix:
            errors.append(f"{case.case_id}: empty dialogue_prefix")
        if not case.forbidden_hidden_facts:
            errors.append(f"{case.case_id}: empty forbidden_hidden_facts")
        if not case.visible_context.get("chief_complaint"):
            errors.append(f"{case.case_id}: missing chief_complaint")

    return errors
