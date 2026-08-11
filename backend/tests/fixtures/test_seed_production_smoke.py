"""Regression tests for the persistent V1.1 E2E seed."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.fixtures import seed_production_smoke as seed


def test_seed_refuses_non_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="ENVIRONMENT must be 'test'"):
        seed.check_environment()


def test_seed_persists_users_patient_consultation_and_messages() -> None:
    session = MagicMock()
    session.execute.return_value.scalar_one.side_effect = [11, 12]

    users, doctor_id, admin_id = seed.seed_users(session)
    patient, consultation, messages = seed.seed_patient_and_consultation(
        session, doctor_id
    )

    statements = [str(item.args[0]) for item in session.execute.call_args_list]
    assert len(users) == 2
    assert (doctor_id, admin_id) == (11, 12)
    assert patient["id"] == seed.FIXED_PATIENT_ID
    assert consultation["id"] == seed.FIXED_CONSULTATION_ID
    assert consultation["doctor_id"] == 11
    assert consultation["status"] == "completed"
    assert len(messages) >= 4
    assert any("INSERT INTO users" in statement for statement in statements)
    assert any("INSERT INTO virtual_patients" in statement for statement in statements)
    assert any("INSERT INTO consultations" in statement for statement in statements)
    assert sum(
        "INSERT INTO consultation_messages" in statement
        for statement in statements
    ) == len(messages)


def test_low_evidence_manifest_checksum_is_deterministic() -> None:
    chunk, first = seed.seed_low_evidence_index()
    _, second = seed.seed_low_evidence_index()

    assert chunk["metadata"]["is_synthetic"] is True
    assert first["chunks"][0]["checksum"] == second["chunks"][0]["checksum"]
