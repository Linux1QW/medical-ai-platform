"""Tests for prompt registry: immutable bundles, deterministic assignment, staged rollout."""
from __future__ import annotations

import pytest

from app.agent_runtime.prompt_registry import (
    PromptRegistry,
    deterministic_assign,
)

# ---------------------------------------------------------------------------
# 1. test_register_bundle
# ---------------------------------------------------------------------------

def test_register_bundle() -> None:
    registry = PromptRegistry()
    bundle = registry.register_bundle(
        name="问诊评估v1",
        version="1.0.0",
        system_prompt="你是一名资深内科医生。",
        safety_policy="禁止给出处方建议。",
    )
    assert bundle.bundle_id == "bundle_000001"
    assert bundle.name == "问诊评估v1"
    assert bundle.version == "1.0.0"
    # content_hash should be a 64-char hex string (SHA-256)
    assert len(bundle.content_hash) == 64
    assert all(c in "0123456789abcdef" for c in bundle.content_hash)


# ---------------------------------------------------------------------------
# 2. test_bundle_immutable — same content → same hash
# ---------------------------------------------------------------------------

def test_bundle_immutable() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(
        name="v1", version="1.0",
        system_prompt="prompt-A", safety_policy="policy-X",
    )
    b2 = registry.register_bundle(
        name="v1-copy", version="1.0",
        system_prompt="prompt-A", safety_policy="policy-X",
    )
    assert b1.content_hash == b2.content_hash
    # But bundle_ids must differ (each registration is unique)
    assert b1.bundle_id != b2.bundle_id


# ---------------------------------------------------------------------------
# 3. test_get_bundle
# ---------------------------------------------------------------------------

def test_get_bundle() -> None:
    registry = PromptRegistry()
    bundle = registry.register_bundle(
        name="test", version="0.1",
        system_prompt="sys", safety_policy="safe",
    )
    retrieved = registry.get_bundle(bundle.bundle_id)
    assert retrieved is not None
    assert retrieved.bundle_id == bundle.bundle_id
    assert retrieved.system_prompt == "sys"
    # Non-existent ID returns None
    assert registry.get_bundle("bundle_999999") is None


# ---------------------------------------------------------------------------
# 4. test_create_experiment — stage=canary_0, rollout=0%
# ---------------------------------------------------------------------------

def test_create_experiment() -> None:
    registry = PromptRegistry()
    baseline = registry.register_bundle(
        name="baseline", version="1.0",
        system_prompt="old", safety_policy="safe",
    )
    treatment = registry.register_bundle(
        name="treatment", version="2.0",
        system_prompt="new", safety_policy="safe",
    )
    exp = registry.create_experiment(
        name="prompt-ab-test",
        baseline_bundle_id=baseline.bundle_id,
        treatment_bundle_id=treatment.bundle_id,
    )
    assert exp.experiment_id == "exp_000003"  # counter=3 after 2 bundles + 1 exp
    assert exp.stage == "canary_0"
    assert exp.rollout_percentage == 0
    assert exp.auto_rollback_triggered is False


# ---------------------------------------------------------------------------
# 5. test_advance_stage — 0→5→25→100
# ---------------------------------------------------------------------------

def test_advance_stage() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(name="b1", version="1", system_prompt="a", safety_policy="b")
    b2 = registry.register_bundle(name="b2", version="1", system_prompt="c", safety_policy="d")
    exp = registry.create_experiment(name="exp", baseline_bundle_id=b1.bundle_id, treatment_bundle_id=b2.bundle_id)

    assert exp.stage == "canary_0"
    assert exp.rollout_percentage == 0

    exp.advance_stage()
    assert exp.stage == "canary_5"
    assert exp.rollout_percentage == 5

    exp.advance_stage()
    assert exp.stage == "canary_25"
    assert exp.rollout_percentage == 25

    exp.advance_stage()
    assert exp.stage == "full_100"
    assert exp.rollout_percentage == 100


# ---------------------------------------------------------------------------
# 6. test_advance_stage_max — past full_100 stays
# ---------------------------------------------------------------------------

def test_advance_stage_max() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(name="b1", version="1", system_prompt="a", safety_policy="b")
    b2 = registry.register_bundle(name="b2", version="1", system_prompt="c", safety_policy="d")
    exp = registry.create_experiment(name="exp", baseline_bundle_id=b1.bundle_id, treatment_bundle_id=b2.bundle_id)

    # Advance to full_100
    for _ in range(3):
        exp.advance_stage()
    assert exp.stage == "full_100"

    # Advancing again should stay at full_100
    result = exp.advance_stage()
    assert result == "full_100"
    assert exp.stage == "full_100"


# ---------------------------------------------------------------------------
# 7. test_deterministic_assign_stable — same doctor_id → same result
# ---------------------------------------------------------------------------

def test_deterministic_assign_stable() -> None:
    results = [deterministic_assign(doctor_id=42, experiment_id="exp_001", percentage=50) for _ in range(10)]
    assert len(set(results)) == 1, "Deterministic assignment must be stable across calls"


# ---------------------------------------------------------------------------
# 8. test_deterministic_assign_0_percent — canary_0 → nobody
# ---------------------------------------------------------------------------

def test_deterministic_assign_0_percent() -> None:
    # With 0%, nobody should be assigned to treatment
    results = [deterministic_assign(doctor_id=did, experiment_id="exp_zero", percentage=0) for did in range(100)]
    assert all(r is False for r in results)


# ---------------------------------------------------------------------------
# 9. test_deterministic_assign_100_percent — full_100 → everyone
# ---------------------------------------------------------------------------

def test_deterministic_assign_100_percent() -> None:
    # With 100%, everyone should be assigned to treatment
    results = [deterministic_assign(doctor_id=did, experiment_id="exp_full", percentage=100) for did in range(100)]
    assert all(r is True for r in results)


# ---------------------------------------------------------------------------
# 10. test_auto_rollback_hidden_leak — hidden_leak → all get baseline
# ---------------------------------------------------------------------------

def test_auto_rollback_hidden_leak() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(name="b1", version="1", system_prompt="a", safety_policy="b")
    b2 = registry.register_bundle(name="b2", version="1", system_prompt="c", safety_policy="d")
    exp = registry.create_experiment(name="exp", baseline_bundle_id=b1.bundle_id, treatment_bundle_id=b2.bundle_id)

    # Advance to full rollout so treatment would normally be assigned
    for _ in range(3):
        exp.advance_stage()
    assert exp.rollout_percentage == 100

    # Trigger rollback via hidden_leak
    should_rb = exp.should_rollback(hidden_leak=True, unsafe_suggestion=False, error_rate=0.0)
    assert should_rb is True
    assert exp.auto_rollback_triggered is True

    # After rollback, all doctors should get baseline regardless of rollout
    for did in range(20):
        assigned = registry.assign_bundle(exp.experiment_id, doctor_id=did)
        assert assigned == b1.bundle_id, f"Doctor {did} should get baseline after rollback"


# ---------------------------------------------------------------------------
# 11. test_auto_rollback_error_rate — >5% → rollback
# ---------------------------------------------------------------------------

def test_auto_rollback_error_rate() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(name="b1", version="1", system_prompt="a", safety_policy="b")
    b2 = registry.register_bundle(name="b2", version="1", system_prompt="c", safety_policy="d")
    exp = registry.create_experiment(name="exp", baseline_bundle_id=b1.bundle_id, treatment_bundle_id=b2.bundle_id)

    # error_rate=0.06 (>0.05) should trigger rollback
    should_rb = exp.should_rollback(hidden_leak=False, unsafe_suggestion=False, error_rate=0.06)
    assert should_rb is True
    assert exp.auto_rollback_triggered is True

    # error_rate=0.04 (<=0.05) should NOT trigger rollback
    exp2 = registry.create_experiment(name="exp2", baseline_bundle_id=b1.bundle_id, treatment_bundle_id=b2.bundle_id)
    should_rb2 = exp2.should_rollback(hidden_leak=False, unsafe_suggestion=False, error_rate=0.04)
    assert should_rb2 is False
    assert exp2.auto_rollback_triggered is False


# ---------------------------------------------------------------------------
# 12. test_create_experiment_invalid_bundle — non-existent → KeyError
# ---------------------------------------------------------------------------

def test_create_experiment_invalid_bundle() -> None:
    registry = PromptRegistry()
    b1 = registry.register_bundle(name="b1", version="1", system_prompt="a", safety_policy="b")

    with pytest.raises(KeyError, match="nonexistent"):
        registry.create_experiment(
            name="bad-exp",
            baseline_bundle_id="nonexistent",
            treatment_bundle_id=b1.bundle_id,
        )

    with pytest.raises(KeyError, match="nonexistent"):
        registry.create_experiment(
            name="bad-exp2",
            baseline_bundle_id=b1.bundle_id,
            treatment_bundle_id="nonexistent",
        )
