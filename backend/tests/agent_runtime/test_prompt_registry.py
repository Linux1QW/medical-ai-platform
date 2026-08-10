"""Tests for prompt registry: immutable bundles, deterministic assignment, staged rollout.

Adapted for the DB-backed async API (PromptRegistry requires AsyncSession).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent_runtime.prompt_registry import (
    PromptBundleView,
    PromptRegistry,
    deterministic_assign,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db() -> MagicMock:
    """Create a mock AsyncSession."""
    db = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    return db


def _make_registry(db: MagicMock | None = None) -> PromptRegistry:
    """Create a PromptRegistry with a mocked DB session."""
    if db is None:
        db = _mock_db()
    return PromptRegistry(db)


# ---------------------------------------------------------------------------
# 1. test_deterministic_assign_stable — same subject → same result
# ---------------------------------------------------------------------------

def test_deterministic_assign_stable() -> None:
    results = [deterministic_assign(subject_id="doc-42", experiment_id="exp_001", percentage=50) for _ in range(10)]
    assert len(set(results)) == 1, "Deterministic assignment must be stable across calls"


# ---------------------------------------------------------------------------
# 2. test_deterministic_assign_0_percent — nobody assigned
# ---------------------------------------------------------------------------

def test_deterministic_assign_0_percent() -> None:
    results = [deterministic_assign(subject_id=f"doc-{did}", experiment_id="exp_zero", percentage=0) for did in range(100)]
    assert all(r is False for r in results)


# ---------------------------------------------------------------------------
# 3. test_deterministic_assign_100_percent — everyone assigned
# ---------------------------------------------------------------------------

def test_deterministic_assign_100_percent() -> None:
    results = [deterministic_assign(subject_id=f"doc-{did}", experiment_id="exp_full", percentage=100) for did in range(100)]
    assert all(r is True for r in results)


# ---------------------------------------------------------------------------
# 4. test_bundle_view_hash — same content → same hash
# ---------------------------------------------------------------------------

def test_bundle_view_hash() -> None:
    h1 = PromptBundleView.compute_hash("prompt-A")
    h2 = PromptBundleView.compute_hash("prompt-A")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# 5. test_bundle_view_hash_differs — different content → different hash
# ---------------------------------------------------------------------------

def test_bundle_view_hash_differs() -> None:
    h1 = PromptBundleView.compute_hash("prompt-A")
    h2 = PromptBundleView.compute_hash("prompt-B")
    assert h1 != h2


# ---------------------------------------------------------------------------
# 6. test_register_bundle_async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_bundle_async() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    # Mock the repo to return None (no existing bundle) and then a created bundle
    mock_bundle = MagicMock()
    mock_bundle.id = 1
    mock_bundle.name = "问诊评估v1"
    mock_bundle.version = "1.0.0"
    registry._repo = MagicMock()
    registry._repo.get_bundle = AsyncMock(return_value=None)
    registry._repo.create_bundle = AsyncMock(return_value=mock_bundle)

    bundle = await registry.register_bundle(
        name="问诊评估v1",
        version="1.0.0",
        system_prompt="你是一名资深内科医生。",
        source_commit="abc123",
        author="tester",
    )
    assert bundle.name == "问诊评估v1"
    assert bundle.version == "1.0.0"


# ---------------------------------------------------------------------------
# 7. test_register_bundle_duplicate_raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_bundle_duplicate_raises() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    existing = MagicMock()
    registry._repo = MagicMock()
    registry._repo.get_bundle = AsyncMock(return_value=existing)

    with pytest.raises(ValueError, match="already exists"):
        await registry.register_bundle(
            name="dup",
            version="1.0",
            system_prompt="sys",
            source_commit="abc",
            author="tester",
        )


# ---------------------------------------------------------------------------
# 8. test_get_bundle_async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_bundle_async() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    mock_bundle = MagicMock()
    mock_bundle.name = "test"
    mock_bundle.version = "0.1"
    registry._repo = MagicMock()
    registry._repo.get_bundle = AsyncMock(return_value=mock_bundle)

    retrieved = await registry.get_bundle("test", "0.1")
    assert retrieved is not None
    assert retrieved.name == "test"


# ---------------------------------------------------------------------------
# 9. test_get_bundle_not_found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_bundle_not_found() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    registry._repo = MagicMock()
    registry._repo.get_bundle = AsyncMock(return_value=None)

    result = await registry.get_bundle("nonexistent", "9.9")
    assert result is None


# ---------------------------------------------------------------------------
# 10. test_assign_bundle_returns_existing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_bundle_returns_existing() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    existing_assignment = MagicMock()
    existing_assignment.variant = "baseline-v1"
    registry._repo = MagicMock()
    registry._repo.get_experiment_variant = AsyncMock(return_value=existing_assignment)

    result = await registry.assign_bundle(
        experiment_id="exp_001",
        subject_id="doc-42",
        baseline_name="baseline-v1",
        treatment_name="treatment-v1",
        rollout_pct=50,
    )
    assert result == "baseline-v1"


# ---------------------------------------------------------------------------
# 11. test_assign_bundle_deterministic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_bundle_deterministic() -> None:
    db = _mock_db()
    registry = PromptRegistry(db)

    mock_assignment = MagicMock()
    mock_assignment.variant = "treatment-v1"
    registry._repo = MagicMock()
    registry._repo.get_experiment_variant = AsyncMock(return_value=None)
    registry._repo.assign_experiment = AsyncMock(return_value=mock_assignment)

    result = await registry.assign_bundle(
        experiment_id="exp_001",
        subject_id="doc-42",
        baseline_name="baseline-v1",
        treatment_name="treatment-v1",
        rollout_pct=100,
    )
    assert result == "treatment-v1"
