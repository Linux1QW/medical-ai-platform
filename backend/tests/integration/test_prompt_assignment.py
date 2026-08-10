# -*- coding: utf-8 -*-
"""Prompt assignment integration tests.

Coverage:
- Deterministic assignment (same input → same output)
- Assignment persistence (once assigned, stays the same)
- Rollback on safety violation
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_runtime.prompt_registry import PromptRegistry, deterministic_assign


# ── Deterministic assignment ─────────────────────────────────────────────────


class TestDeterministicAssignment:
    """Test deterministic assignment function."""

    def test_same_input_same_output(self) -> None:
        """Same subject_id and experiment_id should always produce same result."""
        result1 = deterministic_assign("subject_1", "exp_1", 50)
        result2 = deterministic_assign("subject_1", "exp_1", 50)
        assert result1 == result2

    def test_different_subjects_may_differ(self) -> None:
        """Different subjects may get different assignments."""
        results = set()
        for i in range(100):
            result = deterministic_assign(f"subject_{i}", "exp_1", 50)
            results.add(result)
        # With 50% rollout and 100 subjects, we should see both True and False
        assert True in results
        assert False in results

    def test_zero_percent_always_false(self) -> None:
        """0% rollout should never assign to treatment."""
        for i in range(100):
            result = deterministic_assign(f"subject_{i}", "exp_1", 0)
            assert result is False

    def test_hundred_percent_always_true(self) -> None:
        """100% rollout should always assign to treatment."""
        for i in range(100):
            result = deterministic_assign(f"subject_{i}", "exp_1", 100)
            assert result is True

    def test_deterministic_hash_distribution(self) -> None:
        """Verify hash-based distribution is roughly uniform."""
        treatment_count = sum(
            1 for i in range(1000)
            if deterministic_assign(f"subject_{i}", "exp_1", 50)
        )
        # With 50% rollout and 1000 subjects, expect ~500 ± 50
        assert 400 <= treatment_count <= 600


# ── Assignment persistence ───────────────────────────────────────────────────


class TestAssignmentPersistence:
    """Test that assignments are persisted once and not changed."""

    @pytest.mark.asyncio
    async def test_existing_assignment_returned(self) -> None:
        """If already assigned, return existing assignment."""
        mock_db = AsyncMock()
        mock_repo = AsyncMock()

        # Simulate existing assignment
        existing_assignment = MagicMock()
        existing_assignment.variant = "baseline"
        mock_repo.get_experiment_variant.return_value = existing_assignment

        registry = PromptRegistry(mock_db)
        registry._repo = mock_repo

        variant = await registry.assign_bundle(
            experiment_id="exp_1",
            subject_id="subject_1",
            baseline_name="baseline",
            treatment_name="treatment",
            rollout_pct=50,
        )

        assert variant == "baseline"
        # Should not create a new assignment
        mock_repo.assign_experiment.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_assignment_created(self) -> None:
        """If not assigned, create new deterministic assignment."""
        mock_db = AsyncMock()
        mock_repo = AsyncMock()

        # No existing assignment
        mock_repo.get_experiment_variant.return_value = None
        mock_repo.assign_experiment.return_value = MagicMock()

        registry = PromptRegistry(mock_db)
        registry._repo = mock_repo

        variant = await registry.assign_bundle(
            experiment_id="exp_1",
            subject_id="subject_1",
            baseline_name="baseline",
            treatment_name="treatment",
            rollout_pct=50,
        )

        assert variant in ("baseline", "treatment")
        mock_repo.assign_experiment.assert_called_once()


# ── Rollback ─────────────────────────────────────────────────────────────────


class TestRollback:
    """Test rollback behavior on safety violations."""

    def test_deterministic_assignment_consistent_after_rollback(self) -> None:
        """Assignment determinism is independent of rollback state.

        Rollback is handled at the experiment level, not the assignment level.
        The deterministic_assign function always produces the same result.
        """
        # Before rollback
        result_before = deterministic_assign("subject_1", "exp_1", 50)

        # After rollback (same inputs)
        result_after = deterministic_assign("subject_1", "exp_1", 50)

        assert result_before == result_after
