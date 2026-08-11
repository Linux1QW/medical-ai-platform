"""Tests for V1.2 fault matrix controller."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

from run_v12_fault_matrix import FAULTS, FaultResult, run_fault_matrix


def test_fault_matrix_defines_four_scenarios():
    """Verify that exactly four fault scenarios are defined."""
    assert len(FAULTS) == 4
    assert "dispatcher_outage_recovery" in FAULTS
    assert "redis_state_fail_closed" in FAULTS
    assert "redis_cache_saturation_degrades_only" in FAULTS
    assert "execution_owner_fencing" in FAULTS


@patch("run_v12_fault_matrix._run_single_fault")
def test_run_fault_matrix_returns_all_results(mock_single):
    """Verify that run_fault_matrix returns results for all scenarios."""
    mock_single.return_value = FaultResult(passed=True, message="ok")
    results = run_fault_matrix("test-project")
    assert len(results) == 4
    assert all(r["passed"] for r in results.values())


@patch("run_v12_fault_matrix._run_single_fault")
def test_run_fault_matrix_handles_failure(mock_single):
    """Verify that failures are captured correctly."""
    mock_single.return_value = FaultResult(passed=False, message="simulated failure")
    results = run_fault_matrix("test-project")
    assert len(results) == 4
    assert all(not r["passed"] for r in results.values())
    assert all(r["message"] == "simulated failure" for r in results.values())


@patch("run_v12_fault_matrix._run_single_fault")
def test_run_fault_matrix_mixed_results(mock_single):
    """Verify mixed pass/fail results."""
    call_count = 0

    def side_effect(project, fault):
        nonlocal call_count
        call_count += 1
        if fault == "dispatcher_outage_recovery":
            return FaultResult(passed=True, message="ok")
        return FaultResult(passed=False, message="failed")

    mock_single.side_effect = side_effect
    results = run_fault_matrix("test-project")
    assert results["dispatcher_outage_recovery"]["passed"] is True
    assert results["redis_state_fail_closed"]["passed"] is False
