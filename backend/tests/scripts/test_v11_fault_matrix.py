"""Tests for V1.2 fault matrix controller."""
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

import run_v12_fault_matrix as fault_matrix
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


@patch("run_v12_fault_matrix.subprocess.run")
def test_compose_uses_absolute_files_and_repository_cwd(mock_run):
    mock_run.return_value = CompletedProcess([], 0, "ok", "")

    fault_matrix.compose("probe", "ps")

    command = mock_run.call_args.args[0]
    assert command[:4] == ["docker", "compose", "-p", "probe"]
    assert all(Path(command[index + 1]).is_absolute() for index, value in enumerate(command) if value == "-f")
    assert mock_run.call_args.kwargs["cwd"] == fault_matrix.REPO_ROOT


@patch("run_v12_fault_matrix._wait_until")
@patch("run_v12_fault_matrix._mysql_scalar")
@patch("run_v12_fault_matrix._http_json")
@patch("run_v12_fault_matrix._login", return_value="token")
@patch("run_v12_fault_matrix.compose")
def test_dispatcher_outage_is_observed_and_recovers(
    mock_compose, _mock_login, mock_http, mock_mysql, mock_wait
):
    mock_http.return_value = (202, {"run_id": "run-1"})
    mock_mysql.return_value = "pending"
    mock_wait.side_effect = [True, "completed"]

    result = fault_matrix._test_dispatcher_outage("probe")

    assert result.passed is True
    assert result.evidence["outbox_before"] == "pending"
    assert result.evidence["final_status"] == "completed"
    assert call("probe", "stop", "evaluation-dispatcher") in mock_compose.call_args_list
    assert call("probe", "start", "evaluation-dispatcher", check=False) in mock_compose.call_args_list


@patch("run_v12_fault_matrix._wait_until", side_effect=[True, 200])
@patch("run_v12_fault_matrix._http_json", return_value=(503, {"detail": "unavailable"}))
@patch("run_v12_fault_matrix._login", return_value="token")
@patch("run_v12_fault_matrix.compose")
def test_redis_state_probe_requires_fail_closed_and_restart(
    mock_compose, _mock_login, _mock_http, _mock_wait
):
    result = fault_matrix._test_redis_state_fail_closed("probe")

    assert result.evidence == {"down_status": 503, "recovered_status": 200}
    assert call("probe", "stop", "redis-state") in mock_compose.call_args_list
    assert call("probe", "start", "redis-state", check=False) in mock_compose.call_args_list


@patch("run_v12_fault_matrix._http_json", side_effect=[(200, {}), (200, {})])
@patch("run_v12_fault_matrix._redis_config", side_effect=["0", "allkeys-lru"])
@patch("run_v12_fault_matrix._login", return_value="token")
@patch("run_v12_fault_matrix.compose")
def test_cache_saturation_restores_configuration(
    mock_compose, _mock_login, _mock_config, _mock_http
):
    mock_compose.return_value = CompletedProcess([], 0, "OOM command not allowed", "")

    result = fault_matrix._test_redis_cache_saturation("probe")

    assert result.passed is True
    assert call(
        "probe", "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory", "0", check=False
    ) in mock_compose.call_args_list
    assert call(
        "probe", "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory-policy", "allkeys-lru", check=False
    ) in mock_compose.call_args_list


@patch("run_v12_fault_matrix.compose")
def test_execution_owner_fencing_requires_all_invariants(mock_compose):
    evidence = {
        "first_claim": "started",
        "active_other": "active_other_task",
        "reclaimed": "reclaimed_stale",
        "stale_owner_rejected": True,
    }
    mock_compose.return_value = CompletedProcess([], 0, f"log\n{fault_matrix.json.dumps(evidence)}\n", "")

    result = fault_matrix._test_execution_owner_fencing("probe")

    assert result.passed is True
    assert result.evidence == evidence
