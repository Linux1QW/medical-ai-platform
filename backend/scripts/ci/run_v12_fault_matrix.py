"""Run observed V1.2 fault probes against an isolated Compose environment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FAULTS = (
    "dispatcher_outage_recovery",
    "redis_state_fail_closed",
    "redis_cache_saturation_degrades_only",
    "execution_owner_fencing",
)
PROJECT_NAME = "medical-ai-v12-e2e"
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (REPO_ROOT / "docker-compose.yml", REPO_ROOT / "docker-compose.e2e.yml")
BACKEND_URL = os.environ.get("V12_BACKEND_A", "http://127.0.0.1:8000")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
MYSQL_USER = os.environ.get("MYSQL_USER", "medical_e2e")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "e2e-medical-password")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "medical_ai_e2e")


@dataclass
class FaultResult:
    passed: bool
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def compose(project: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run Docker Compose from the repository root regardless of caller cwd."""
    command = ["docker", "compose", "-p", project]
    for compose_file in COMPOSE_FILES:
        command.extend(["-f", str(compose_file)])
    command.extend(args)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _http_json(
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{BACKEND_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload
    except URLError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {path}: {exc}") from exc


def _login(username: str) -> str:
    status, payload = _http_json(
        "POST",
        "/api/v1/auth/login",
        body={"username": username, "password": E2E_PASSWORD},
    )
    _require(status == 200, f"login for {username} returned {status}: {payload}")
    token = payload.get("access_token") if isinstance(payload, dict) else None
    _require(bool(token), f"login for {username} returned no access token")
    return str(token)


def _mysql_scalar(project: str, query: str) -> str:
    result = compose(
        project,
        "exec",
        "-T",
        "mysql",
        "mysql",
        f"-u{MYSQL_USER}",
        f"-p{MYSQL_PASSWORD}",
        f"-D{MYSQL_DATABASE}",
        "-Nse",
        query,
    )
    return result.stdout.strip()


def _wait_until(predicate, *, timeout: float, interval: float = 1.0, description: str) -> Any:
    deadline = time.monotonic() + timeout
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {description}; last={last_value!r}")


def run_fault_matrix(project: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for fault in FAULTS:
        try:
            results[fault] = asdict(_run_single_fault(project, fault))
        except Exception as exc:
            results[fault] = {
                "passed": False,
                "message": f"{type(exc).__name__}: {exc}",
                "evidence": {},
            }
    return results


def _run_single_fault(project: str, fault: str) -> FaultResult:
    handlers = {
        "dispatcher_outage_recovery": _test_dispatcher_outage,
        "redis_state_fail_closed": _test_redis_state_fail_closed,
        "redis_cache_saturation_degrades_only": _test_redis_cache_saturation,
        "execution_owner_fencing": _test_execution_owner_fencing,
    }
    handler = handlers.get(fault)
    if handler is None:
        return FaultResult(False, f"Unknown fault: {fault}")
    return handler(project)


def _test_dispatcher_outage(project: str) -> FaultResult:
    token = _login("doctor_v11")
    compose(project, "stop", "evaluation-dispatcher")
    run_id = ""
    try:
        status, payload = _http_json(
            "POST",
            "/api/v1/evaluations/",
            token=token,
            body={"consultation_id": 2},
        )
        _require(status == 202, f"evaluation submission returned {status}: {payload}")
        run_id = str(payload.get("run_id", ""))
        _require(bool(run_id), "evaluation submission returned no run_id")
        outbox_before = _mysql_scalar(
            project,
            f"SELECT status FROM evaluation_dispatch_outbox WHERE run_id='{run_id}'",
        )
        _require(outbox_before == "pending", f"outbox was {outbox_before!r}, expected pending")
    finally:
        compose(project, "start", "evaluation-dispatcher", check=False)

    _wait_until(
        lambda: _mysql_scalar(
            project,
            f"SELECT status FROM evaluation_dispatch_outbox WHERE run_id='{run_id}'",
        )
        == "published",
        timeout=30,
        description="outbox publication after dispatcher recovery",
    )

    def terminal_status() -> str | None:
        status, payload = _http_json(
            "GET", f"/api/v1/evaluations/runs/{run_id}/status", token=token
        )
        if status != 200 or not isinstance(payload, dict):
            return None
        value = str(payload.get("status", ""))
        return value if value in {"completed", "needs_review", "reviewed", "failed"} else None

    final_status = _wait_until(
        terminal_status,
        timeout=120,
        interval=2,
        description="evaluation terminal status after dispatcher recovery",
    )
    _require(final_status != "failed", "evaluation reached failed after dispatcher recovery")
    return FaultResult(
        True,
        "dispatcher outage accepted via outbox and recovered",
        {"run_id": run_id, "outbox_before": "pending", "outbox_after": "published", "final_status": final_status},
    )


def _wait_redis(project: str) -> bool:
    result = compose(project, "exec", "-T", "redis-state", "redis-cli", "ping", check=False)
    return result.returncode == 0 and "PONG" in result.stdout


def _test_redis_state_fail_closed(project: str) -> FaultResult:
    token = _login("doctor_v12")
    compose(project, "stop", "redis-state")
    unavailable_status = 0
    try:
        unavailable_status, _ = _http_json("GET", "/api/v1/auth/me", token=token)
        _require(unavailable_status == 503, f"protected API returned {unavailable_status}, expected 503")
    finally:
        compose(project, "start", "redis-state", check=False)

    _wait_until(
        lambda: _wait_redis(project),
        timeout=30,
        description="redis-state recovery",
    )
    recovered_status = _wait_until(
        lambda: (lambda response: response[0] if response[0] == 200 else None)(
            _http_json("GET", "/api/v1/auth/me", token=token)
        ),
        timeout=30,
        description="protected API recovery",
    )
    return FaultResult(
        True,
        "redis-state failed closed and recovered",
        {"down_status": unavailable_status, "recovered_status": recovered_status},
    )


def _redis_config(project: str, key: str) -> str:
    result = compose(
        project, "exec", "-T", "redis-cache", "redis-cli", "--raw", "CONFIG", "GET", key
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    _require(len(lines) >= 2, f"unexpected CONFIG GET {key} output: {result.stdout!r}")
    return lines[-1]


def _test_redis_cache_saturation(project: str) -> FaultResult:
    token = _login("doctor_v12")
    old_maxmemory = _redis_config(project, "maxmemory")
    old_policy = _redis_config(project, "maxmemory-policy")
    oom_output = ""
    try:
        compose(project, "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory-policy", "noeviction")
        compose(project, "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory", "1")
        write = compose(
            project,
            "exec",
            "-T",
            "redis-cache",
            "redis-cli",
            "SET",
            "fault:cache-saturation",
            "payload",
            check=False,
        )
        oom_output = f"{write.stdout}\n{write.stderr}".strip()
        _require("OOM" in oom_output.upper(), f"cache write did not produce OOM: {oom_output}")
        health_status, _ = _http_json("GET", "/health")
        auth_status, _ = _http_json("GET", "/api/v1/auth/me", token=token)
        _require(health_status == 200, f"core health returned {health_status}")
        _require(auth_status == 200, f"state-backed auth returned {auth_status}")
    finally:
        compose(project, "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory", old_maxmemory, check=False)
        compose(project, "exec", "-T", "redis-cache", "redis-cli", "CONFIG", "SET", "maxmemory-policy", old_policy, check=False)
        compose(project, "exec", "-T", "redis-cache", "redis-cli", "DEL", "fault:cache-saturation", check=False)
    return FaultResult(
        True,
        "cache OOM degraded without affecting core state/auth",
        {"oom_observed": True, "health_status": 200, "auth_status": 200},
    )


def _test_execution_owner_fencing(project: str) -> FaultResult:
    result = compose(
        project,
        "exec",
        "-T",
        "backend",
        "python",
        "scripts/ci/v12_fencing_probe.py",
    )
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    _require(bool(lines), f"fencing probe emitted no JSON: {result.stdout} {result.stderr}")
    evidence = json.loads(lines[-1])
    _require(evidence.get("first_claim") == "started", f"unexpected first claim: {evidence}")
    _require(evidence.get("active_other") == "active_other_task", f"active owner was not protected: {evidence}")
    _require(evidence.get("reclaimed") == "reclaimed_stale", f"stale lease was not reclaimed: {evidence}")
    _require(evidence.get("stale_owner_rejected") is True, f"stale owner write was accepted: {evidence}")
    return FaultResult(True, "execution-owner fencing verified", evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V1.2 observed fault matrix")
    parser.add_argument("--project", default=PROJECT_NAME)
    parser.add_argument("--output", default="artifacts/fault-matrix.json")
    args = parser.parse_args()
    results = run_fault_matrix(args.project)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for result in results.values() if result["passed"])
    print(f"Fault matrix: {passed}/{len(results)} passed")
    for name, result in results.items():
        print(f"  {'PASS' if result['passed'] else 'FAIL'}: {name}: {result['message']}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
