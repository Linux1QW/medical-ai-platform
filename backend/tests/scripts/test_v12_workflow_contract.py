"""Static contracts preventing false-green V1.2 CI/RC orchestration."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CI = ROOT / ".github" / "workflows" / "ci.yml"
RC = ROOT / ".github" / "workflows" / "v12-rc.yml"


def _job(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    assert match is not None, f"workflow job {name!r} not found"
    return match.group(0)


def test_ci_playwright_job_is_self_contained_and_targets_real_base_url() -> None:
    job = _job(CI.read_text(encoding="utf-8"), "playwright-e2e")

    assert "E2E_BASE_URL: http://localhost:5173" in job
    assert "PLAYWRIGHT_TEST_BASE_URL" not in job
    assert "alembic upgrade head" in job
    assert "tests.fixtures.seed_production_smoke" in job
    assert "tests.fixtures.seed_v12_coach_e2e" in job
    assert job.count("npx playwright test") == 1
    assert "PLAYWRIGHT_JSON_OUTPUT_NAME: playwright-results.json" in job
    assert '"status":"completed"' not in job
    assert "tests.fixtures.mock_openai_server" in job
    assert job.count("COACH_ENABLED: 'true'") >= 2
    assert "--port 8000" in job
    assert "--port 8001" in job
    assert "curl -sf http://localhost:8001/health" in job


def test_ci_dependency_and_integration_gates_are_fail_closed() -> None:
    workflow = CI.read_text(encoding="utf-8")
    assert "pip check" in workflow
    assert "npm audit --omit=dev --audit-level=high" in workflow
    assert "MYSQL_INTEGRATION_DATABASE_URL:" in workflow


def test_rc_browser_job_migrates_seeds_and_starts_two_backends() -> None:
    job = _job(RC.read_text(encoding="utf-8"), "browser-e2e")

    assert "E2E_BASE_URL: http://127.0.0.1:5173" in job
    assert "PLAYWRIGHT_TEST_BASE_URL" not in job
    assert "alembic upgrade head" in job
    assert "tests.fixtures.seed_production_smoke" in job
    assert "tests.fixtures.seed_v12_coach_e2e" in job
    assert job.count("COACH_ENABLED: 'true'") >= 2
    assert "--port 8000" in job
    assert "--port 8001" in job
    assert "curl -sf http://127.0.0.1:8001/health" in job


def test_rc_dual_load_initializes_its_own_database() -> None:
    job = _job(RC.read_text(encoding="utf-8"), "dual-load")

    assert "alembic upgrade head" in job
    assert "tests.fixtures.seed_v12_coach_e2e" in job


def test_rc_fault_matrix_runs_against_compose_environment() -> None:
    job = _job(RC.read_text(encoding="utf-8"), "disconnect-replay")

    assert "docker compose" in job
    assert "docker-compose.e2e.yml" in job
    assert "run_v12_fault_matrix.py" in job
    assert "down -v" in job


def test_rc_job_ids_and_needs_expressions_are_github_compatible() -> None:
    workflow = RC.read_text(encoding="utf-8")

    assert "  coach-72-case-live:" in workflow
    assert "  72-case-live:" not in workflow
    assert "needs.72-case-live" not in workflow
    assert "needs['coach-72-case-live'].result" in workflow
    assert not re.search(r"needs\.[A-Za-z0-9_]+-", workflow)


def test_rc_database_seed_is_fail_closed_and_verified() -> None:
    workflow = RC.read_text(encoding="utf-8")
    setup = _job(workflow, "setup-services")
    migration = _job(workflow, "db-migrate-seed")

    for job in (setup, migration):
        assert "seed.sql || true" not in job
        assert "seed_patients.py || true" not in job
        assert "SELECT COUNT(*) FROM users" in job
        assert "SELECT COUNT(*) FROM virtual_patients" in job
