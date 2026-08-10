"""Test that V1.1 Alembic migrations generate valid offline SQL."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _run_alembic_sql() -> subprocess.CompletedProcess:
    """Run alembic upgrade head --sql and return the result."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=BACKEND_ROOT,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def test_offline_upgrade_produces_outbox_ddl():
    """alembic upgrade head --sql must include CREATE TABLE for outbox."""
    result = _run_alembic_sql()
    assert result.returncode == 0, result.stderr
    assert "CREATE TABLE evaluation_dispatch_outbox" in result.stdout


def test_offline_upgrade_produces_review_indexes():
    """alembic upgrade head --sql must include review/audit index creation."""
    result = _run_alembic_sql()
    assert result.returncode == 0, result.stderr
    # 检查 review 相关的索引创建
    output = result.stdout.upper()
    assert "CREATE INDEX" in output or "ADD CONSTRAINT" in output


def test_offline_upgrade_does_not_execute_data_checks():
    """alembic upgrade head --sql must NOT attempt data queries (offline mode)."""
    result = _run_alembic_sql()
    assert result.returncode == 0, result.stderr
    # 离线模式不应包含 SELECT 查询（数据检查被 is_offline_mode 保护）
    # 注意：DDL 中可能包含 SELECT 用于 literal_binds，但不应包含 evaluations 表的查询
    stdout_lower = result.stdout.lower()
    # 确保不包含 duplicate check 或 orphan check 的查询
    assert "having cnt > 1" not in stdout_lower, (
        "Offline SQL should not contain duplicate run_id check"
    )
    assert "left join evaluation_runs" not in stdout_lower, (
        "Offline SQL should not contain orphan run_id check"
    )


def test_offline_upgrade_includes_outbox_indexes():
    """alembic upgrade head --sql must include outbox index creation."""
    result = _run_alembic_sql()
    assert result.returncode == 0, result.stderr
    output = result.stdout
    # 检查 outbox 相关索引
    assert "ix_dispatch_outbox_status_next_attempt" in output
    assert "ix_dispatch_outbox_lease_expires" in output
    assert "ix_dispatch_outbox_created_at" in output


def test_offline_upgrade_includes_evaluation_run_id_unique_index():
    """alembic upgrade head --sql must include unique index on evaluations.run_id."""
    result = _run_alembic_sql()
    assert result.returncode == 0, result.stderr
    assert "ux_evaluations_run_id" in result.stdout
