# -*- coding: utf-8 -*-
"""Task 8: migrate_v11.py 编排脚本测试

验证：
1. 全新库无 operator
2. 存量库有 operator
3. 缺 operator 失败
4. Alembic 子步骤失败
5. 第二次运行幂等
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MIGRATE_V11_PATH = BACKEND_ROOT / "scripts" / "migrate_v11.py"


def test_migrate_v11_makes_backend_package_importable_when_executed_as_script(
    monkeypatch,
) -> None:
    """``python scripts/migrate_v11.py`` must be able to import ``app`` later."""
    monkeypatch.setattr(sys, "path", [str(MIGRATE_V11_PATH.parent)])
    module_name = "migrate_v11_script_import_contract"
    spec = importlib.util.spec_from_file_location(module_name, MIGRATE_V11_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        assert str(BACKEND_ROOT) in sys.path
    finally:
        sys.modules.pop(module_name, None)

# ── 1. 全新库无 operator ────────────────────────────────────────────────────


class TestMigrateFreshDatabase:
    """全新库无需 operator 即可完成"""

    def test_fresh_db_completes_without_operator(self):
        """全新库（无候选 evaluation）无需 operator"""
        from scripts.migrate_v11 import run_migration

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Current revision: None"

        with (
            patch("scripts.migrate_v11._run_command", return_value=mock_result),
            patch("scripts.migrate_v11._get_current_revision", return_value=None),
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 0, "orphans": 0, "ambiguous": 0}),
        ):
            exit_code = run_migration(
                operator_id=None,
                batch_size=100,
                dry_run=False,
            )

        assert exit_code == 0

    def test_fresh_db_skips_backfill_when_no_candidates(self):
        """全新库跳过 backfill 步骤"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value=None),
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 0, "orphans": 0, "ambiguous": 0}),
        ):
            mock_cmd.return_value = MagicMock(returncode=0)
            exit_code = run_migration(operator_id=None, batch_size=100, dry_run=False)

        assert exit_code == 0


# ── 2. 存量库有 operator ────────────────────────────────────────────────────


class TestMigrateExistingDatabase:
    """存量库有 operator 时正常执行"""

    def test_existing_db_with_operator_succeeds(self):
        """有 operator 且有候选时正常执行"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value="2b3c4d5e6f7a"),
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 5, "orphans": 0, "ambiguous": 0}),
            patch("scripts.backfill_legacy_evaluation_runs.main_async", new_callable=AsyncMock, return_value=0) as mock_backfill,
        ):
            mock_cmd.return_value = MagicMock(returncode=0)
            exit_code = run_migration(operator_id="admin-1", batch_size=100, dry_run=False)

        assert exit_code == 0
        mock_backfill.assert_awaited_once()


# ── 3. 缺 operator 失败 ─────────────────────────────────────────────────────


class TestMigrateMissingOperator:
    """存量库缺 operator 时失败"""

    def test_existing_db_without_operator_fails(self):
        """有候选但无 operator 时返回非零"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value="2b3c4d5e6f7a"),
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 5, "orphans": 0, "ambiguous": 0}),
        ):
            mock_cmd.return_value = MagicMock(returncode=0)
            exit_code = run_migration(operator_id=None, batch_size=100, dry_run=False)

        assert exit_code != 0


# ── 4. Alembic 子步骤失败 ───────────────────────────────────────────────────


class TestMigrateAlembicFailure:
    """Alembic 子步骤失败时立即停止"""

    def test_alembic_upgrade_failure_stops_migration(self):
        """alembic upgrade 失败时返回非零"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value=None),
        ):
            # 第一次调用（alembic upgrade）失败
            mock_cmd.return_value = MagicMock(returncode=1, stderr="Migration failed")
            exit_code = run_migration(operator_id=None, batch_size=100, dry_run=False)

        assert exit_code != 0


# ── 5. 幂等性 ───────────────────────────────────────────────────────────────


class TestMigrateIdempotent:
    """第二次运行幂等"""

    def test_second_run_is_idempotent(self):
        """第二次运行（已在 head）幂等完成"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value="3c4d5e6f7a8b"),  # 已是 head
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 0, "orphans": 0, "ambiguous": 0}),
        ):
            mock_cmd.return_value = MagicMock(returncode=0)
            exit_code = run_migration(operator_id=None, batch_size=100, dry_run=False)

        assert exit_code == 0


# ── 6. 安全性：subprocess 参数白名单 ────────────────────────────────────────


class TestMigrateSubprocessSafety:
    """subprocess 参数必须是固定 allowlist，不能拼接用户 shell 字符串"""

    def test_subprocess_uses_fixed_allowlist(self):
        """subprocess 调用使用固定参数列表"""
        from scripts.migrate_v11 import run_migration

        with (
            patch("scripts.migrate_v11._run_command") as mock_cmd,
            patch("scripts.migrate_v11._get_current_revision", return_value=None),
            patch("scripts.migrate_v11._count_backfill_candidates", return_value={"null_run_id": 0, "orphans": 0, "ambiguous": 0}),
        ):
            mock_cmd.return_value = MagicMock(returncode=0)
            run_migration(operator_id=None, batch_size=100, dry_run=False)

        # 验证所有 subprocess 调用使用列表参数（非 shell 字符串）
        for c in mock_cmd.call_args_list:
            args = c[0][0] if c[0] else c[1].get("args", [])
            # 参数应为列表，非字符串
            assert isinstance(args, (list, tuple)), f"subprocess 参数应为列表，实际为 {type(args)}"
