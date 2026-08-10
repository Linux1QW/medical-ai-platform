"""Tests for V1.2 runtime remediation migration (5e6f7a8b9c0d).

Verifies migration chain integrity and that the migration module
exposes correct upgrade/downgrade callables.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ALEMBIC_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _ensure_mocks():
    """Ensure alembic.versions can be resolved for migration imports."""
    if "alembic.versions" not in sys.modules:
        versions_mod = types.ModuleType("alembic.versions")
        versions_mod.__path__ = [str(ALEMBIC_VERSIONS_DIR)]
        sys.modules["alembic.versions"] = versions_mod
    if "alembic.op" not in sys.modules:
        sys.modules["alembic.op"] = MagicMock()


_ensure_mocks()


def _load_migration(name: str):
    """Load a migration module by file path."""
    filepath = ALEMBIC_VERSIONS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"alembic.versions.{name}", filepath)
    assert spec is not None, f"Cannot find migration file: {filepath}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"alembic.versions.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


MIGRATION_NAME = "5e6f7a8b9c0d_v12_runtime_remediation"


def test_migration_module_importable():
    mod = _load_migration(MIGRATION_NAME)
    assert mod is not None


def test_migration_revision_ids():
    mod = _load_migration(MIGRATION_NAME)
    assert mod.revision == "5e6f7a8b9c0d"
    assert mod.down_revision == "4d5e6f7a8b9c"


def test_migration_has_upgrade_and_downgrade():
    mod = _load_migration(MIGRATION_NAME)
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)


def test_migration_chain_continuity():
    """Verify the previous migration in the chain exists."""
    prev_mod = _load_migration("4d5e6f7a8b9c_v12_agent_runtime")
    assert prev_mod.revision == "4d5e6f7a8b9c"


def test_new_tables_defined_in_migration():
    """Smoke-test that upgrade() references the new tables by reading source."""
    import inspect
    mod = _load_migration(MIGRATION_NAME)
    source = inspect.getsource(mod.upgrade)
    assert "coach_stream_events" in source
    assert "trainee_memory_consents" in source


def test_downgrade_removes_remediation_additions():
    """Smoke-test that downgrade() references removal of new objects."""
    import inspect
    mod = _load_migration(MIGRATION_NAME)
    source = inspect.getsource(mod.downgrade)
    assert "coach_stream_events" in source
    assert "trainee_memory_consents" in source
    assert "public_id" in source
    assert "suggestion_id" in source
