"""Tests for V1.2 runtime remediation migration (5e6f7a8b9c0d).

Verifies migration chain integrity and that the migration module
exposes correct upgrade/downgrade callables.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock


def _ensure_alembic_mocks():
    """Ensure alembic.op and alembic modules are mockable for import."""
    if "alembic.op" not in sys.modules:
        mock_op = MagicMock()
        sys.modules["alembic.op"] = mock_op


_ensure_alembic_mocks()

MIGRATION_MODULE = "alembic.versions.5e6f7a8b9c0d_v12_runtime_remediation"


def test_migration_module_importable():
    mod = importlib.import_module(MIGRATION_MODULE)
    assert mod is not None


def test_migration_revision_ids():
    mod = importlib.import_module(MIGRATION_MODULE)
    assert mod.revision == "5e6f7a8b9c0d"
    assert mod.down_revision == "4d5e6f7a8b9c"


def test_migration_has_upgrade_and_downgrade():
    mod = importlib.import_module(MIGRATION_MODULE)
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)


def test_migration_chain_continuity():
    """Verify the previous migration in the chain exists."""
    prev_mod = importlib.import_module(
        "alembic.versions.4d5e6f7a8b9c_v12_agent_runtime"
    )
    assert prev_mod.revision == "4d5e6f7a8b9c"


def test_new_tables_defined_in_migration():
    """Smoke-test that upgrade() references the new tables by reading source."""
    import inspect
    mod = importlib.import_module(MIGRATION_MODULE)
    source = inspect.getsource(mod.upgrade)
    assert "coach_stream_events" in source
    assert "trainee_memory_consents" in source


def test_downgrade_removes_remediation_additions():
    """Smoke-test that downgrade() references removal of new objects."""
    import inspect
    mod = importlib.import_module(MIGRATION_MODULE)
    source = inspect.getsource(mod.downgrade)
    assert "coach_stream_events" in source
    assert "trainee_memory_consents" in source
    assert "public_id" in source
    assert "suggestion_id" in source
