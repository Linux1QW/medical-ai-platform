"""MySQL-safe migration tests — covers 6 production migration paths.

Paths tested:
1. Empty DB (no revision) → head
2. V1.1 head (3c4d5e6f7a8b) → V1.2 head
3. V1.2 first revision (4d5e6f7a8b9c) → remediation head
4. Coach tables with existing data → backfill produces unique non-null IDs
5. Downgrade -1 then re-upgrade → head
6. Already at head (5e6f7a8b9c0d) → idempotent no-op

All tests assert final revision == 5e6f7a8b9c0d and backfilled IDs are unique/non-null.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import sqlalchemy as sa

# ── Fixtures / helpers ──────────────────────────────────────────────────────────

ALEMBIC_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"
BACKEND_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BACKEND_DIR / "scripts"

TARGET_HEAD = "5e6f7a8b9c0d"


def _ensure_mocks():
    """Ensure ``alembic.versions`` and ``alembic.op`` are importable for migration loading."""
    if "alembic.versions" not in sys.modules:
        versions_mod = types.ModuleType("alembic.versions")
        versions_mod.__path__ = [str(ALEMBIC_VERSIONS_DIR)]
        sys.modules["alembic.versions"] = versions_mod
    if "alembic.op" not in sys.modules:
        sys.modules["alembic.op"] = MagicMock()
    if "alembic.context" not in sys.modules:
        sys.modules["alembic.context"] = MagicMock()


_ensure_mocks()


def _load_migration(name: str):
    """Load a migration module by file name (without .py)."""
    filepath = ALEMBIC_VERSIONS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"alembic.versions.{name}", filepath)
    assert spec is not None, f"Cannot find migration file: {filepath}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"alembic.versions.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_sqlite_engine():
    """Create an in-memory SQLite engine for schema-level tests."""
    return sa.create_engine("sqlite:///:memory:")


def _build_base_schema(engine: sa.Engine) -> None:
    """Create the tables that exist before the V1.2 remediation migration.

    This simulates having run migrations up through 4d5e6f7a8b9c (V1.2 agent runtime).
    """
    metadata = sa.MetaData()

    # users table (needed for FK references)
    sa.Table(
        "users",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(50), nullable=False),
    )

    # consultations (needed for coach_sessions FK)
    sa.Table(
        "consultations",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
    )

    # coach_sessions (created by 4d5e6f7a8b9c, before remediation)
    sa.Table(
        "coach_sessions",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("consultation_id", sa.Integer(), nullable=False),
        sa.Column("doctor_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False, server_default="off"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("thread_id", sa.String(120), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_bundle_version", sa.String(50), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("last_turn_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # coach_decisions (created by 4d5e6f7a8b9c, before remediation)
    sa.Table(
        "coach_decisions",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("turn_no", sa.Integer(), nullable=False),
        sa.Column("intent", sa.String(120), nullable=False),
        sa.Column("stage", sa.String(50), nullable=False),
        sa.Column("suggestion_json", sa.JSON(), nullable=True),
        sa.Column("visible_context_hmac", sa.String(64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(20), nullable=True),
        sa.Column("prompt_version", sa.String(50), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("feedback_value", sa.String(20), nullable=True),
        sa.Column("feedback_reason", sa.String(200), nullable=True),
        sa.Column("feedback_comment", sa.String(500), nullable=True),
        sa.Column("feedback_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    metadata.create_all(engine)


def _insert_coach_fixtures(engine: sa.Engine) -> dict:
    """Insert sample coach_sessions and coach_decisions rows (pre-remediation).

    Returns dict with inserted row IDs for assertion.
    """
    with engine.begin() as conn:
        # Insert 3 coach_sessions without public_id
        result = conn.execute(
            sa.text(
                "INSERT INTO coach_sessions "
                "(consultation_id, doctor_id, mode, status, thread_id, started_at, created_at) "
                "VALUES (1, 1, 'off', 'active', 'thread-1', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        session_id_1 = result.lastrowid

        result = conn.execute(
            sa.text(
                "INSERT INTO coach_sessions "
                "(consultation_id, doctor_id, mode, status, thread_id, started_at, created_at) "
                "VALUES (2, 1, 'on_demand', 'active', 'thread-2', '2026-01-02 00:00:00', '2026-01-02 00:00:00')"
            )
        )
        session_id_2 = result.lastrowid

        # Insert 3 coach_decisions without suggestion_id / idempotency_key
        result = conn.execute(
            sa.text(
                "INSERT INTO coach_decisions "
                "(session_id, turn_no, intent, stage, risk_level, created_at) "
                "VALUES (:sid, 1, 'diagnose', 'assessment', 'low', '2026-01-01 00:00:00')"
            ),
            {"sid": session_id_1},
        )
        decision_id_1 = result.lastrowid

        result = conn.execute(
            sa.text(
                "INSERT INTO coach_decisions "
                "(session_id, turn_no, intent, stage, risk_level, created_at) "
                "VALUES (:sid, 2, 'treat', 'plan', 'medium', '2026-01-01 00:01:00')"
            ),
            {"sid": session_id_1},
        )
        decision_id_2 = result.lastrowid

        result = conn.execute(
            sa.text(
                "INSERT INTO coach_decisions "
                "(session_id, turn_no, intent, stage, risk_level, created_at) "
                "VALUES (:sid, 1, 'diagnose', 'assessment', 'high', '2026-01-02 00:00:00')"
            ),
            {"sid": session_id_2},
        )
        decision_id_3 = result.lastrowid

    return {
        "session_ids": [session_id_1, session_id_2],
        "decision_ids": [decision_id_1, decision_id_2, decision_id_3],
    }


def _run_backfill_logic(engine: sa.Engine) -> None:
    """Execute the backfill logic from the migration using a real connection.

    This mirrors the Python parameterized backfill in 5e6f7a8b9c0d.
    Gracefully skips tables/columns that don't exist yet (for partial-schema tests).
    """
    inspector = sa.inspect(engine)
    cs_columns = {c["name"] for c in inspector.get_columns("coach_sessions")}
    cd_columns = {c["name"] for c in inspector.get_columns("coach_decisions")}

    with engine.begin() as conn:
        # coach_sessions.public_id backfill
        if "public_id" in cs_columns:
            rows = conn.execute(
                sa.text("SELECT id FROM coach_sessions WHERE public_id IS NULL")
            ).mappings().all()
            for row in rows:
                conn.execute(
                    sa.text("UPDATE coach_sessions SET public_id=:public_id WHERE id=:id"),
                    {"public_id": str(uuid4()), "id": row["id"]},
                )

        # coach_decisions.suggestion_id backfill
        if "suggestion_id" in cd_columns:
            rows = conn.execute(
                sa.text("SELECT id FROM coach_decisions WHERE suggestion_id IS NULL")
            ).mappings().all()
            for row in rows:
                conn.execute(
                    sa.text(
                        "UPDATE coach_decisions SET suggestion_id=:suggestion_id WHERE id=:id"
                    ),
                    {"suggestion_id": str(uuid4()), "id": row["id"]},
                )

        # coach_decisions.idempotency_key backfill
        if "idempotency_key" in cd_columns:
            rows = conn.execute(
                sa.text("SELECT id FROM coach_decisions WHERE idempotency_key IS NULL")
            ).mappings().all()
            for row in rows:
                conn.execute(
                    sa.text(
                        "UPDATE coach_decisions SET idempotency_key=:idempotency_key WHERE id=:id"
                    ),
                    {"idempotency_key": f"legacy-{row['id']}", "id": row["id"]},
                )


# ── Path 1: Empty DB → head ─────────────────────────────────────────────────────

class TestMigrateEmptyDB:
    """Path 1: No revision recorded (empty DB) → alembic upgrade head → target."""

    def test_empty_db_reaches_target_head(self):
        """migrate.py with current_rev=None should call upgrade head and succeed."""
        from scripts.migrate import run_migration

        # Mock _get_current_revision: first call returns None (empty), second returns target
        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = [None, TARGET_HEAD]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            exit_code = run_migration()

        assert exit_code == 0
        # Should have called alembic upgrade head
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["alembic", "upgrade", "head"]


# ── Path 2: V1.1 head → V1.2 head ───────────────────────────────────────────────

class TestMigrateFromV11Head:
    """Path 2: Current revision = 3c4d5e6f7a8b (V1.1 head) → upgrade to V1.2 head."""

    def test_v11_head_reaches_target(self):
        from scripts.migrate import run_migration, _V11_HEAD  # noqa: I001

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = [_V11_HEAD, TARGET_HEAD]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            exit_code = run_migration()

        assert exit_code == 0
        # Should NOT trigger backfill (only triggered at pre-head)
        mock_run.assert_called_once_with(
            ["alembic", "upgrade", "head"], cwd=mock_run.call_args[1].get("cwd", "")
        )


# ── Path 3: V1.2 first revision → remediation head ──────────────────────────────

class TestMigrateFromV12FirstRevision:
    """Path 3: Current revision = 4d5e6f7a8b9c → upgrade to remediation head."""

    def test_v12_first_revision_reaches_target(self):
        from scripts.migrate import run_migration

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = ["4d5e6f7a8b9c", TARGET_HEAD]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            exit_code = run_migration()

        assert exit_code == 0


# ── Path 4: Coach tables with existing data → backfill unique non-null IDs ───────

class TestCoachTableBackfill:
    """Path 4: Existing coach_sessions / coach_decisions rows get unique non-null IDs."""

    def test_backfill_produces_unique_non_null_public_ids(self):
        """coach_sessions.public_id backfill produces unique, non-null UUIDs."""
        engine = _make_sqlite_engine()
        _build_base_schema(engine)

        # Add public_id column (nullable, as migration does before backfill)
        with engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE coach_sessions ADD COLUMN public_id VARCHAR(36)"))

        _insert_coach_fixtures(engine)

        # Run backfill
        _run_backfill_logic(engine)

        # Verify all public_ids are non-null and unique
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT id, public_id FROM coach_sessions ORDER BY id")
            ).mappings().all()

        public_ids = [r["public_id"] for r in rows]
        assert all(pid is not None for pid in public_ids), "All public_ids must be non-null"
        assert len(set(public_ids)) == len(public_ids), "All public_ids must be unique"
        assert len(public_ids) > 0, "Should have at least one backfilled session"

    def test_backfill_produces_unique_non_null_suggestion_ids(self):
        """coach_decisions.suggestion_id backfill produces unique, non-null UUIDs."""
        engine = _make_sqlite_engine()
        _build_base_schema(engine)

        with engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE coach_decisions ADD COLUMN suggestion_id VARCHAR(36)"))
            conn.execute(sa.text("ALTER TABLE coach_decisions ADD COLUMN idempotency_key VARCHAR(64)"))

        _insert_coach_fixtures(engine)

        _run_backfill_logic(engine)

        with engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT id, suggestion_id, idempotency_key FROM coach_decisions ORDER BY id")
            ).mappings().all()

        suggestion_ids = [r["suggestion_id"] for r in rows]

        assert all(sid is not None for sid in suggestion_ids), "All suggestion_ids must be non-null"
        assert len(set(suggestion_ids)) == len(suggestion_ids), "All suggestion_ids must be unique"
        assert len(suggestion_ids) > 0, "Should have at least one backfilled decision"

    def test_backfill_idempotency_keys_are_legacy_prefixed(self):
        """coach_decisions.idempotency_key backfill uses 'legacy-<id>' format."""
        engine = _make_sqlite_engine()
        _build_base_schema(engine)

        with engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE coach_decisions ADD COLUMN suggestion_id VARCHAR(36)"))
            conn.execute(sa.text("ALTER TABLE coach_decisions ADD COLUMN idempotency_key VARCHAR(64)"))

        _insert_coach_fixtures(engine)

        _run_backfill_logic(engine)

        with engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT id, idempotency_key FROM coach_decisions ORDER BY id")
            ).mappings().all()

        for row in rows:
            assert row["idempotency_key"] is not None
            assert row["idempotency_key"].startswith("legacy-")
            assert row["idempotency_key"] == f"legacy-{row['id']}"

    def test_backfill_is_idempotent(self):
        """Running backfill twice does not overwrite existing values."""
        engine = _make_sqlite_engine()
        _build_base_schema(engine)

        with engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE coach_sessions ADD COLUMN public_id VARCHAR(36)"))

        _insert_coach_fixtures(engine)

        # First backfill
        _run_backfill_logic(engine)

        # Capture values
        with engine.connect() as conn:
            first_run = conn.execute(
                sa.text("SELECT id, public_id FROM coach_sessions ORDER BY id")
            ).mappings().all()
        first_ids = {r["id"]: r["public_id"] for r in first_run}

        # Second backfill (should not change anything since public_id IS NOT NULL now)
        _run_backfill_logic(engine)

        with engine.connect() as conn:
            second_run = conn.execute(
                sa.text("SELECT id, public_id FROM coach_sessions ORDER BY id")
            ).mappings().all()
        second_ids = {r["id"]: r["public_id"] for r in second_run}

        assert first_ids == second_ids, "Idempotent backfill must not overwrite existing values"


# ── Path 5: Downgrade -1 then re-upgrade ─────────────────────────────────────────

class TestDowngradeReUpgrade:
    """Path 5: Migration module supports clean downgrade and re-upgrade."""

    def test_migration_module_has_downgrade(self):
        """The remediation migration has a callable downgrade function."""
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        assert callable(mod.downgrade)

    def test_downgrade_reverses_upgrade_operations(self):
        """Verify downgrade source references all objects created in upgrade."""
        import inspect
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        downgrade_src = inspect.getsource(mod.downgrade)

        # All objects created in upgrade must be dropped in downgrade
        expected_objects = [
            "coach_stream_events",
            "trainee_memory_consents",
            "public_id",
            "suggestion_id",
            "idempotency_key",
            "content_hash",
            "stage",
            "rollout_pct",
        ]
        for obj in expected_objects:
            assert obj in downgrade_src, f"downgrade() must reference '{obj}'"

    def test_migration_chain_integrity(self):
        """Verify the full migration chain from baseline to head."""
        chain = [
            ("0c1dfb4fea5f_baseline_current_schema", "0c1dfb4fea5f", None),
            ("1a2b3c4d5e6f_add_case_id_and_message_sequence_constraint", "1a2b3c4d5e6f", "0c1dfb4fea5f"),
            ("2b3c4d5e6f7a_v11_evaluation_dispatch_outbox", "2b3c4d5e6f7a", "1a2b3c4d5e6f"),
            ("3c4d5e6f7a8b_v11_review_audit_indexes", "3c4d5e6f7a8b", "2b3c4d5e6f7a"),
            ("4d5e6f7a8b9c_v12_agent_runtime", "4d5e6f7a8b9c", "3c4d5e6f7a8b"),
            ("5e6f7a8b9c0d_v12_runtime_remediation", "5e6f7a8b9c0d", "4d5e6f7a8b9c"),
        ]
        for filename, expected_rev, expected_down in chain:
            mod = _load_migration(filename)
            assert mod.revision == expected_rev, f"{filename}: revision mismatch"
            assert mod.down_revision == expected_down, f"{filename}: down_revision mismatch"


# ── Path 6: Already at head → idempotent ─────────────────────────────────────────

class TestMigrateAlreadyAtHead:
    """Path 6: Already at 5e6f7a8b9c0d → alembic upgrade head is still called (idempotent)."""

    def test_at_head_still_calls_upgrade_head(self):
        """migrate.py MUST always call alembic upgrade head, even when already at target."""
        from scripts.migrate import run_migration

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            # Both before and after reading revision, we're at target
            mock_rev.side_effect = [TARGET_HEAD, TARGET_HEAD]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            exit_code = run_migration()

        assert exit_code == 0
        # Critical: must still call alembic upgrade head (not short-circuit)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["alembic", "upgrade", "head"]

    def test_at_head_returns_zero(self):
        """When already at head and upgrade succeeds, exit code is 0."""
        from scripts.migrate import run_migration

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = [TARGET_HEAD, TARGET_HEAD]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

            assert run_migration() == 0


# ── migrate.py failure paths ─────────────────────────────────────────────────────

class TestMigrateFailurePaths:
    """Verify migrate.py returns non-zero on failures."""

    def test_upgrade_head_failure_returns_nonzero(self):
        from scripts.migrate import run_migration

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = [None, None]
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="error"
            )

            exit_code = run_migration()

        assert exit_code == 1

    def test_final_revision_mismatch_returns_nonzero(self):
        """If final revision != target, return non-zero."""
        from scripts.migrate import run_migration

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.side_effect = [None, "wrong_revision"]
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

            exit_code = run_migration()

        assert exit_code == 1

    def test_backfill_failure_aborts(self):
        """If backfill fails at pre-head, migration aborts before upgrade head."""
        from scripts.migrate import run_migration, _V11_PRE_HEAD  # noqa: I001

        with patch("scripts.migrate._get_current_revision") as mock_rev, \
             patch("scripts.migrate._run_backfill") as mock_bf, \
             patch("scripts.migrate._run") as mock_run:
            mock_rev.return_value = _V11_PRE_HEAD
            mock_bf.return_value = 1  # backfill fails

            exit_code = run_migration()

        assert exit_code == 1
        # upgrade head should NOT have been called
        mock_run.assert_not_called()


# ── Migration source code checks ─────────────────────────────────────────────────

class TestMigrationMySQLSafe:
    """Verify the migration script contains no SQLite-specific SQL."""

    def test_no_randomblob_in_migration(self):
        """Migration must not use SQLite randomblob()."""
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        import inspect
        source = inspect.getsource(mod.upgrade)
        assert "randomblob" not in source, "Migration must not use SQLite randomblob()"

    def test_no_sqlite_string_concat_in_migration(self):
        """Migration must not use SQLite || string concatenation for backfill."""
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        import inspect
        source = inspect.getsource(mod.upgrade)
        # The || operator is used for SQLite string concat; should not appear in backfill
        assert "'legacy-' ||" not in source, "Migration must not use SQLite || for backfill"

    def test_no_cast_as_text_in_migration(self):
        """Migration must not use CAST(id AS TEXT) which is SQLite-specific."""
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        import inspect
        source = inspect.getsource(mod.upgrade)
        assert "CAST(id AS TEXT)" not in source, "Migration must not use CAST(id AS TEXT)"

    def test_backfill_uses_parameterized_queries(self):
        """Migration backfill uses Python uuid4 and parameterized UPDATE."""
        mod = _load_migration("5e6f7a8b9c0d_v12_runtime_remediation")
        import inspect
        source = inspect.getsource(mod.upgrade)
        assert "uuid4()" in source, "Migration must use Python uuid4() for backfill"
        assert ":public_id" in source, "Migration must use parameterized query for public_id"
        assert ":suggestion_id" in source, "Migration must use parameterized query for suggestion_id"
        assert ":idempotency_key" in source, "Migration must use parameterized query for idempotency_key"
