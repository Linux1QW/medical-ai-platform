"""v1.2 runtime remediation

Revision ID: 5e6f7a8b9c0d
Revises: 4d5e6f7a8b9c
"""

import sqlalchemy as sa

from alembic import op

revision = "5e6f7a8b9c0d"
down_revision = "4d5e6f7a8b9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. coach_sessions: add public_id + check constraints ────────────────
    op.add_column(
        "coach_sessions",
        sa.Column("public_id", sa.String(36), nullable=True),
    )
    # backfill public_id with uuid for existing rows
    op.execute(
        "UPDATE coach_sessions SET public_id = "
        "(SELECT lower(hex(randomblob(4))) || '-' || hex(randomblob(2)) || '-4' "
        "|| substr(hex(randomblob(2)),2) || '-' "
        "|| substr('89ab', abs(random()) % 4 + 1, 1) "
        "|| substr(hex(randomblob(2)),2) || '-' || hex(randomblob(6))) "
        "WHERE public_id IS NULL"
    )
    op.alter_column(
        "coach_sessions", "public_id",
        existing_type=sa.String(36), nullable=False,
    )
    op.create_unique_constraint("uq_coach_session_public_id", "coach_sessions", ["public_id"])
    op.create_check_constraint(
        "ck_coach_session_mode",
        "coach_sessions",
        "mode IN ('off', 'on_demand', 'shadow')",
    )
    op.create_check_constraint(
        "ck_coach_session_status",
        "coach_sessions",
        "status IN ('active', 'ended', 'error')",
    )

    # ── 2. coach_decisions: add suggestion_id, idempotency_key + constraints ─
    op.add_column(
        "coach_decisions",
        sa.Column("suggestion_id", sa.String(36), nullable=True),
    )
    op.add_column(
        "coach_decisions",
        sa.Column("idempotency_key", sa.String(64), nullable=True),
    )
    # backfill
    op.execute(
        "UPDATE coach_decisions SET suggestion_id = "
        "(SELECT lower(hex(randomblob(4))) || '-' || hex(randomblob(2)) || '-4' "
        "|| substr(hex(randomblob(2)),2) || '-' "
        "|| substr('89ab', abs(random()) % 4 + 1, 1) "
        "|| substr(hex(randomblob(2)),2) || '-' || hex(randomblob(6))) "
        "WHERE suggestion_id IS NULL"
    )
    op.execute(
        "UPDATE coach_decisions SET idempotency_key = "
        "'legacy-' || CAST(id AS TEXT) "
        "WHERE idempotency_key IS NULL"
    )
    op.alter_column(
        "coach_decisions", "suggestion_id",
        existing_type=sa.String(36), nullable=False,
    )
    op.alter_column(
        "coach_decisions", "idempotency_key",
        existing_type=sa.String(64), nullable=False,
    )
    op.create_unique_constraint(
        "uq_coach_decision_suggestion_id", "coach_decisions", ["suggestion_id"]
    )
    op.create_unique_constraint(
        "uq_coach_decision_session_idempotency",
        "coach_decisions",
        ["session_id", "idempotency_key"],
    )
    op.create_check_constraint(
        "ck_coach_decision_risk_level",
        "coach_decisions",
        "risk_level IN ('low', 'medium', 'high', 'critical')",
    )

    # ── 3. coach_stream_events (new table) ──────────────────────────────────
    op.create_table(
        "coach_stream_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(20), nullable=False),
        sa.Column("data_json", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["coach_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
        sa.UniqueConstraint(
            "session_id", "sequence", name="uq_coach_stream_session_sequence"
        ),
    )
    op.create_index(
        "ix_coach_stream_session_expires",
        "coach_stream_events",
        ["session_id", "expires_at"],
    )

    # ── 4. trainee_memories: add FKs, index, check constraint ───────────────
    op.create_foreign_key(
        "fk_trainee_memory_doctor_id_users",
        "trainee_memories",
        "users",
        ["doctor_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_trainee_memory_reviewer_id_users",
        "trainee_memories",
        "users",
        ["reviewer_id"],
        ["id"],
    )
    op.create_index(
        "ix_trainee_memory_doctor_status_expires",
        "trainee_memories",
        ["doctor_id", "status", "expires_at"],
    )
    op.create_check_constraint(
        "ck_trainee_memory_status",
        "trainee_memories",
        "status IN ('candidate', 'approved', 'rejected', 'expired')",
    )

    # ── 5. trainee_memory_consents (new table) ──────────────────────────────
    op.create_table(
        "trainee_memory_consents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doctor_id", sa.Integer(), nullable=False),
        sa.Column("granted", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("granted_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["doctor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── 6. prompt_bundles: unique (name, version) + content_hash ────────────
    op.add_column(
        "prompt_bundles",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    op.create_unique_constraint(
        "uq_prompt_bundle_name_version",
        "prompt_bundles",
        ["name", "version"],
    )

    # ── 7. experiment_assignments: stage, rollout_pct, check constraints ────
    op.add_column(
        "experiment_assignments",
        sa.Column("stage", sa.String(20), nullable=False, server_default="planned"),
    )
    op.add_column(
        "experiment_assignments",
        sa.Column("rollout_pct", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_check_constraint(
        "ck_experiment_stage",
        "experiment_assignments",
        "stage IN ('planned', 'running', 'completed', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_experiment_rollout_pct",
        "experiment_assignments",
        "rollout_pct BETWEEN 0 AND 100",
    )


def downgrade() -> None:
    # ── 7. experiment_assignments ───────────────────────────────────────────
    op.drop_constraint("ck_experiment_rollout_pct", "experiment_assignments", type_="check")
    op.drop_constraint("ck_experiment_stage", "experiment_assignments", type_="check")
    op.drop_column("experiment_assignments", "rollout_pct")
    op.drop_column("experiment_assignments", "stage")

    # ── 6. prompt_bundles ──────────────────────────────────────────────────
    op.drop_constraint("uq_prompt_bundle_name_version", "prompt_bundles", type_="unique")
    op.drop_column("prompt_bundles", "content_hash")

    # ── 5. trainee_memory_consents ──────────────────────────────────────────
    op.drop_table("trainee_memory_consents")

    # ── 4. trainee_memories ─────────────────────────────────────────────────
    op.drop_constraint("ck_trainee_memory_status", "trainee_memories", type_="check")
    op.drop_index("ix_trainee_memory_doctor_status_expires", table_name="trainee_memories")
    op.drop_constraint("fk_trainee_memory_reviewer_id_users", "trainee_memories", type_="foreignkey")
    op.drop_constraint("fk_trainee_memory_doctor_id_users", "trainee_memories", type_="foreignkey")

    # ── 3. coach_stream_events ──────────────────────────────────────────────
    op.drop_index("ix_coach_stream_session_expires", table_name="coach_stream_events")
    op.drop_table("coach_stream_events")

    # ── 2. coach_decisions ─────────────────────────────────────────────────
    op.drop_constraint("ck_coach_decision_risk_level", "coach_decisions", type_="check")
    op.drop_constraint("uq_coach_decision_session_idempotency", "coach_decisions", type_="unique")
    op.drop_constraint("uq_coach_decision_suggestion_id", "coach_decisions", type_="unique")
    op.drop_column("coach_decisions", "idempotency_key")
    op.drop_column("coach_decisions", "suggestion_id")

    # ── 1. coach_sessions ──────────────────────────────────────────────────
    op.drop_constraint("ck_coach_session_status", "coach_sessions", type_="check")
    op.drop_constraint("ck_coach_session_mode", "coach_sessions", type_="check")
    op.drop_constraint("uq_coach_session_public_id", "coach_sessions", type_="unique")
    op.drop_column("coach_sessions", "public_id")
