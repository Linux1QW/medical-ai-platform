"""v1.2 agent runtime

Revision ID: 4d5e6f7a8b9c
Revises: 3c4d5e6f7a8b
"""

import sqlalchemy as sa

from alembic import op

revision = "4d5e6f7a8b9c"
down_revision = "3c4d5e6f7a8b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. coach_sessions ────────────────────────────────────────────────────
    op.create_table(
        "coach_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
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
        sa.ForeignKeyConstraint(["consultation_id"], ["consultations.id"]),
        sa.ForeignKeyConstraint(["doctor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("consultation_id"),
        sa.UniqueConstraint("thread_id"),
    )

    # ── 2. coach_decisions ───────────────────────────────────────────────────
    op.create_table(
        "coach_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
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
        sa.ForeignKeyConstraint(
            ["session_id"], ["coach_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id", "turn_no", name="uq_coach_decision_session_turn"
        ),
    )

    # ── 3. agent_trace_events ────────────────────────────────────────────────
    op.create_table(
        "agent_trace_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(120), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("agent_name", sa.String(80), nullable=True),
        sa.Column("node_name", sa.String(80), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=True),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("input_hmac", sa.String(64), nullable=True),
        sa.Column("output_hmac", sa.String(64), nullable=True),
        sa.Column("input_char_count", sa.Integer(), nullable=True),
        sa.Column("output_char_count", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "trace_id", "sequence", name="uq_trace_event_trace_sequence"
        ),
    )
    op.create_index(
        "ix_trace_event_session_created",
        "agent_trace_events",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_trace_event_type_created",
        "agent_trace_events",
        ["event_type", "created_at"],
    )
    op.create_index(
        "ix_trace_event_status_created",
        "agent_trace_events",
        ["status", "created_at"],
    )

    # ── 4. trainee_memories ──────────────────────────────────────────────────
    op.create_table(
        "trainee_memories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doctor_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("skill_dimension", sa.String(80), nullable=False),
        sa.Column("summary", sa.String(500), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), nullable=True),
        sa.Column("review_comment", sa.String(500), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── 5. prompt_bundles ────────────────────────────────────────────────────
    op.create_table(
        "prompt_bundles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("node_prompts", sa.JSON(), nullable=True),
        sa.Column("output_schemas", sa.JSON(), nullable=True),
        sa.Column("model_config", sa.JSON(), nullable=True),
        sa.Column("skill_manifest_checksum", sa.String(64), nullable=True),
        sa.Column("source_commit", sa.String(40), nullable=False),
        sa.Column("author", sa.String(120), nullable=False),
        sa.Column("approval_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── 6. experiment_assignments ────────────────────────────────────────────
    op.create_table(
        "experiment_assignments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(80), nullable=False),
        sa.Column("subject_id", sa.String(120), nullable=False),
        sa.Column("variant", sa.String(50), nullable=False),
        sa.Column("weight", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assigned_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "experiment_id", "subject_id", name="uq_experiment_subject"
        ),
    )


def downgrade() -> None:
    op.drop_table("experiment_assignments")
    op.drop_table("prompt_bundles")
    op.drop_table("trainee_memories")
    op.drop_index(
        "ix_trace_event_status_created", table_name="agent_trace_events"
    )
    op.drop_index(
        "ix_trace_event_type_created", table_name="agent_trace_events"
    )
    op.drop_index(
        "ix_trace_event_session_created", table_name="agent_trace_events"
    )
    op.drop_table("agent_trace_events")
    op.drop_table("coach_decisions")
    op.drop_table("coach_sessions")
