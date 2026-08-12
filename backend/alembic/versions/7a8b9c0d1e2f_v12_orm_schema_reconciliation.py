"""Reconcile the V1.2 MySQL schema with current ORM metadata.

Revision ID: 7a8b9c0d1e2f
Revises: 6f7a8b9c0d1e
"""

import sqlalchemy as sa

from alembic import op

revision = "7a8b9c0d1e2f"
down_revision = "6f7a8b9c0d1e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply ORM fields that were added after the earlier V1.2 DDL revisions."""
    op.alter_column(
        "agent_trace_events", "status", existing_type=sa.String(20),
        existing_nullable=False, comment="started / finished / error / blocked",
    )

    op.alter_column(
        "coach_sessions", "public_id", existing_type=sa.String(36),
        existing_nullable=False, comment="durable public identifier",
    )
    op.alter_column(
        "coach_sessions", "mode", existing_type=sa.String(20),
        existing_nullable=False, comment="off / on_demand / shadow",
    )
    op.alter_column(
        "coach_sessions", "status", existing_type=sa.String(20),
        existing_nullable=False, comment="active / ended / error",
    )

    op.alter_column(
        "coach_decisions", "suggestion_id", existing_type=sa.String(36),
        existing_nullable=False, nullable=False,
        comment="durable public identifier for the suggestion",
    )
    op.alter_column(
        "coach_decisions", "idempotency_key", existing_type=sa.String(64),
        existing_nullable=False, nullable=False,
        comment="client-provided idempotency key",
    )
    op.alter_column(
        "coach_decisions", "locked_at", existing_type=sa.DateTime(),
        existing_nullable=True, comment="when the decision was locked for processing",
    )
    op.alter_column(
        "coach_decisions", "lease_expires_at", existing_type=sa.DateTime(),
        existing_nullable=True, comment="lease expiry for the lock",
    )

    op.add_column(
        "coach_stream_events", sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE coach_stream_events SET created_at = UTC_TIMESTAMP() "
            "WHERE created_at IS NULL"
        )
    )
    op.alter_column(
        "coach_stream_events", "created_at", existing_type=sa.DateTime(),
        existing_nullable=True, nullable=False,
    )
    op.drop_index("event_id", table_name="coach_stream_events")
    op.create_unique_constraint(
        "uq_coach_stream_event_id", "coach_stream_events", ["event_id"],
    )
    op.alter_column(
        "coach_stream_events", "event_id", existing_type=sa.String(36),
        existing_nullable=False, comment="UUID for the event",
    )
    op.alter_column(
        "coach_stream_events", "sequence", existing_type=sa.Integer(),
        existing_nullable=False, comment="monotonically increasing per session",
    )
    op.alter_column(
        "coach_stream_events", "event_type", existing_type=sa.String(20),
        existing_nullable=False, type_=sa.String(50),
        comment="e.g. suggestion, heartbeat, error",
    )
    op.alter_column(
        "coach_stream_events", "data_json", existing_type=sa.JSON(),
        existing_nullable=False, nullable=True, comment="event payload",
    )
    op.alter_column(
        "coach_stream_events", "expires_at", existing_type=sa.DateTime(),
        existing_nullable=False, nullable=True, comment="TTL for cleanup",
    )

    op.add_column(
        "consultations",
        sa.Column(
            "memory_state", sa.Text(), nullable=True,
            comment="患者智能体记忆状态（JSON 序列化的 MemoryState）",
        ),
    )
    op.alter_column(
        "evaluation_dispatch_outbox", "lease_owner", existing_type=sa.String(160),
        existing_nullable=True, comment="当前租约持有者 (hostname:pid:uuid)",
    )

    for _name, column in (
        ("execution_owner", sa.Column("execution_owner", sa.String(160), nullable=True, comment="执行者标识 (hostname:pid:invocation_uuid)")),
        ("execution_task_id", sa.Column("execution_task_id", sa.String(36), nullable=True, comment="Celery task_id")),
        ("heartbeat_at", sa.Column("heartbeat_at", sa.DateTime(), nullable=True, comment="最近心跳时间")),
        ("lease_expires_at", sa.Column("lease_expires_at", sa.DateTime(), nullable=True, comment="租约过期时间")),
        ("cancel_requested_at", sa.Column("cancel_requested_at", sa.DateTime(), nullable=True, comment="取消请求时间")),
        ("cancel_requested_by", sa.Column("cancel_requested_by", sa.Integer(), nullable=True, comment="请求取消的用户 ID")),
    ):
        op.add_column("evaluation_runs", column)
    op.alter_column(
        "evaluation_runs", "started_at", existing_type=sa.DateTime(),
        existing_nullable=False, nullable=True,
    )

    op.alter_column(
        "experiment_assignments", "stage", existing_type=sa.String(20),
        existing_nullable=False, comment="planned / running / completed / cancelled",
    )
    op.alter_column(
        "experiment_assignments", "rollout_pct", existing_type=sa.Integer(),
        existing_nullable=False, comment="0-100 percentage for gradual rollout",
    )
    op.alter_column(
        "prompt_bundles", "status", existing_type=sa.String(20),
        existing_nullable=False, comment="draft / active / archived",
    )
    op.alter_column(
        "prompt_bundles", "content_hash", existing_type=sa.String(64),
        existing_nullable=True,
        comment="SHA-256 hash of bundle content for integrity verification",
    )
    op.alter_column(
        "trainee_memories", "status", existing_type=sa.String(20),
        existing_nullable=False, comment="candidate / approved / rejected / expired",
    )
    op.alter_column(
        "trainee_memory_consents", "granted", existing_type=sa.Boolean(),
        existing_nullable=False, comment="boolean: 1=granted, 0=not granted",
    )
    op.create_unique_constraint(
        "uq_trainee_memory_consent_doctor_id", "trainee_memory_consents", ["doctor_id"],
    )


def downgrade() -> None:
    """Restore the exact schema represented by durable-streaming revision."""
    op.drop_constraint(
        "uq_trainee_memory_consent_doctor_id", "trainee_memory_consents", type_="unique",
    )
    op.alter_column(
        "trainee_memory_consents", "granted", existing_type=sa.Boolean(),
        existing_nullable=False, comment=None,
    )
    op.alter_column("trainee_memories", "status", existing_type=sa.String(20), existing_nullable=False, comment=None)
    op.alter_column("prompt_bundles", "content_hash", existing_type=sa.String(64), existing_nullable=True, comment=None)
    op.alter_column("prompt_bundles", "status", existing_type=sa.String(20), existing_nullable=False, comment=None)
    op.alter_column("experiment_assignments", "rollout_pct", existing_type=sa.Integer(), existing_nullable=False, comment=None)
    op.alter_column("experiment_assignments", "stage", existing_type=sa.String(20), existing_nullable=False, comment=None)

    # The previous revision required started_at. Refuse a lossy downgrade when
    # rows created under V1.2 legitimately have no start timestamp.
    bind = op.get_bind()
    if bind is not None:
        missing_started_at = bind.execute(
            sa.text(
                "SELECT COUNT(*) FROM evaluation_runs WHERE started_at IS NULL"
            )
        ).scalar()
        if missing_started_at:
            raise RuntimeError(
                "Cannot downgrade: evaluation_runs contains NULL started_at rows"
            )
    op.alter_column(
        "evaluation_runs", "started_at", existing_type=sa.DateTime(),
        existing_nullable=True, nullable=False,
    )
    for name in (
        "cancel_requested_by", "cancel_requested_at", "lease_expires_at",
        "heartbeat_at", "execution_task_id", "execution_owner",
    ):
        op.drop_column("evaluation_runs", name)

    op.alter_column("evaluation_dispatch_outbox", "lease_owner", existing_type=sa.String(160), existing_nullable=True, comment="当前租约持有者")
    op.drop_column("consultations", "memory_state")

    op.alter_column("coach_stream_events", "expires_at", existing_type=sa.DateTime(), existing_nullable=True, nullable=False, comment=None)
    op.alter_column("coach_stream_events", "data_json", existing_type=sa.JSON(), existing_nullable=True, nullable=False, comment=None)
    op.alter_column("coach_stream_events", "event_type", existing_type=sa.String(50), existing_nullable=False, type_=sa.String(20), comment=None)
    op.alter_column("coach_stream_events", "sequence", existing_type=sa.Integer(), existing_nullable=False, comment=None)
    op.alter_column("coach_stream_events", "event_id", existing_type=sa.String(36), existing_nullable=False, comment=None)
    op.drop_constraint("uq_coach_stream_event_id", "coach_stream_events", type_="unique")
    op.create_index("event_id", "coach_stream_events", ["event_id"], unique=True)
    op.drop_column("coach_stream_events", "created_at")

    op.alter_column("coach_decisions", "lease_expires_at", existing_type=sa.DateTime(), existing_nullable=True, comment=None)
    op.alter_column("coach_decisions", "locked_at", existing_type=sa.DateTime(), existing_nullable=True, comment=None)
    op.alter_column("coach_decisions", "idempotency_key", existing_type=sa.String(64), existing_nullable=False, comment=None)
    op.alter_column("coach_decisions", "suggestion_id", existing_type=sa.String(36), existing_nullable=False, comment=None)
    op.alter_column("coach_sessions", "status", existing_type=sa.String(20), existing_nullable=False, comment=None)
    op.alter_column("coach_sessions", "mode", existing_type=sa.String(20), existing_nullable=False, comment=None)
    op.alter_column("coach_sessions", "public_id", existing_type=sa.String(36), existing_nullable=False, comment=None)
    op.alter_column("agent_trace_events", "status", existing_type=sa.String(20), existing_nullable=False, comment=None)
