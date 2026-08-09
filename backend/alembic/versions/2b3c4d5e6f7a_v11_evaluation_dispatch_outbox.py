"""v1.1 evaluation dispatch outbox

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
"""

import sqlalchemy as sa

from alembic import op

revision = "2b3c4d5e6f7a"
down_revision = "1a2b3c4d5e6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 前置检查：非空 run_id 重复则失败 ─────────────────────────────────────
    conn = op.get_bind()
    dup_check = conn.execute(
        sa.text(
            "SELECT run_id, COUNT(*) AS cnt FROM evaluations "
            "WHERE run_id IS NOT NULL "
            "GROUP BY run_id HAVING cnt > 1"
        )
    )
    duplicates = dup_check.fetchall()
    if duplicates:
        raise RuntimeError(
            f"Migration blocked: found {len(duplicates)} duplicate non-null run_id(s) "
            f"in evaluations table. Resolve duplicates before running this migration."
        )

    # ── 1. 创建 evaluation_dispatch_outbox 表 ────────────────────────────────
    op.create_table(
        "evaluation_dispatch_outbox",
        sa.Column("event_id", sa.String(36), primary_key=True, comment="Outbox event UUID"),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            comment="关联的评估运行 ID",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
            comment="pending | leased | published | cancelled | dead_letter",
        ),
        sa.Column("task_name", sa.String(100), nullable=False, comment="Celery task 名称"),
        sa.Column("payload", sa.JSON(), nullable=False, comment="投递 payload（去标识化）"),
        sa.Column(
            "attempt", sa.Integer(), nullable=False, server_default="0", comment="已尝试次数"
        ),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=True, comment="下次重试时间"),
        sa.Column(
            "lease_owner", sa.String(160), nullable=True, comment="当前租约持有者"
        ),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True, comment="租约过期时间"),
        sa.Column("last_error_code", sa.String(100), nullable=True, comment="最近错误码"),
        sa.Column("last_task_id", sa.String(36), nullable=True, comment="最近 Celery task ID"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
            comment="更新时间",
        ),
        sa.Column("published_at", sa.DateTime(), nullable=True, comment="成功投递时间"),
    )

    # 索引
    op.create_index(
        "ix_dispatch_outbox_status_next_attempt",
        "evaluation_dispatch_outbox",
        ["status", "next_attempt_at"],
    )
    op.create_index(
        "ix_dispatch_outbox_lease_expires",
        "evaluation_dispatch_outbox",
        ["lease_expires_at"],
    )
    op.create_index(
        "ix_dispatch_outbox_created_at",
        "evaluation_dispatch_outbox",
        ["created_at"],
    )

    # ── 2. 修改 evaluation_runs 表 ────────────────────────────────────────────
    # started_at 放宽为 nullable（已是 nullable，但确认 server_default 无约束）
    # status server_default 调整为 queued
    op.alter_column(
        "evaluation_runs",
        "status",
        server_default="queued",
        existing_type=sa.String(30),
        existing_nullable=False,
    )
    # attempt server_default 调整为 0
    op.alter_column(
        "evaluation_runs",
        "attempt",
        server_default="0",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )

    # ── 3. evaluations.run_id 添加允许多 NULL 的唯一索引 ─────────────────────
    op.create_index(
        "ux_evaluations_run_id",
        "evaluations",
        ["run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_evaluations_run_id", table_name="evaluations")
    op.alter_column(
        "evaluation_runs",
        "attempt",
        server_default=None,
        existing_type=sa.Integer(),
        existing_nullable=False,
    )
    op.alter_column(
        "evaluation_runs",
        "status",
        server_default=None,
        existing_type=sa.String(30),
        existing_nullable=False,
    )
    op.drop_index("ix_dispatch_outbox_created_at", table_name="evaluation_dispatch_outbox")
    op.drop_index("ix_dispatch_outbox_lease_expires", table_name="evaluation_dispatch_outbox")
    op.drop_index(
        "ix_dispatch_outbox_status_next_attempt", table_name="evaluation_dispatch_outbox"
    )
    op.drop_table("evaluation_dispatch_outbox")
