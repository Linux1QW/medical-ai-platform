"""v1.1 review audit indexes

Revision ID: 3c4d5e6f7a8b
Revises: 2b3c4d5e6f7a
"""

import sqlalchemy as sa

from alembic import context, op

revision = "3c4d5e6f7a8b"
down_revision = "2b3c4d5e6f7a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. review_records 新增 original_scores JSON ─────────────────────────
    op.add_column(
        "review_records",
        sa.Column("original_scores", sa.JSON(), nullable=True, comment="复核时五维原始分数快照"),
    )

    # ── 2. evaluation_runs 复合索引 ─────────────────────────────────────────
    op.create_index(
        "ix_evaluation_runs_consultation_created",
        "evaluation_runs",
        ["consultation_id", "created_at"],
    )
    op.create_index(
        "ix_evaluation_runs_status_updated",
        "evaluation_runs",
        ["status", "updated_at"],
    )
    op.create_index(
        "ix_evaluation_runs_evaluation_id",
        "evaluation_runs",
        ["evaluation_id"],
    )

    # ── 3. FK evaluations.run_id → evaluation_runs.id ───────────────────────
    # precheck: 不存在 orphan run_id（仅在线模式）
    if not context.is_offline_mode():
        conn = op.get_bind()
        orphan_check = conn.execute(
            sa.text(
                "SELECT e.run_id FROM evaluations e "
                "LEFT JOIN evaluation_runs er ON e.run_id = er.id "
                "WHERE e.run_id IS NOT NULL AND er.id IS NULL"
            )
        )
        orphans = orphan_check.fetchall()
        if orphans:
            raise RuntimeError(
                f"Migration blocked: found {len(orphans)} orphan run_id(s) in evaluations table. "
                f"Run backfill_legacy_evaluation_runs.py before this migration."
            )

    op.create_foreign_key(
        "fk_evaluations_run_id_evaluation_runs",
        "evaluations",
        "evaluation_runs",
        ["run_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint("fk_evaluations_run_id_evaluation_runs", "evaluations", type_="foreignkey")
    op.drop_index("ix_evaluation_runs_evaluation_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_status_updated", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_consultation_created", table_name="evaluation_runs")
    op.drop_column("review_records", "original_scores")
