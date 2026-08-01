"""add case id mapping and message sequence constraint

Revision ID: 1a2b3c4d5e6f
Revises: 0c1dfb4fea5f
"""

import sqlalchemy as sa

from alembic import op

revision = "1a2b3c4d5e6f"
down_revision = "0c1dfb4fea5f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "virtual_patients",
        sa.Column("case_id", sa.String(length=100), nullable=True, comment="数据集病例稳定标识"),
    )
    op.create_index(
        "ix_virtual_patients_case_id", "virtual_patients", ["case_id"], unique=True
    )
    op.create_unique_constraint(
        "uq_consultation_message_sequence",
        "consultation_messages",
        ["consultation_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_consultation_message_sequence", "consultation_messages", type_="unique"
    )
    op.drop_index("ix_virtual_patients_case_id", table_name="virtual_patients")
    op.drop_column("virtual_patients", "case_id")
