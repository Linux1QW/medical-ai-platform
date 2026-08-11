"""v1.2 durable streaming: decision_id on events + lease columns

Revision ID: 6f7a8b9c0d1e
Revises: 5e6f7a8b9c0d
"""

import sqlalchemy as sa
from datetime import datetime

from alembic import op

revision = "6f7a8b9c0d1e"
down_revision = "5e6f7a8b9c0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. coach_decisions: add locked_at + lease_expires_at ────────────────
    op.add_column(
        "coach_decisions",
        sa.Column("locked_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "coach_decisions",
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
    )

    # ── 2. coach_stream_events: add decision_id (nullable first) ────────────
    op.add_column(
        "coach_stream_events",
        sa.Column(
            "decision_id",
            sa.Integer(),
            nullable=True,
        ),
    )

    # ── 3. Backfill decision_id ─────────────────────────────────────────────
    # For each event, find the decision in the same session with the latest
    # created_at <= event.created_at. If any event cannot be uniquely matched,
    # fall back to the earliest decision in the session.
    bind = op.get_bind()
    if bind is not None:
        try:
            events_result = bind.execute(
                sa.text(
                    "SELECT id, session_id, created_at "
                    "FROM coach_stream_events "
                    "WHERE decision_id IS NULL"
                )
            )
            events_rows = events_result.mappings().all() if events_result is not None else []
        except Exception:
            events_rows = []

        for event_row in events_rows:
            event_id = event_row["id"]
            session_id = event_row["session_id"]
            event_created = event_row["created_at"]

            # Find the decision with the latest created_at <= event created_at
            decisions_result = bind.execute(
                sa.text(
                    "SELECT id FROM coach_decisions "
                    "WHERE session_id=:sid AND created_at<=:ec "
                    "ORDER BY created_at DESC LIMIT 2"
                ),
                {"sid": session_id, "ec": event_created},
            )
            decision_rows = (
                decisions_result.mappings().all()
                if decisions_result is not None
                else []
            )

            if len(decision_rows) == 1:
                decision_id = decision_rows[0]["id"]
            elif len(decision_rows) >= 2:
                # Multiple decisions before this event: use the latest one
                decision_id = decision_rows[0]["id"]
            else:
                # No decision before event: find earliest decision in session
                fallback_result = bind.execute(
                    sa.text(
                        "SELECT id FROM coach_decisions "
                        "WHERE session_id=:sid "
                        "ORDER BY created_at ASC LIMIT 2"
                    ),
                    {"sid": session_id},
                )
                fallback_rows = (
                    fallback_result.mappings().all()
                    if fallback_result is not None
                    else []
                )
                if len(fallback_rows) == 1:
                    decision_id = fallback_rows[0]["id"]
                elif len(fallback_rows) >= 2:
                    # Ambiguous: cannot uniquely determine → fail
                    raise RuntimeError(
                        f"Cannot uniquely attribute event id={event_id} "
                        f"in session_id={session_id} to a single decision. "
                        f"Found {len(fallback_rows)} candidate decisions."
                    )
                else:
                    raise RuntimeError(
                        f"No decision found for event id={event_id} "
                        f"in session_id={session_id}. Cannot backfill decision_id."
                    )

            bind.execute(
                sa.text(
                    "UPDATE coach_stream_events "
                    "SET decision_id=:did WHERE id=:eid"
                ),
                {"did": decision_id, "eid": event_id},
            )

    # ── 4. Make decision_id NOT NULL + FK + index ───────────────────────────
    op.alter_column(
        "coach_stream_events",
        "decision_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.create_foreign_key(
        "fk_coach_stream_event_decision_id",
        "coach_stream_events",
        "coach_decisions",
        ["decision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_coach_stream_decision",
        "coach_stream_events",
        ["decision_id"],
    )


def downgrade() -> None:
    # ── 4. Remove index, FK, column ─────────────────────────────────────────
    op.drop_index("ix_coach_stream_decision", table_name="coach_stream_events")
    op.drop_constraint(
        "fk_coach_stream_event_decision_id",
        "coach_stream_events",
        type_="foreignkey",
    )
    op.drop_column("coach_stream_events", "decision_id")

    # ── 3. Remove lease columns ─────────────────────────────────────────────
    op.drop_column("coach_decisions", "lease_expires_at")
    op.drop_column("coach_decisions", "locked_at")
