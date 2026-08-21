"""add usage events

Revision ID: c51be0bcaffb
Revises: 71db456cd49a
Create Date: 2026-08-11 17:39:22.917769

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c51be0bcaffb"
down_revision: str | Sequence[str] | None = "71db456cd49a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Native Postgres enum introduced by this migration. `create_type=False` on the
# column stops `create_table` from auto-emitting CREATE TYPE; we create/drop it
# explicitly so the ordering is under our control (matching the other tables).
usage_event_type = postgresql.ENUM(
    "token_usage",
    "tool_call",
    name="usage_event_type",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    usage_event_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "usage_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("type", usage_event_type, nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("tool_name", sa.String(length=100), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_usage_events_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_usage_events")),
    )
    op.create_index(
        op.f("ix_usage_events_occurred_at"), "usage_events", ["occurred_at"], unique=False
    )
    op.create_index(op.f("ix_usage_events_user_id"), "usage_events", ["user_id"], unique=False)
    op.create_index(
        "ix_usage_events_user_occurred", "usage_events", ["user_id", "occurred_at"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_events_user_occurred", table_name="usage_events")
    op.drop_index(op.f("ix_usage_events_user_id"), table_name="usage_events")
    op.drop_index(op.f("ix_usage_events_occurred_at"), table_name="usage_events")
    op.drop_table("usage_events")
    usage_event_type.drop(op.get_bind(), checkfirst=True)
