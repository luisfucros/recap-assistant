"""add reading_progress reading_events

Revision ID: a76b0cdc7292
Revises: f64732f8bfbf
Create Date: 2026-08-03 16:09:19.192673

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a76b0cdc7292"
down_revision: str | Sequence[str] | None = "f64732f8bfbf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Native Postgres enum types introduced by this migration. `create_type=False` on
# the column objects stops `create_table` from auto-emitting CREATE TYPE; we
# create/drop them explicitly so the ordering is under our control.
reading_status = postgresql.ENUM(
    "not_started",
    "reading",
    "completed",
    "cancelled",
    name="reading_status",
    create_type=False,
)
reading_event_type = postgresql.ENUM(
    "position_advanced",
    "status_changed",
    "session",
    "completed",
    name="reading_event_type",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    reading_status.create(op.get_bind(), checkfirst=True)
    reading_event_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "reading_progress",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("current_page", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_summarized_page", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", reading_status, server_default="not_started", nullable=False),
        sa.Column("spoiler_safe", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_accessed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_reading_progress_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_reading_progress_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reading_progress")),
        sa.UniqueConstraint("user_id", "document_id", name="uq_reading_progress_user_document"),
    )
    op.create_index(
        op.f("ix_reading_progress_document_id"),
        "reading_progress",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reading_progress_user_id"), "reading_progress", ["user_id"], unique=False
    )

    op.create_table(
        "reading_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("type", reading_event_type, nullable=False),
        sa.Column("from_page", sa.Integer(), nullable=True),
        sa.Column("to_page", sa.Integer(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_reading_events_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_reading_events_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reading_events")),
    )
    op.create_index(
        op.f("ix_reading_events_document_id"), "reading_events", ["document_id"], unique=False
    )
    op.create_index(
        op.f("ix_reading_events_occurred_at"), "reading_events", ["occurred_at"], unique=False
    )
    op.create_index(op.f("ix_reading_events_user_id"), "reading_events", ["user_id"], unique=False)
    op.create_index(
        "ix_reading_events_user_occurred",
        "reading_events",
        ["user_id", "occurred_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_reading_events_user_occurred", table_name="reading_events")
    op.drop_index(op.f("ix_reading_events_user_id"), table_name="reading_events")
    op.drop_index(op.f("ix_reading_events_occurred_at"), table_name="reading_events")
    op.drop_index(op.f("ix_reading_events_document_id"), table_name="reading_events")
    op.drop_table("reading_events")
    op.drop_index(op.f("ix_reading_progress_user_id"), table_name="reading_progress")
    op.drop_index(op.f("ix_reading_progress_document_id"), table_name="reading_progress")
    op.drop_table("reading_progress")
    reading_event_type.drop(op.get_bind(), checkfirst=True)
    reading_status.drop(op.get_bind(), checkfirst=True)
