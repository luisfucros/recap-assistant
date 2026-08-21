"""add long term memory

Revision ID: e46ef33e43ae
Revises: 615e168b4453
Create Date: 2026-08-10 10:51:50.590873

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e46ef33e43ae"
down_revision: str | Sequence[str] | None = "615e168b4453"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Native Postgres enum introduced by this migration. `create_type=False` on the
# column stops `create_table` from auto-emitting CREATE TYPE; we create/drop it
# explicitly so the ordering is under our control (matching the other tables).
memory_type = postgresql.ENUM(
    "preference",
    "summary",
    "concept",
    "fact",
    "habit",
    "faq",
    name="memory_type",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    memory_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "long_term_memory",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("type", memory_type, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("embedding_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_long_term_memory_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_long_term_memory_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_long_term_memory")),
    )
    op.create_index(
        op.f("ix_long_term_memory_document_id"), "long_term_memory", ["document_id"], unique=False
    )
    op.create_index(
        "ix_long_term_memory_user_document_page",
        "long_term_memory",
        ["user_id", "document_id", "page_start", "page_end"],
        unique=False,
    )
    op.create_index(
        op.f("ix_long_term_memory_user_id"), "long_term_memory", ["user_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_long_term_memory_user_id"), table_name="long_term_memory")
    op.drop_index("ix_long_term_memory_user_document_page", table_name="long_term_memory")
    op.drop_index(op.f("ix_long_term_memory_document_id"), table_name="long_term_memory")
    op.drop_table("long_term_memory")
    memory_type.drop(op.get_bind(), checkfirst=True)
