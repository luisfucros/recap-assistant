"""add documents chunks outbox

Revision ID: f64732f8bfbf
Revises: 9b8aa5234228
Create Date: 2026-07-22 09:44:48.014012

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f64732f8bfbf"
down_revision: str | Sequence[str] | None = "9b8aa5234228"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Native Postgres enum types. `create_type=False` on the column objects stops
# `create_table` from auto-emitting CREATE TYPE; we create/drop them explicitly
# so the ordering is under our control. `language` already exists (added in the
# preceding migration), so it is referenced but never (re-)created or dropped here.
document_format = postgresql.ENUM("pdf", name="document_format", create_type=False)
document_status = postgresql.ENUM(
    "pending", "processing", "indexed", "failed", name="document_status", create_type=False
)
language = postgresql.ENUM("en", "es", "de", "fr", "it", name="language", create_type=False)


def upgrade() -> None:
    """Upgrade schema."""
    # New enum types this migration introduces (language predates it).
    document_format.create(op.get_bind(), checkfirst=True)
    document_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("aggregate_id", sa.Uuid(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox")),
    )
    op.create_index(
        "ix_outbox_unprocessed",
        "outbox",
        ["created_at"],
        unique=False,
        postgresql_where=sa.text("processed_at IS NULL"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=True),
        sa.Column("author", sa.String(length=512), nullable=True),
        sa.Column("filename", sa.String(length=1024), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("format", document_format, nullable=False),
        sa.Column("language", language, nullable=True),
        sa.Column("status", document_status, server_default="pending", nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("embed_model", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_documents_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "user_id", "content_sha256", name="uq_documents_user_id_content_sha256"
        ),
    )
    op.create_index(op.f("ix_documents_user_id"), "documents", ["user_id"], unique=False)
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("chapter", sa.String(length=512), nullable=True),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("vector_id", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_chunks_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
    )
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_document_id_ordinal", "chunks", ["document_id", "ordinal"], unique=False
    )
    op.create_index(op.f("ix_chunks_user_id"), "chunks", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_chunks_user_id"), table_name="chunks")
    op.drop_index("ix_chunks_document_id_ordinal", table_name="chunks")
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_documents_user_id"), table_name="documents")
    op.drop_table("documents")
    op.drop_index(
        "ix_outbox_unprocessed",
        table_name="outbox",
        postgresql_where=sa.text("processed_at IS NULL"),
    )
    op.drop_table("outbox")
    # Drop only the enum types this migration created; leave `language` in place.
    document_status.drop(op.get_bind(), checkfirst=True)
    document_format.drop(op.get_bind(), checkfirst=True)
