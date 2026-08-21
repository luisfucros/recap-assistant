"""The ``Document`` and ``Chunk`` models — a user's uploaded books/docs.

A ``Document`` is the aggregate the ingestion pipeline drives from ``pending`` to
``indexed``; its ``Chunk`` rows are the parsed, page-tagged text spans whose
vectors live in Qdrant (``vector_id`` points at the Qdrant point). Text is the
source of truth here — vectors can be regenerated from ``Chunk.text`` on a
provider switch without re-parsing.

Two load-bearing invariants are expressed in the schema:

* **Per-user duplicate rejection** — a unique ``(user_id, content_sha256)``
  constraint. The same bytes uploaded twice by one user collide at the DB (so
  the upload path can rely on ``ON CONFLICT`` rather than an app-level lock),
  while identical content across *different* users stays separate (isolation
  wins over dedup).
* **Per-user isolation** — ``user_id`` is carried on both tables (denormalized
  onto ``chunks`` to mirror the Qdrant payload and let chunk queries filter by
  owner without a join). Deleting a user or a document cascades to its chunks.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import DocumentFormat, DocumentStatus, Language
from shared.db.base import Base
from shared.models.types import DOCUMENT_FORMAT_TYPE, DOCUMENT_STATUS_TYPE, LANGUAGE_TYPE


class Document(Base):
    """An uploaded document owned by one user and ingested asynchronously."""

    __tablename__ = "documents"
    __table_args__ = (
        # Per-user duplicate rejection + race-safety for concurrent uploads.
        UniqueConstraint("user_id", "content_sha256", name="uq_documents_user_id_content_sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Bibliographic metadata (extracted at parse time; may be unknown at upload).
    title: Mapped[str | None] = mapped_column(String(1024))
    author: Mapped[str | None] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(1024))

    # Content-addressed object-storage key: "<user_id>/sha256/<hash>.<ext>".
    object_key: Mapped[str] = mapped_column(String(1024))
    # Hex SHA-256 of the original bytes; half of the per-user uniqueness key.
    content_sha256: Mapped[str] = mapped_column(String(64))
    format: Mapped[DocumentFormat] = mapped_column(DOCUMENT_FORMAT_TYPE)
    # Detected at ingestion (null until then); user-overridable afterwards.
    language: Mapped[Language | None] = mapped_column(LANGUAGE_TYPE)

    status: Mapped[DocumentStatus] = mapped_column(
        DOCUMENT_STATUS_TYPE,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )
    # Human-readable reason set only when status is FAILED.
    failure_reason: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    # Which embedder produced the document's current vectors — a provider switch
    # re-embeds from chunk text and updates this.
    embed_model: Mapped[str] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Bumped on every status transition (claim, terminal success/failure); backs
    # the stuck-document sweep, which has no other way to tell how long a row has
    # sat in ``pending``/``processing``.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # Set when the document reaches the terminal INDEXED state.
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chunk(Base):
    """A parsed, page-tagged text span of a document; its vector lives in Qdrant."""

    __tablename__ = "chunks"
    __table_args__ = (
        # Chunk order within a document, and owner-scoped listing.
        Index("ix_chunks_document_id_ordinal", "document_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # Denormalized owner id: mirrors the Qdrant payload and lets chunk queries
    # enforce per-user isolation without joining through documents.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Position of this chunk in the document (0-based, contiguous).
    ordinal: Mapped[int] = mapped_column(Integer)
    # Inclusive 1-based page span this chunk was drawn from (drives read-range
    # scoping); null when the format has no page structure.
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    chapter: Mapped[str | None] = mapped_column(String(512))
    section: Mapped[str | None] = mapped_column(String(512))

    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    # Hash of the chunk text for retrieval-time near-duplicate collapsing.
    content_hash: Mapped[str] = mapped_column(String(64))
    # Id of the corresponding Qdrant point; null until the vector is upserted.
    vector_id: Mapped[str | None] = mapped_column(String(64))
