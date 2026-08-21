"""``LongTermMemory``: the cross-session memory row (relational side).

This is the third memory tier, orthogonal to the LangGraph checkpointer
(short-term, per-conversation run state) and the agent scratchpad (turn-scoped
working notes): a memory here persists across sessions and documents, and is
what powers instant recaps and personalization. The row is the source of truth
for its content; ``embedding_id`` points at the corresponding vector in the
``long_term_memory`` Qdrant collection (mirroring how ``Chunk.vector_id`` points
at a ``document_chunks`` point), so semantic recall can hydrate full content from
here. Writing both sides together is the memory service's job, not the ORM's.

``SUMMARY``-type memories are tied to ``(document_id, page_start, page_end)`` —
the page-range recap loop (FR-4.3) — so a later "what happened before page N"
retrieves the saved summary instead of re-reading. The rest (``PREFERENCE``,
``CONCEPT``, ``FACT``, ``HABIT``, ``FAQ``) are user-level facts with no document
tie, so ``document_id``/``page_start``/``page_end`` stay null for them.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import MemoryType
from shared.db.base import Base
from shared.models.types import MEMORY_TYPE_TYPE


class LongTermMemory(Base):
    """A durable, cross-session memory: a user fact or a page-range summary."""

    __tablename__ = "long_term_memory"
    __table_args__ = (
        # Recap lookup: "this user's summaries for this document, by page range".
        Index(
            "ix_long_term_memory_user_document_page",
            "user_id",
            "document_id",
            "page_start",
            "page_end",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Only set for summary memories (a page-range recap of one document); null
    # for user-level facts/preferences/habits that don't tie to a document.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    type: Mapped[MemoryType] = mapped_column(MEMORY_TYPE_TYPE)
    content: Mapped[str] = mapped_column(Text)

    # Inclusive 1-based page span this memory covers (summary memories only).
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)

    # Id of the corresponding Qdrant point; null until the vector is upserted.
    embedding_id: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
