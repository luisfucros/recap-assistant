"""Per-user data access for documents and their chunks.

Both repositories are :class:`~shared.repositories.base.UserScopedRepository`
subjects: every query is filtered by the owning ``user_id`` bound at
construction, so the per-user isolation invariant cannot be forgotten at a call
site. The owner id always comes from the authenticated context, never from a
client- or LLM-supplied argument.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select

from shared.models.document import Chunk, Document
from shared.repositories.base import UserScopedRepository


class DocumentRepository(UserScopedRepository[Document]):
    """Owner-scoped access to :class:`~shared.models.document.Document` rows."""

    model = Document

    async def get_by_content_sha256(self, content_sha256: str) -> Document | None:
        """Return the user's document with this content hash, if any.

        Backs duplicate detection: an existing row here is what a re-upload
        returns as the ``409 DUPLICATE_DOCUMENT`` target, without exposing
        whether *another* user holds the same content.
        """
        result = await self._session.execute(
            self._scoped_select().where(self.model.content_sha256 == content_sha256)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, *, limit: int = 10, offset: int = 0) -> Sequence[Document]:
        """Return a page of the user's documents, newest first."""
        result = await self._session.execute(
            self._scoped_select().order_by(self.model.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def count(self) -> int:
        """Return the total number of the user's documents (for pagination)."""
        result = await self._session.execute(
            select(func.count()).select_from(self._scoped_select().subquery())
        )
        return int(result.scalar_one())

    async def delete(self, document: Document) -> None:
        """Delete an owned document row; its chunks go via the DB FK cascade.

        The caller must have loaded ``document`` through this repository (so it is
        owner-scoped). Chunk rows are removed by the ``ON DELETE CASCADE`` foreign
        key, not the ORM, so no relationship configuration is required.
        """
        await self._session.delete(document)
        await self._session.flush()


class ChunkRepository(UserScopedRepository[Chunk]):
    """Owner-scoped access to a document's :class:`~shared.models.document.Chunk` rows."""

    model = Chunk

    async def add_many(self, chunks: Sequence[Chunk]) -> None:
        """Persist a batch of chunks, rejecting any not owned by this repository.

        Bulk path for the ingestion worker. The ownership check mirrors
        :meth:`UserScopedRepository.add` so a stray ``user_id`` can't be written
        through the batch API either.
        """
        for chunk in chunks:
            if chunk.user_id != self._user_id:
                raise ValueError("chunk.user_id does not match the repository's owner")
        self._session.add_all(list(chunks))
        await self._session.flush()

    async def list_by_ids(self, ids: Sequence[uuid.UUID]) -> Sequence[Chunk]:
        """Return the user's chunks for a set of ids (order unspecified).

        Backs retrieval text-hydration: a vector search returns point ids (chunk
        ids) whose text must be loaded from Postgres (the source of truth). The
        query stays ``user_id``-scoped, so ids that belong to another user simply
        don't come back — a second guard behind the vector store's own filter.
        """
        if not ids:
            return []
        result = await self._session.execute(
            self._scoped_select().where(self.model.id.in_(list(ids)))
        )
        return result.scalars().all()

    async def list_by_document(
        self, document_id: uuid.UUID, *, limit: int = 1000, offset: int = 0
    ) -> Sequence[Chunk]:
        """Return the document's chunks in reading order (``ordinal`` ascending)."""
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.document_id == document_id)
            .order_by(self.model.ordinal.asc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_by_document_page_range(
        self, document_id: uuid.UUID, *, page_start: int, page_end: int
    ) -> Sequence[Chunk]:
        """Return the document's chunks overlapping ``[page_start, page_end]``, in order.

        Coverage (not relevance) fetch backing the ``summarize`` tool: a recap of a
        page span needs *every* chunk in that span, so this returns each chunk whose
        page range overlaps the request (``page_start <= page_end`` and
        ``page_end >= page_start``), ordered by ``ordinal``. Chunks with no page
        tags are excluded — they can't be placed within the span. Stays
        ``user_id``-scoped, so only the caller's chunks are ever returned.
        """
        result = await self._session.execute(
            self._scoped_select()
            .where(
                self.model.document_id == document_id,
                self.model.page_start.is_not(None),
                self.model.page_end.is_not(None),
                self.model.page_start <= page_end,
                self.model.page_end >= page_start,
            )
            .order_by(self.model.ordinal.asc())
        )
        return result.scalars().all()

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Delete all of the user's chunks for a document (idempotent re-ingest)."""
        await self._session.execute(
            delete(self.model).where(
                self.model.user_id == self._user_id,
                self.model.document_id == document_id,
            )
        )
