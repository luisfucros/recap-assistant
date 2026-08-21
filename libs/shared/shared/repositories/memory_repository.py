"""Per-user data access for long-term memories.

Every query is filtered by the owning ``user_id`` (via
:class:`~shared.repositories.base.UserScopedRepository`), and the owner id
always comes from the authenticated context, never a client- or
LLM-supplied argument — the same isolation invariant as every other
user-owned table.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select

from shared.core.enums import MemoryType
from shared.models.memory import LongTermMemory
from shared.repositories.base import UserScopedRepository


class LongTermMemoryRepository(UserScopedRepository[LongTermMemory]):
    """Owner-scoped access to :class:`~shared.models.memory.LongTermMemory` rows."""

    model = LongTermMemory

    async def list_recent(self, *, limit: int = 100, offset: int = 0) -> Sequence[LongTermMemory]:
        """Return a page of the user's memories of any type, newest first.

        Backs the privacy view (FR-4.5) when no type filter is given — the base
        ``list()`` has no defined order, which would make an unfiltered page
        inconsistent with :meth:`list_by_type`'s (ordered) results.
        """
        result = await self._session.execute(
            self._scoped_select().order_by(self.model.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def list_by_type(
        self, memory_type: MemoryType, *, limit: int = 100, offset: int = 0
    ) -> Sequence[LongTermMemory]:
        """Return a page of the user's memories of one type, newest first."""
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.type == memory_type)
            .order_by(self.model.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_by_ids(self, ids: Sequence[uuid.UUID]) -> Sequence[LongTermMemory]:
        """Return the user's memories for a set of ids (order unspecified).

        Backs retrieval text-hydration: a vector search returns point ids
        (memory ids) whose content must be loaded from Postgres (the source of
        truth). The query stays ``user_id``-scoped, so ids belonging to another
        user simply don't come back — a second guard behind the vector store's
        own filter.
        """
        if not ids:
            return []
        result = await self._session.execute(
            self._scoped_select().where(self.model.id.in_(list(ids)))
        )
        return result.scalars().all()

    async def list_by_document(
        self, document_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> Sequence[LongTermMemory]:
        """Return a page of the user's memories tied to one document, newest first."""
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.document_id == document_id)
            .order_by(self.model.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_summaries_covering(
        self, document_id: uuid.UUID, *, max_page_end: int | None = None
    ) -> Sequence[LongTermMemory]:
        """Return the document's summary memories, oldest page range first.

        Backs the recap loop: "what happened before page N" reads these in
        page order rather than by recency. ``max_page_end`` is the spoiler-safe
        bound — when set, summaries reaching past it are excluded so a recap
        never surfaces an unread span.
        """
        conditions = [
            self.model.document_id == document_id,
            self.model.type == MemoryType.SUMMARY,
        ]
        if max_page_end is not None:
            conditions.append(self.model.page_end <= max_page_end)
        result = await self._session.execute(
            self._scoped_select().where(*conditions).order_by(self.model.page_start.asc())
        )
        return result.scalars().all()

    async def delete(self, memory: LongTermMemory) -> None:
        """Delete an owned memory row (privacy: FR-4.5 view/delete).

        The caller must have loaded ``memory`` through this repository (so it is
        owner-scoped); the corresponding Qdrant point is deleted separately by
        the memory service, which knows the vector store.
        """
        await self._session.delete(memory)
        await self._session.flush()

    async def count(self) -> int:
        """Return the total number of the user's memories (for pagination)."""
        result = await self._session.execute(
            select(func.count()).select_from(self._scoped_select().subquery())
        )
        return int(result.scalar_one())
