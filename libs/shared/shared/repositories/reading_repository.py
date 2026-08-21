"""Per-user data access for reading progress and the reading-event trail.

Both repositories are :class:`~shared.repositories.base.UserScopedRepository`
subjects: every query is filtered by the owning ``user_id`` bound at
construction, so a caller can never read or widen another user's reading state.
The owner id always comes from the authenticated context, never from a client-
or LLM-supplied argument.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select

from shared.core.enums import ReadingStatus
from shared.models.reading import ReadingEvent, ReadingProgress
from shared.repositories.base import UserScopedRepository


class ReadingProgressRepository(UserScopedRepository[ReadingProgress]):
    """Owner-scoped access to :class:`~shared.models.reading.ReadingProgress`."""

    model = ReadingProgress

    async def get_by_document(self, document_id: uuid.UUID) -> ReadingProgress | None:
        """Return the user's progress row for a document, or ``None`` if untracked.

        The natural-key lookup behind get-or-create: there is at most one row per
        ``(user_id, document_id)`` (unique constraint), so this resolves to a
        single row.
        """
        result = await self._session.execute(
            self._scoped_select().where(self.model.document_id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_by_status(
        self, status: ReadingStatus, *, limit: int = 100, offset: int = 0
    ) -> Sequence[ReadingProgress]:
        """Return the user's progress rows in one status, most-recently-read first."""
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.status == status)
            .order_by(self.model.last_accessed_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_recent(self, *, limit: int = 10, offset: int = 0) -> Sequence[ReadingProgress]:
        """Return the user's progress rows ordered by most recently accessed."""
        result = await self._session.execute(
            self._scoped_select()
            .order_by(self.model.last_accessed_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_by_status(self, status: ReadingStatus) -> int:
        """Return the exact number of the user's documents in one reading status.

        Used by analytics for started/completed/cancelled totals — an exact count
        rather than ``len(list_by_status(...))``, which is capped by its page size.
        """
        result = await self._session.execute(
            select(func.count()).select_from(
                self._scoped_select().where(self.model.status == status).subquery()
            )
        )
        return int(result.scalar_one())


class ReadingEventRepository(UserScopedRepository[ReadingEvent]):
    """Owner-scoped, append-only access to the reading-event analytics trail.

    Events are only ever inserted (via the inherited :meth:`add`) and read back
    for analytics; there is no update path, preserving the auditable history.
    """

    model = ReadingEvent

    async def list_since(self, since: datetime, *, limit: int = 10_000) -> Sequence[ReadingEvent]:
        """Return the user's events at or after ``since``, oldest first.

        The oldest-first order over a bounded window is what the analytics
        aggregation (pace, streaks, pages-over-time) folds over.
        """
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.occurred_at >= since)
            .order_by(self.model.occurred_at.asc())
            .limit(limit)
        )
        return result.scalars().all()


__all__ = ["ReadingEventRepository", "ReadingProgressRepository"]
