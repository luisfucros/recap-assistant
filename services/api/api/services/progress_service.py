"""Reading-state business logic: positions, statuses, and the analytics trail.

:class:`ProgressService` is the single writer of :class:`ReadingProgress` and the
append-only :class:`ReadingEvent` trail. It owns three things the rest of the app
relies on:

* **Position of record** — get-or-create the per-``(user, document)`` progress
  row and advance ``current_page``; retrieval later defaults to
  ``page_end <= current_page`` from here.
* **Status lifecycle** — ``not_started → reading → completed/cancelled`` with
  auto-promotion (recording a page starts *reading*; reaching the last page marks
  *completed*) and explicit overrides (cancel/reopen).
* **Event emission** — every forward move or status change appends a
  :class:`ReadingEvent` so pace/streaks/history stay derivable and auditable
  (FR-17); the mutable progress row stays a cheap read/update path.

Ownership is enforced end-to-end: the target document is loaded through a
user-scoped :class:`DocumentRepository` (404 if not the caller's), and progress /
event rows are written through user-scoped repositories, so a caller can only
ever touch their own reading state — the ``user_id`` never comes from client
input.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from loguru import logger

from shared.core.enums import ReadingEventType, ReadingStatus
from shared.core.errors import InvalidInputError, NotFoundError
from shared.models.reading import ReadingEvent, ReadingProgress
from shared.repositories import (
    DocumentRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
)


def _now() -> datetime:
    """Current UTC time (isolated so tests can patch reading-time cheaply)."""
    return datetime.now(UTC)


class _Unset:
    """Sentinel distinguishing an omitted field from an explicit ``None``.

    Used for the per-document spoiler-safe override: ``None`` clears it (defer to
    the user default) while omitting the argument leaves the stored value alone.
    """


_UNSET = _Unset()


class ProgressService:
    """Read and update per-document reading progress, emitting analytics events."""

    async def get_progress(
        self, *, progress: ReadingProgressRepository, document_id: uuid.UUID
    ) -> ReadingProgress | None:
        """Return the user's progress for a document, or ``None`` if untracked."""
        return await progress.get_by_document(document_id)

    async def reading_list(
        self, *, progress: ReadingProgressRepository
    ) -> dict[ReadingStatus, Sequence[ReadingProgress]]:
        """Group the user's tracked documents by reading status.

        Returns one entry per active status (``reading``/``completed``/
        ``cancelled``), each most-recently-read first — the shape the reading-list
        UI renders as sections. ``not_started`` is omitted: a row only exists once
        the user has interacted with the document.
        """
        return {
            status: await progress.list_by_status(status)
            for status in (
                ReadingStatus.READING,
                ReadingStatus.COMPLETED,
                ReadingStatus.CANCELLED,
            )
        }

    async def recently_accessed(
        self, *, progress: ReadingProgressRepository, limit: int = 10
    ) -> Sequence[ReadingProgress]:
        """Return the user's most-recently-accessed documents, newest first."""
        return await progress.list_recent(limit=limit)

    async def update_progress(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession; kept import-light at this layer
        documents: DocumentRepository,
        progress: ReadingProgressRepository,
        events: ReadingEventRepository,
        document_id: uuid.UUID,
        current_page: int | None = None,
        status: ReadingStatus | None = None,
        spoiler_safe: bool | None | _Unset = _UNSET,
    ) -> ReadingProgress:
        """Apply a combined position/status/spoiler update in one transaction.

        Get-or-creates the progress row and changes only what's supplied. Setting
        ``current_page`` re-resolves the status (auto-promotion to *reading* /
        *completed*) unless an explicit ``status`` overrides it; passing ``status``
        alone changes it without moving the page. A *forward* page move appends a
        ``POSITION_ADVANCED`` event and a status change appends a status/completed
        event — the spoiler override is config, not activity, so it emits nothing.
        ``spoiler_safe`` uses a sentinel: ``None`` clears the per-document override
        (defer to the user default), while omitting it leaves the stored value.

        Raises:
            NotFoundError: The document doesn't exist or isn't the caller's.
            InvalidInputError: ``current_page`` is negative or past the last page.
        """
        document = await documents.get_or_404(document_id)
        row = await self._get_or_create(progress, document_id)
        previous_page, previous_status = row.current_page, row.status

        if current_page is not None:
            self._validate_page(current_page, document.page_count)
            row.current_page = current_page
            row.status = self._resolve_status(
                explicit=status,
                current_page=current_page,
                previous=previous_status,
                page_count=document.page_count,
            )
        elif status is not None:
            row.status = status

        if not isinstance(spoiler_safe, _Unset):
            row.spoiler_safe = spoiler_safe

        row.last_accessed_at = _now()
        await self._emit(
            events,
            document_id=document_id,
            previous_page=previous_page,
            current_page=row.current_page,
            previous_status=previous_status,
            new_status=row.status,
        )
        await session.commit()
        logger.info(
            "progress.update: document {} page {}->{} status {}->{}",
            document_id,
            previous_page,
            row.current_page,
            previous_status.value,
            row.status.value,
        )
        return row

    async def record_position(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession
        documents: DocumentRepository,
        progress: ReadingProgressRepository,
        events: ReadingEventRepository,
        document_id: uuid.UUID,
        current_page: int,
        status: ReadingStatus | None = None,
    ) -> ReadingProgress:
        """Set the user's current page (and optionally status) for a document.

        A focused wrapper over :meth:`update_progress`: only a *forward* move emits
        a ``POSITION_ADVANCED`` event.

        Raises:
            NotFoundError: The document doesn't exist or isn't the caller's.
            InvalidInputError: ``current_page`` is negative or past the last page.
        """
        return await self.update_progress(
            session=session,
            documents=documents,
            progress=progress,
            events=events,
            document_id=document_id,
            current_page=current_page,
            status=status,
        )

    async def set_status(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession
        documents: DocumentRepository,
        progress: ReadingProgressRepository,
        events: ReadingEventRepository,
        document_id: uuid.UUID,
        status: ReadingStatus,
    ) -> ReadingProgress:
        """Change a document's reading status without moving the page.

        A focused wrapper over :meth:`update_progress` for actions like *cancel* or
        *mark complete* / *reopen*; keeps the current page as-is.

        Raises:
            NotFoundError: The document doesn't exist or isn't the caller's.
        """
        return await self.update_progress(
            session=session,
            documents=documents,
            progress=progress,
            events=events,
            document_id=document_id,
            status=status,
        )

    async def advance_summarized_page(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession
        progress: ReadingProgressRepository,
        document_id: uuid.UUID,
        page: int,
    ) -> ReadingProgress:
        """Advance the recap high-water mark (``last_summarized_page``).

        Called by the progress→summary→memory loop after a span is summarized. The
        mark only ever moves forward — a smaller ``page`` is a no-op — so a
        re-run never re-summarizes already-covered pages (FR-3.1).

        Raises:
            NotFoundError: No progress row exists for the document.
        """
        row = await progress.get_by_document(document_id)
        if row is None:
            raise NotFoundError()
        row.last_summarized_page = max(row.last_summarized_page, page)
        row.last_accessed_at = _now()
        await session.commit()
        logger.info(
            "progress.advance_summarized_page: document {} -> page {}",
            document_id,
            row.last_summarized_page,
        )
        return row

    # --- internals --------------------------------------------------------- #

    @staticmethod
    def _validate_page(current_page: int, page_count: int | None) -> None:
        """Reject a page that is negative or beyond the document's length."""
        if current_page < 0:
            raise InvalidInputError("current_page must be non-negative.")
        if page_count is not None and current_page > page_count:
            raise InvalidInputError(
                f"current_page {current_page} is past the last page ({page_count})."
            )

    async def _get_or_create(
        self, progress: ReadingProgressRepository, document_id: uuid.UUID
    ) -> ReadingProgress:
        """Return the existing progress row or insert a fresh ``not_started`` one.

        Initial page/summary/status are set explicitly (not left to the column
        defaults, which only populate at flush) so the in-memory row is fully
        defined the moment it's created — the caller reads ``current_page`` and
        ``status`` before the transaction commits.
        """
        row = await progress.get_by_document(document_id)
        if row is not None:
            return row
        return await progress.add(
            ReadingProgress(
                user_id=progress.user_id,
                document_id=document_id,
                current_page=0,
                last_summarized_page=0,
                status=ReadingStatus.NOT_STARTED,
            )
        )

    @staticmethod
    def _resolve_status(
        *,
        explicit: ReadingStatus | None,
        current_page: int,
        previous: ReadingStatus,
        page_count: int | None,
    ) -> ReadingStatus:
        """Decide the resulting status from an optional override and the position.

        An explicit status always wins (cancel/reopen/complete). Otherwise the
        position auto-promotes: reaching the last page marks ``completed``, and
        recording any page from ``not_started`` starts ``reading``; every other
        state is left unchanged.
        """
        if explicit is not None:
            return explicit
        if page_count is not None and current_page >= page_count and current_page > 0:
            return ReadingStatus.COMPLETED
        if current_page > 0 and previous == ReadingStatus.NOT_STARTED:
            return ReadingStatus.READING
        return previous

    async def _emit(
        self,
        events: ReadingEventRepository,
        *,
        document_id: uuid.UUID,
        previous_page: int,
        current_page: int,
        previous_status: ReadingStatus,
        new_status: ReadingStatus,
    ) -> None:
        """Append reading events for a forward move and/or a status change.

        A forward page move yields ``POSITION_ADVANCED``; a transition into
        ``completed`` yields the dedicated ``COMPLETED`` event, and any other
        status change yields ``STATUS_CHANGED``. No-op moves emit nothing, so the
        trail carries only real activity.
        """
        if current_page > previous_page:
            await events.add(
                ReadingEvent(
                    user_id=events.user_id,
                    document_id=document_id,
                    type=ReadingEventType.POSITION_ADVANCED,
                    from_page=previous_page,
                    to_page=current_page,
                )
            )
        if new_status != previous_status:
            event_type = (
                ReadingEventType.COMPLETED
                if new_status == ReadingStatus.COMPLETED
                else ReadingEventType.STATUS_CHANGED
            )
            await events.add(
                ReadingEvent(
                    user_id=events.user_id,
                    document_id=document_id,
                    type=event_type,
                    from_page=None,
                    to_page=current_page,
                )
            )
