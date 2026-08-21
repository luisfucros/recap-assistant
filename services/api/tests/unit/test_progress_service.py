"""Unit tests for the reading ProgressService (positions, status, events).

Repositories and the session are faked at the boundary; the status-resolution,
high-water-mark, validation, and event-emission logic under test is real.
"""

import uuid

import pytest
from api.services.progress_service import ProgressService

from shared.core.enums import DocumentFormat, ReadingEventType, ReadingStatus
from shared.core.errors import InvalidInputError, NotFoundError
from shared.models.document import Document
from shared.models.reading import ReadingEvent, ReadingProgress

pytestmark = pytest.mark.unit


class _FakeDocRepo:
    def __init__(self, document: Document | None) -> None:
        self._document = document

    async def get_or_404(self, document_id: uuid.UUID) -> Document:
        if self._document is None or self._document.id != document_id:
            raise NotFoundError()
        return self._document


class _FakeProgressRepo:
    def __init__(self, owner: uuid.UUID, existing: ReadingProgress | None = None) -> None:
        self._owner = owner
        self._existing = existing
        self.added: list[ReadingProgress] = []

    @property
    def user_id(self) -> uuid.UUID:
        return self._owner

    async def get_by_document(self, document_id: uuid.UUID) -> ReadingProgress | None:
        return self._existing

    async def add(self, entity: ReadingProgress) -> ReadingProgress:
        if entity.user_id != self._owner:
            raise ValueError("entity.user_id does not match the repository's owner")
        self.added.append(entity)
        self._existing = entity
        return entity


class _FakeEventRepo:
    def __init__(self, owner: uuid.UUID) -> None:
        self._owner = owner
        self.added: list[ReadingEvent] = []

    @property
    def user_id(self) -> uuid.UUID:
        return self._owner

    async def add(self, entity: ReadingEvent) -> ReadingEvent:
        if entity.user_id != self._owner:
            raise ValueError("entity.user_id does not match the repository's owner")
        self.added.append(entity)
        return entity


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


def _document(user_id: uuid.UUID, *, page_count: int | None = 100) -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/abc.pdf",
        content_sha256="abc",
        format=DocumentFormat.PDF,
        embed_model="m",
        page_count=page_count,
    )


def _existing_progress(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    current_page: int,
    status: ReadingStatus,
    last_summarized_page: int = 0,
) -> ReadingProgress:
    return ReadingProgress(
        id=uuid.uuid4(),
        user_id=user_id,
        document_id=document_id,
        current_page=current_page,
        last_summarized_page=last_summarized_page,
        status=status,
    )


def _types(events: list[ReadingEvent]) -> list[ReadingEventType]:
    return [e.type for e in events]


# --- record_position ----------------------------------------------------- #


async def test_first_position_creates_row_promotes_to_reading_and_emits_events() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    progress, events, session = _FakeProgressRepo(owner), _FakeEventRepo(owner), _FakeSession()
    service = ProgressService()

    row = await service.record_position(
        session=session,
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        current_page=10,
    )

    assert row.current_page == 10
    assert row.status is ReadingStatus.READING
    assert len(progress.added) == 1  # a fresh row was created
    assert _types(events.added) == [
        ReadingEventType.POSITION_ADVANCED,
        ReadingEventType.STATUS_CHANGED,
    ]
    advanced = events.added[0]
    assert (advanced.from_page, advanced.to_page) == (0, 10)
    assert session.commits == 1


async def test_reaching_last_page_auto_completes() -> None:
    owner = uuid.uuid4()
    doc = _document(owner, page_count=50)
    existing = _existing_progress(owner, doc.id, current_page=40, status=ReadingStatus.READING)
    progress, events, session = (
        _FakeProgressRepo(owner, existing),
        _FakeEventRepo(owner),
        _FakeSession(),
    )

    row = await ProgressService().record_position(
        session=session,
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        current_page=50,
    )

    assert row.status is ReadingStatus.COMPLETED
    # The completion transition uses the dedicated COMPLETED event type.
    assert ReadingEventType.COMPLETED in _types(events.added)
    assert ReadingEventType.STATUS_CHANGED not in _types(events.added)


async def test_backward_move_emits_no_position_event_but_still_commits() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    existing = _existing_progress(owner, doc.id, current_page=30, status=ReadingStatus.READING)
    progress, events, session = (
        _FakeProgressRepo(owner, existing),
        _FakeEventRepo(owner),
        _FakeSession(),
    )

    row = await ProgressService().record_position(
        session=session,
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        current_page=20,
    )

    assert row.current_page == 20
    assert events.added == []  # no forward move, no status change
    assert session.commits == 1


async def test_explicit_status_overrides_inference() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    existing = _existing_progress(owner, doc.id, current_page=10, status=ReadingStatus.READING)
    progress, events = _FakeProgressRepo(owner, existing), _FakeEventRepo(owner)

    row = await ProgressService().record_position(
        session=_FakeSession(),
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        current_page=10,
        status=ReadingStatus.CANCELLED,
    )

    assert row.status is ReadingStatus.CANCELLED
    assert _types(events.added) == [ReadingEventType.STATUS_CHANGED]


async def test_negative_page_rejected() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    with pytest.raises(InvalidInputError):
        await ProgressService().record_position(
            session=_FakeSession(),
            documents=_FakeDocRepo(doc),
            progress=_FakeProgressRepo(owner),
            events=_FakeEventRepo(owner),
            document_id=doc.id,
            current_page=-1,
        )


async def test_page_past_end_rejected() -> None:
    owner = uuid.uuid4()
    doc = _document(owner, page_count=100)
    with pytest.raises(InvalidInputError):
        await ProgressService().record_position(
            session=_FakeSession(),
            documents=_FakeDocRepo(doc),
            progress=_FakeProgressRepo(owner),
            events=_FakeEventRepo(owner),
            document_id=doc.id,
            current_page=101,
        )


async def test_record_position_on_foreign_document_is_404() -> None:
    owner = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await ProgressService().record_position(
            session=_FakeSession(),
            documents=_FakeDocRepo(None),
            progress=_FakeProgressRepo(owner),
            events=_FakeEventRepo(owner),
            document_id=uuid.uuid4(),
            current_page=5,
        )


# --- set_status ---------------------------------------------------------- #


async def test_set_status_cancel_keeps_page_and_emits_status_change() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    existing = _existing_progress(owner, doc.id, current_page=42, status=ReadingStatus.READING)
    progress, events, session = (
        _FakeProgressRepo(owner, existing),
        _FakeEventRepo(owner),
        _FakeSession(),
    )

    row = await ProgressService().set_status(
        session=session,
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        status=ReadingStatus.CANCELLED,
    )

    assert row.status is ReadingStatus.CANCELLED
    assert row.current_page == 42  # unchanged
    assert _types(events.added) == [ReadingEventType.STATUS_CHANGED]
    assert session.commits == 1


# --- advance_summarized_page (high-water mark) --------------------------- #


async def test_advance_summarized_page_only_moves_forward() -> None:
    owner = uuid.uuid4()
    doc_id = uuid.uuid4()
    existing = _existing_progress(
        owner, doc_id, current_page=80, status=ReadingStatus.READING, last_summarized_page=50
    )
    progress, session = _FakeProgressRepo(owner, existing), _FakeSession()

    row = await ProgressService().advance_summarized_page(
        session=session, progress=progress, document_id=doc_id, page=70
    )
    assert row.last_summarized_page == 70

    # A lower page is a no-op (never regresses the high-water mark).
    row = await ProgressService().advance_summarized_page(
        session=session, progress=progress, document_id=doc_id, page=30
    )
    assert row.last_summarized_page == 70


async def test_advance_summarized_page_without_progress_is_404() -> None:
    owner = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await ProgressService().advance_summarized_page(
            session=_FakeSession(),
            progress=_FakeProgressRepo(owner),
            document_id=uuid.uuid4(),
            page=10,
        )


# --- reading_list -------------------------------------------------------- #


async def test_reading_list_groups_by_active_statuses() -> None:
    class _GroupingRepo:
        def __init__(self) -> None:
            self.queried: list[ReadingStatus] = []

        async def list_by_status(self, status, *, limit=100, offset=0):
            self.queried.append(status)
            return [status]  # stand-in payload keyed by status

    repo = _GroupingRepo()
    grouped = await ProgressService().reading_list(progress=repo)  # type: ignore[arg-type]

    assert set(grouped) == {
        ReadingStatus.READING,
        ReadingStatus.COMPLETED,
        ReadingStatus.CANCELLED,
    }
    assert ReadingStatus.NOT_STARTED not in grouped
    assert repo.queried == [
        ReadingStatus.READING,
        ReadingStatus.COMPLETED,
        ReadingStatus.CANCELLED,
    ]


# --- update_progress composition (position + status + spoiler override) --- #


async def test_update_progress_sets_page_status_and_spoiler_in_one_commit() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    progress, events, session = _FakeProgressRepo(owner), _FakeEventRepo(owner), _FakeSession()

    row = await ProgressService().update_progress(
        session=session,
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=events,
        document_id=doc.id,
        current_page=15,
        spoiler_safe=False,
    )

    assert row.current_page == 15
    assert row.status is ReadingStatus.READING  # auto-promoted from not_started
    assert row.spoiler_safe is False
    assert session.commits == 1


async def test_update_progress_spoiler_none_clears_override() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    existing = _existing_progress(owner, doc.id, current_page=10, status=ReadingStatus.READING)
    existing.spoiler_safe = True  # a per-doc override is set
    progress = _FakeProgressRepo(owner, existing)

    row = await ProgressService().update_progress(
        session=_FakeSession(),
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=_FakeEventRepo(owner),
        document_id=doc.id,
        spoiler_safe=None,  # explicit None clears the override
    )

    assert row.spoiler_safe is None


async def test_update_progress_omitting_spoiler_leaves_it_unchanged() -> None:
    owner = uuid.uuid4()
    doc = _document(owner)
    existing = _existing_progress(owner, doc.id, current_page=10, status=ReadingStatus.READING)
    existing.spoiler_safe = True
    progress = _FakeProgressRepo(owner, existing)

    row = await ProgressService().update_progress(
        session=_FakeSession(),
        documents=_FakeDocRepo(doc),
        progress=progress,
        events=_FakeEventRepo(owner),
        document_id=doc.id,
        current_page=20,  # no spoiler_safe arg → sentinel → unchanged
    )

    assert row.spoiler_safe is True
    assert row.current_page == 20
