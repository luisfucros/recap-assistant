"""Reading-progress routes: list, get, and update per-document reading state.

Handlers stay thin: they delegate to
:class:`~api.services.progress_service.ProgressService` and translate an untracked
document to a 404. Every repository is already scoped to the authenticated user
(the owner comes from the access-token cookie, never the path/body), so a caller
can only ever see or change their own reading state.
"""

import uuid

from fastapi import APIRouter

from api.deps import (
    DbSession,
    DocumentRepositoryDep,
    ProgressRepositoryDep,
    ProgressServiceDep,
    ReadingEventRepositoryDep,
)
from api.schemas import ReadingListResponse, ReadingProgressPublic, UpdateProgressRequest
from shared.core.enums import ReadingStatus
from shared.core.errors import NotFoundError

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("", response_model=ReadingListResponse, summary="List your reading, grouped by status")
async def list_progress(
    progress_service: ProgressServiceDep, progress: ProgressRepositoryDep
) -> ReadingListResponse:
    """Return the caller's tracked documents grouped into reading/completed/cancelled.

    Each group is ordered most-recently-accessed first. Documents never opened
    have no progress row and so appear in no group.
    """
    grouped = await progress_service.reading_list(progress=progress)
    return ReadingListResponse(
        reading=[ReadingProgressPublic.model_validate(r) for r in grouped[ReadingStatus.READING]],
        completed=[
            ReadingProgressPublic.model_validate(r) for r in grouped[ReadingStatus.COMPLETED]
        ],
        cancelled=[
            ReadingProgressPublic.model_validate(r) for r in grouped[ReadingStatus.CANCELLED]
        ],
    )


@router.get(
    "/{document_id}",
    response_model=ReadingProgressPublic,
    summary="Get your reading state for a document",
)
async def get_progress(
    document_id: uuid.UUID,
    progress_service: ProgressServiceDep,
    progress: ProgressRepositoryDep,
) -> ReadingProgressPublic:
    """Return the caller's reading state for a document.

    A 404 means the document is untracked (never opened) or isn't the caller's —
    the two are deliberately indistinguishable so existence never leaks. A client
    may treat 404 as "not started".
    """
    row = await progress_service.get_progress(progress=progress, document_id=document_id)
    if row is None:
        raise NotFoundError()
    return ReadingProgressPublic.model_validate(row)


@router.put(
    "/{document_id}",
    response_model=ReadingProgressPublic,
    summary="Update your reading state for a document",
)
async def update_progress(
    document_id: uuid.UUID,
    payload: UpdateProgressRequest,
    progress_service: ProgressServiceDep,
    documents: DocumentRepositoryDep,
    progress: ProgressRepositoryDep,
    events: ReadingEventRepositoryDep,
    session: DbSession,
) -> ReadingProgressPublic:
    """Set position, status, and/or the per-document spoiler-safe override.

    Only fields present in the body change: ``current_page`` moves the position
    (auto-promoting status), ``status`` overrides it, and an explicit
    ``spoiler_safe`` (including ``null`` to clear) sets the per-document override.
    404 if the document isn't the caller's; 422 if the page is past the last page.
    """
    provided = payload.model_dump(exclude_unset=True)
    spoiler_kwargs = {"spoiler_safe": payload.spoiler_safe} if "spoiler_safe" in provided else {}
    row = await progress_service.update_progress(
        session=session,
        documents=documents,
        progress=progress,
        events=events,
        document_id=document_id,
        current_page=payload.current_page,
        status=payload.status,
        **spoiler_kwargs,
    )
    return ReadingProgressPublic.model_validate(row)
