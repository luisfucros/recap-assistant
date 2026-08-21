"""Document routes: upload, list, detail, update, delete.

Handlers stay thin: they validate the upload at the boundary (type + size),
delegate the store-and-enqueue handoff to
:class:`~api.services.ingestion_service.IngestionService` and lifecycle actions
to :class:`~api.services.document_service.DocumentService`, and translate domain
errors to HTTP. All reads/writes go through a ``DocumentRepository`` already
scoped to the authenticated user, so a caller only ever sees their own
documents. Parsing, chunking, and embedding happen out-of-band in the ingestion
service — an upload returns immediately with ``status=pending``.
"""

import uuid

from fastapi import APIRouter, Query, UploadFile, status

from api.deps import (
    CurrentUser,
    DbSession,
    DocumentRepositoryDep,
    DocumentServiceDep,
    IngestionServiceDep,
    OutboxRepositoryDep,
    SettingsDep,
)
from api.errors import APIError
from api.schemas import DocumentPage, DocumentPublic, UpdateDocumentRequest
from api.services.ingestion_service import DocumentNotFailedError, DuplicateDocumentError
from api.uploads import (
    DocumentTooLargeError,
    UnsupportedDocumentFormatError,
    read_within_limit,
    resolve_format,
)
from shared.core.enums import DocumentFormat

router = APIRouter(prefix="/documents", tags=["documents"])


def _resolve_format_or_415(content_type: str | None) -> DocumentFormat:
    """Map the upload's content type to a format, or raise a 415."""
    try:
        return resolve_format(content_type)
    except UnsupportedDocumentFormatError as exc:
        raise APIError(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Unsupported document type; only PDF uploads are accepted.",
            code="UNSUPPORTED_MEDIA_TYPE",
        ) from exc


async def _read_or_413(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload within the size cap, or raise a 413."""
    try:
        return await read_within_limit(file.read, max_bytes=max_bytes)
    except DocumentTooLargeError as exc:
        raise APIError(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Upload exceeds the maximum size of {exc.limit_bytes} bytes.",
            code="PAYLOAD_TOO_LARGE",
        ) from exc


@router.post(
    "",
    response_model=DocumentPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    file: UploadFile,
    user: CurrentUser,
    ingestion: IngestionServiceDep,
    documents: DocumentRepositoryDep,
    outbox: OutboxRepositoryDep,
    session: DbSession,
    settings: SettingsDep,
) -> DocumentPublic:
    """Accept a PDF, store it, and queue ingestion; returns the ``pending`` document.

    Ingestion (parse/chunk/embed) runs asynchronously — poll the returned
    document until its ``status`` becomes ``indexed``. A re-upload of content the
    user already has returns ``409 DUPLICATE_DOCUMENT`` with a ``Location`` header
    pointing at the existing document. Rejects non-PDF types (415) and uploads
    over the size cap (413).
    """
    document_format = _resolve_format_or_415(file.content_type)
    data = await _read_or_413(file, settings.max_upload_bytes)
    try:
        document = await ingestion.upload(
            session=session,
            documents=documents,
            outbox=outbox,
            user_id=user.id,
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            document_format=document_format,
            data=data,
        )
    except DuplicateDocumentError as exc:
        raise APIError(
            status.HTTP_409_CONFLICT,
            "A document with identical content already exists.",
            code="DUPLICATE_DOCUMENT",
            headers={"Location": f"{settings.api_v1_prefix}/documents/{exc.existing_id}"},
        ) from exc
    return DocumentPublic.model_validate(document)


@router.post(
    "/{document_id}/retry",
    response_model=DocumentPublic,
    summary="Retry a failed document's ingestion",
)
async def retry_document(
    document_id: uuid.UUID,
    ingestion: IngestionServiceDep,
    documents: DocumentRepositoryDep,
    outbox: OutboxRepositoryDep,
    session: DbSession,
) -> DocumentPublic:
    """Re-queue ingestion for a ``failed`` document without re-uploading it.

    The stored original is reused as-is; only the document's state and outbox
    event are reset. 404 if the document isn't the caller's; 409 if it isn't
    currently ``failed`` (nothing to retry otherwise).
    """
    try:
        document = await ingestion.retry(
            session=session, documents=documents, outbox=outbox, document_id=document_id
        )
    except DocumentNotFailedError as exc:
        raise APIError(
            status.HTTP_409_CONFLICT,
            "Only a failed document can be retried.",
            code="DOCUMENT_NOT_FAILED",
        ) from exc
    return DocumentPublic.model_validate(document)


@router.get("", response_model=DocumentPage, summary="List your documents")
async def list_documents(
    documents: DocumentRepositoryDep,
    page: int = Query(1, ge=1, description="1-based page number."),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (max 100)."),
) -> DocumentPage:
    """Return a page of the caller's documents, newest first."""
    offset = (page - 1) * page_size
    items = await documents.list_recent(limit=page_size, offset=offset)
    total = await documents.count()
    return DocumentPage(
        items=[DocumentPublic.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentPublic, summary="Get one of your documents")
async def get_document(document_id: uuid.UUID, documents: DocumentRepositoryDep) -> DocumentPublic:
    """Return a single document by id (404 if it isn't the caller's)."""
    document = await documents.get_or_404(document_id)
    return DocumentPublic.model_validate(document)


@router.patch(
    "/{document_id}", response_model=DocumentPublic, summary="Update a document's metadata"
)
async def update_document(
    document_id: uuid.UUID,
    payload: UpdateDocumentRequest,
    document_service: DocumentServiceDep,
    documents: DocumentRepositoryDep,
    session: DbSession,
    user: CurrentUser,
) -> DocumentPublic:
    """Partially update a document (currently the ``language`` override).

    Only provided fields change; 404 if the document isn't the caller's.
    """
    document = await document_service.update_language(
        session=session,
        documents=documents,
        document_id=document_id,
        language=payload.language,
    )
    return DocumentPublic.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all its data",
)
async def delete_document(
    document_id: uuid.UUID,
    document_service: DocumentServiceDep,
    documents: DocumentRepositoryDep,
    session: DbSession,
    user: CurrentUser,
) -> None:
    """Delete a document with its chunks, vectors, and stored original.

    Idempotent from the client's view for a given id only while it exists; a
    second delete returns 404. 404 if the document isn't the caller's.
    """
    await document_service.delete(
        session=session,
        documents=documents,
        user_id=user.id,
        document_id=document_id,
    )
