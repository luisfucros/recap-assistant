"""The API service's ingestion entry point: validate, store, enqueue — no more.

This service does the *handoff* only. It computes the content hash, stores the
original in content-addressed object storage, and — in **one DB transaction** —
inserts a ``pending`` ``documents`` row plus a ``document.uploaded`` outbox
event. It deliberately does **no** parsing, chunking, or embedding: that is the
separate ingestion (Celery) service's job, triggered by the outbox event. The
two services never call each other. :meth:`IngestionService.retry` re-emits that
same handoff for an already-stored, ``failed`` document, so a user never has to
re-upload identical bytes just to try ingestion again.

Two invariants are enforced here:

* **Per-user duplicate rejection, race-safe.** A user re-uploading identical
  bytes gets :class:`DuplicateDocumentError` carrying the existing document id.
  A fast pre-check handles the common case; the unique ``(user_id,
  content_sha256)`` constraint is the authority under concurrency — if two
  identical uploads race past the pre-check, the DB lets one commit and the
  other's ``IntegrityError`` is translated to the same duplicate result. The
  loser's stored object is byte-identical at the same content-addressed key, so
  it is a harmless idempotent overwrite, never an orphan.
* **Outbox atomicity.** The row and its event commit together, so a committed
  document always has exactly one pending ingestion event (and vice versa).
"""

import uuid

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.enums import DocumentFormat, DocumentStatus
from shared.core.events import DOCUMENT_UPLOADED
from shared.ingestion_core.content_address import object_key, sha256_hexdigest
from shared.models.document import Document
from shared.providers.base import StorageProvider
from shared.repositories import DocumentRepository, OutboxRepository


class DuplicateDocumentError(Exception):
    """The user already has a document with identical content.

    Carries the existing document's id so the API can point the client at it.
    """

    def __init__(self, existing_id: uuid.UUID) -> None:
        self.existing_id = existing_id
        super().__init__(f"document already exists: {existing_id}")


class DocumentNotFailedError(Exception):
    """A retry was requested for a document that isn't ``failed``.

    Only a ``failed`` document has anything to retry — ``pending``/``processing``
    already has ingestion in flight (or about to be, via the stuck-document
    sweep), and ``indexed`` has nothing left to do.
    """

    def __init__(self, document_id: uuid.UUID) -> None:
        self.document_id = document_id
        super().__init__(f"document is not failed, cannot retry: {document_id}")


class IngestionService:
    """Validate-store-enqueue handoff for uploaded documents (API side)."""

    def __init__(self, *, storage: StorageProvider, embed_model: str) -> None:
        """Wire the service to object storage and the active embedder's model id.

        Args:
            storage: Where the original bytes are stored (content-addressed).
            embed_model: Identifier of the embedder that will produce this
                document's vectors, recorded so a later provider switch can tell
                which documents need re-embedding.
        """
        self._storage = storage
        self._embed_model = embed_model

    async def upload(
        self,
        *,
        session: AsyncSession,
        documents: DocumentRepository,
        outbox: OutboxRepository,
        user_id: uuid.UUID,
        filename: str,
        content_type: str,
        document_format: DocumentFormat,
        data: bytes,
    ) -> Document:
        """Store an upload and enqueue its ingestion; return the ``pending`` row.

        The ``documents`` repository is already scoped to ``user_id`` (bound from
        the authenticated context, never from the request body), so the write
        can only land under the caller's ownership.

        Raises:
            DuplicateDocumentError: The user already has this content.
        """
        content_sha256 = sha256_hexdigest(data)

        # Fast path: skip the storage write when we already hold this content.
        existing = await documents.get_by_content_sha256(content_sha256)
        if existing is not None:
            raise DuplicateDocumentError(existing.id)

        key = object_key(user_id, content_sha256, document_format.value)
        await self._storage.put(key, data, content_type)

        document = Document(
            user_id=user_id,
            filename=filename,
            object_key=key,
            content_sha256=content_sha256,
            format=document_format,
            status=DocumentStatus.PENDING,
            embed_model=self._embed_model,
        )
        try:
            await documents.add(document)
            await outbox.add(
                event_type=DOCUMENT_UPLOADED,
                aggregate_id=document.id,
                payload={"document_id": str(document.id), "user_id": str(user_id)},
            )
            await session.commit()
        except IntegrityError as exc:
            # Lost an upload race on the unique constraint: roll back and return
            # the winner's id, so concurrent identical uploads behave like the
            # sequential duplicate case.
            await session.rollback()
            winner = await documents.get_by_content_sha256(content_sha256)
            if winner is None:  # pragma: no cover — constraint fired, row must exist
                raise
            logger.info(
                "ingestion.upload: lost duplicate-upload race for user {}, existing document {}",
                user_id,
                winner.id,
            )
            raise DuplicateDocumentError(winner.id) from exc
        logger.info(
            "ingestion.upload: document {} stored and queued for user {}", document.id, user_id
        )
        return document

    async def retry(
        self,
        *,
        session: AsyncSession,
        documents: DocumentRepository,
        outbox: OutboxRepository,
        document_id: uuid.UUID,
    ) -> Document:
        """Re-queue ingestion for a ``failed`` document, without re-uploading it.

        The original bytes are already in content-addressed object storage (a
        failed ingestion never deletes them), so a retry only needs to reset the
        document's state and re-emit the ``document.uploaded`` outbox event — the
        pipeline itself is idempotent and always reruns from scratch, discarding
        any partial chunks/vectors the failed attempt left behind.

        Raises:
            DocumentNotFailedError: The document isn't currently ``failed``.
        """
        document = await documents.get_or_404(document_id)
        if document.status is not DocumentStatus.FAILED:
            raise DocumentNotFailedError(document.id)
        document.status = DocumentStatus.PENDING
        document.failure_reason = None
        await outbox.add(
            event_type=DOCUMENT_UPLOADED,
            aggregate_id=document.id,
            payload={"document_id": str(document.id), "user_id": str(document.user_id)},
        )
        await session.commit()
        logger.info(
            "ingestion.retry: document {} re-queued for user {}", document.id, document.user_id
        )
        return document
