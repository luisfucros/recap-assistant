"""Document lifecycle beyond upload: deletion and metadata updates.

Deletion spans three stores — Qdrant vectors, the stored original, and the
relational row (whose chunks cascade at the DB) — which cannot be made atomic
across systems. The order is chosen so a partial failure is safely retryable:
delete the external artifacts first (both idempotent), then the DB row last as
the authority. A crash between steps leaves a re-delete that converges.

All reads/writes go through a ``DocumentRepository`` already scoped to the
authenticated user, so a caller can only delete or update their own documents;
an unknown or unowned id surfaces as 404 (never revealing another user's data).
"""

import uuid

from loguru import logger

from shared.core.enums import Language
from shared.models.document import Document
from shared.providers.base import StorageProvider
from shared.repositories import DocumentRepository
from shared.vectorstore import ChunkVectorStore


class DocumentService:
    """Delete documents (with their vectors/object) and update their metadata."""

    def __init__(self, *, storage: StorageProvider, vector_store: ChunkVectorStore) -> None:
        """Wire the service to object storage and the chunk vector store."""
        self._storage = storage
        self._vectors = vector_store

    async def delete(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession; kept import-light at this layer
        documents: DocumentRepository,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Delete a document and all its artifacts (404 if not the caller's).

        Raises:
            NotFoundError: The document doesn't exist or isn't owned by the user.
        """
        document = await documents.get_or_404(document_id)
        # External artifacts first (both idempotent), then the row as the authority.
        await self._vectors.delete_by_document(user_id=user_id, document_id=document_id)
        await self._storage.delete(document.object_key)
        await documents.delete(document)
        await session.commit()
        logger.info("document.delete: deleted {}", document_id)

    async def update_language(
        self,
        *,
        session,  # noqa: ANN001 — AsyncSession
        documents: DocumentRepository,
        document_id: uuid.UUID,
        language: Language | None,
    ) -> Document:
        """Apply a partial metadata update (currently the language override).

        A ``None`` language leaves the value unchanged, so a PATCH with no fields
        is a no-op that still returns the current document.

        Raises:
            NotFoundError: The document doesn't exist or isn't owned by the user.
        """
        document = await documents.get_or_404(document_id)
        if language is not None:
            document.language = language
            logger.info("document.update_language: document {} -> {}", document_id, language.value)
        await session.commit()
        return document
