"""Re-embedding maintenance job (embedding provider switch).

When the embedding provider/model changes, existing vectors are stale. This job
re-embeds documents **from ``chunks.text``** — the durable source of truth — with
the now-active embedder, replaces their Qdrant points, and updates
``documents.embed_model``. Text is never re-parsed, so re-embedding is cheap
relative to full ingestion.

Two entry points:

* :func:`run_reembed` — re-embed one document into the existing collection
  (same-dimension switch, or a targeted refresh).
* :func:`reembed_all` — a full switch: rebuild the collection at the new
  dimension, then re-embed every indexed document into it. This reads across all
  users (a trusted system maintenance context, with no authenticated user), but
  each document's data stays scoped to its own ``user_id`` on write.
"""

import uuid

from loguru import logger
from sqlalchemy import select

from ingestion.base_task import AsyncTask
from ingestion.celery_app import app
from ingestion.resources import IngestionResources, get_ingestion_resources
from shared.core.enums import DocumentStatus
from shared.models.document import Document
from shared.repositories import ChunkRepository, DocumentRepository
from shared.vectorstore import ChunkVectorStore, build_chunk_payload


async def run_reembed(
    resources: IngestionResources, *, document_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Re-embed one indexed document's chunks and update its ``embed_model``.

    No-op if the document is missing or not ``indexed`` (only indexed documents
    have chunks to re-embed). Point ids are the chunk ids (stable), so re-embedding
    replaces vectors in place. Assumes the collection already matches the active
    embedder's dimension — a dimension change goes through :func:`reembed_all`.
    """
    settings = resources.settings
    log = logger.bind(document_id=str(document_id), user_id=str(user_id))
    async with resources.sessionmaker() as session:
        document = await DocumentRepository(session, user_id).get(document_id)
        if document is None or document.status is not DocumentStatus.INDEXED:
            log.debug("reembed: skipped (missing or not indexed)")
            return
        chunks = list(await ChunkRepository(session, user_id).list_by_document(document_id))
        title, author, language = document.title, document.author, document.language

    texts = [chunk.text for chunk in chunks]
    batch_size = settings.embed_batch_size
    embed_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
    log.info(
        "reembed: embedding {} chunks in {} batches (batch_size={})",
        len(texts),
        embed_batches,
        batch_size,
    )
    vectors = await resources.embedder.embed(texts, batch_size=batch_size) if texts else []
    log.info("reembed: re-embedded {} chunks (batch_size={})", len(texts), batch_size)

    store = ChunkVectorStore(
        resources.qdrant,
        collection=settings.qdrant_chunks_collection,
        dim=resources.embedder.dim,
    )
    await store.ensure_collection()
    await store.delete_by_document(user_id=user_id, document_id=document_id)
    if chunks:
        await store.upsert(
            ids=[chunk.vector_id for chunk in chunks],
            vectors=vectors,
            payloads=[
                build_chunk_payload(chunk, title=title, author=author, language=language)
                for chunk in chunks
            ],
        )

    async with resources.sessionmaker() as session:
        document = await DocumentRepository(session, user_id).get(document_id)
        if document is None:
            log.warning("reembed: document disappeared before embed_model stamp")
            return
        document.embed_model = settings.embedding_model
        await session.commit()
    log.info("reembed: document re-embedded with {}", settings.embedding_model)


async def reembed_all(resources: IngestionResources) -> int:
    """Rebuild the chunk collection at the active dimension and re-embed everything.

    Returns the number of documents re-embedded. Destructive: it drops and
    recreates the Qdrant collection (necessary when the vector dimension changed),
    so during the run documents are un-searchable until re-embedded.
    """
    settings = resources.settings
    store = ChunkVectorStore(
        resources.qdrant,
        collection=settings.qdrant_chunks_collection,
        dim=resources.embedder.dim,
    )
    await store.recreate()
    logger.warning(
        "reembed_all: chunk collection recreated; documents unsearchable until re-embedded"
    )

    async with resources.sessionmaker() as session:
        result = await session.execute(
            select(Document.id, Document.user_id).where(Document.status == DocumentStatus.INDEXED)
        )
        targets = result.all()
    logger.info("reembed_all: re-embedding {} documents", len(targets))

    for document_id, user_id in targets:
        await run_reembed(resources, document_id=document_id, user_id=user_id)
    logger.info("reembed_all: finished ({} documents re-embedded)", len(targets))
    return len(targets)


@app.task(base=AsyncTask, bind=True, name="ingestion.reembed_document")
def reembed_document(self: AsyncTask, document_id: str, user_id: str) -> None:
    """Re-embed a single document (enqueue after a provider switch)."""
    self.run_async(
        run_reembed(
            get_ingestion_resources(),
            document_id=uuid.UUID(document_id),
            user_id=uuid.UUID(user_id),
        )
    )


@app.task(base=AsyncTask, bind=True, name="ingestion.reembed_all_documents")
def reembed_all_documents(self: AsyncTask) -> int:
    """Full re-embed of the corpus (embedding provider/dimension switch)."""
    return self.run_async(reembed_all(get_ingestion_resources()))
