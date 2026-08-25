"""The ingestion pipeline: turn an uploaded document into indexed chunks+vectors.

This is the ingestion service's core work, invoked by the Celery ``ingest_document``
task. It fetches the stored original, parses it, detects its language, chunks it,
embeds the chunks in bounded batches, upserts vectors to Qdrant, and finally
persists the chunks and flips the document to ``indexed`` — all guarding two
invariants from the design:

* **Atomic, outbox-protected terminal status (FR-1.7.1).** The ``chunks`` insert,
  the ``status=indexed`` transition, and the ``document.indexed`` outbox event
  commit **together in one transaction**, and only *after* the Qdrant upsert has
  succeeded. A connection failure mid-run therefore never leaves a document
  showing ``indexed`` with missing vectors, and never emits the downstream event
  for a partial result.
* **Idempotency.** A re-run (Celery re-delivery, or a retry after a partial
  failure) deletes the document's existing vectors and chunks before writing the
  fresh set, so processing the same document twice converges to one clean result.

The worker runs as a trusted system context, but still scopes every write by the
``user_id`` carried in the (self-produced) outbox payload, so per-user isolation
holds end-to-end.
"""

import uuid

from loguru import logger
from sqlalchemy import func

from ingestion.resources import IngestionResources
from shared.core.enums import DocumentStatus
from shared.core.events import DOCUMENT_INDEXED
from shared.ingestion_core.chunking import ChunkData, chunk_document
from shared.ingestion_core.language import detect_language
from shared.ingestion_core.parsing import ParserFactory
from shared.models.document import Chunk
from shared.observability import observe_document_chunks, record_document_ingested
from shared.repositories import ChunkRepository, DocumentRepository, OutboxRepository
from shared.vectorstore import ChunkVectorStore, build_chunk_payload, chunk_point_id

# Cap on the stored failure reason so a huge exception message can't bloat the row.
_MAX_FAILURE_REASON = 2000


async def run_ingestion(
    resources: IngestionResources, *, document_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Run the full pipeline for one document (see module docstring for guarantees).

    Safe to call more than once for the same document: an already-``indexed``
    document is skipped, and a re-run replaces any prior chunks/vectors.
    """
    settings = resources.settings
    # Bound to every log emitted for this run, so a document's whole ingestion —
    # across all four phases below — is correlatable in the worker's logs without
    # threading the ids through each helper explicitly.
    log = logger.bind(document_id=str(document_id), user_id=str(user_id))

    # 1) Claim the work: load the document and mark it processing. Skip if it is
    #    gone (deleted) or already indexed (a duplicate delivery).
    async with resources.sessionmaker() as session:
        documents = DocumentRepository(session, user_id)
        document = await documents.get(document_id)
        if document is None or document.status is DocumentStatus.INDEXED:
            log.debug("ingestion: skipped (missing or already indexed)")
            return
        document.status = DocumentStatus.PROCESSING
        document.failure_reason = None
        await session.commit()
        # expire_on_commit=False keeps these readable after the session closes.
        object_key = document.object_key
        document_format = document.format
    log.info("ingestion: started")

    # 2) Heavy work, holding no DB transaction: fetch → parse → detect → chunk →
    #    embed. Each phase is a trace span (no-op unless Langfuse is configured).
    tracer = resources.tracer
    data = await resources.storage.get(object_key)
    with tracer.span("parse", document_id=str(document_id)):
        parsed = ParserFactory().for_format(document_format).parse(data)
    log.debug("ingestion: parsed ({} pages)", parsed.page_count)
    with tracer.span("detect_language"):
        language = detect_language(parsed.full_text(), default=settings.default_language)
    log.debug("ingestion: detected language {}", language)
    with tracer.span("chunk"):
        chunk_datas = chunk_document(
            parsed,
            chunk_size_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
        )
        chunk_rows = _build_chunk_rows(chunk_datas, document_id=document_id, user_id=user_id)
    log.debug("ingestion: chunked into {} chunks", len(chunk_rows))
    texts = [row.text for row in chunk_rows]
    batch_size = settings.embed_batch_size
    embed_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
    log.info(
        "ingestion: embedding {} chunks in {} batches (batch_size={})",
        len(texts),
        embed_batches,
        batch_size,
    )
    with tracer.span("embed", chunks=len(texts)):
        vectors = await resources.embedder.embed(texts, batch_size=batch_size) if texts else []
    log.info("ingestion: embedded {} chunks (batch_size={})", len(texts), batch_size)

    # 3) Upsert vectors BEFORE the terminal DB commit. Clearing first makes the
    #    upsert idempotent across retries/partial runs.
    store = ChunkVectorStore(
        resources.qdrant,
        collection=settings.qdrant_chunks_collection,
        dim=resources.embedder.dim,
    )
    with tracer.span("qdrant_upsert", points=len(chunk_rows)):
        await store.ensure_collection()
        await store.delete_by_document(user_id=user_id, document_id=document_id)
        if chunk_rows:
            await store.upsert(
                ids=[row.vector_id for row in chunk_rows],
                vectors=vectors,
                payloads=[
                    build_chunk_payload(
                        row, title=parsed.title, author=parsed.author, language=language
                    )
                    for row in chunk_rows
                ],
            )
    log.debug("ingestion: upserted {} vectors", len(chunk_rows))

    # 4) Terminal transaction: replace chunks, flip to indexed, record metadata,
    #    and emit the downstream event — atomically, only now that vectors exist.
    with tracer.span("persist", chunks=len(chunk_rows)):
        async with resources.sessionmaker() as session:
            chunks = ChunkRepository(session, user_id)
            await chunks.delete_by_document(document_id)
            if chunk_rows:
                await chunks.add_many(chunk_rows)

            documents = DocumentRepository(session, user_id)
            document = await documents.get(document_id)
            if document is None:
                # Deleted mid-flight: undo the vectors we just wrote and stop.
                await store.delete_by_document(user_id=user_id, document_id=document_id)
                log.warning("ingestion: document deleted mid-flight; rolled back vectors")
                return
            document.status = DocumentStatus.INDEXED
            document.language = language
            document.title = document.title or parsed.title
            document.author = document.author or parsed.author
            document.page_count = parsed.page_count
            document.indexed_at = func.now()

            outbox = OutboxRepository(session)
            await outbox.add(
                event_type=DOCUMENT_INDEXED,
                aggregate_id=document_id,
                payload={
                    "document_id": str(document_id),
                    "user_id": str(user_id),
                    "chunk_count": len(chunk_rows),
                },
            )
            await session.commit()

    log.info("ingestion: indexed ({} chunks)", len(chunk_rows))
    record_document_ingested("indexed")
    observe_document_chunks(len(chunk_rows))


async def fail_document(
    resources: IngestionResources,
    *,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    reason: str,
) -> None:
    """Mark a document ``failed`` with a (length-capped) reason. Best-effort."""
    log = logger.bind(document_id=str(document_id), user_id=str(user_id))
    async with resources.sessionmaker() as session:
        documents = DocumentRepository(session, user_id)
        document = await documents.get(document_id)
        if document is None:
            log.debug("ingestion: fail_document skipped (document not found)")
            return
        document.status = DocumentStatus.FAILED
        document.failure_reason = reason[:_MAX_FAILURE_REASON]
        await session.commit()
    log.error("ingestion: marked failed: {}", reason[:200])
    record_document_ingested("failed")


def _build_chunk_rows(
    chunk_datas: list[ChunkData], *, document_id: uuid.UUID, user_id: uuid.UUID
) -> list[Chunk]:
    """Turn chunk data into ``Chunk`` rows, minting ids so ``vector_id`` matches.

    Each chunk's Qdrant point id is its own UUID, so the row and its vector share
    one identity and a re-embed can find the point from the row.
    """
    rows: list[Chunk] = []
    for data in chunk_datas:
        chunk_id = uuid.uuid4()
        rows.append(
            Chunk(
                id=chunk_id,
                document_id=document_id,
                user_id=user_id,
                ordinal=data.ordinal,
                page_start=data.page_start,
                page_end=data.page_end,
                chapter=data.chapter,
                section=data.section,
                text=data.text,
                token_count=data.token_count,
                content_hash=data.content_hash,
                vector_id=chunk_point_id(chunk_id),
            )
        )
    return rows
