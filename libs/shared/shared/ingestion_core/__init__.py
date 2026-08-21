"""Parsing, chunking, and embedding building blocks shared by the ingestion pipeline.

Pure logic (no Celery, no DB, no network): a format-Strategy parser, a page-tagged
chunker, language detection over the supported set, and content-addressing
helpers. The ingestion service composes these into its task; keeping them here
makes each independently unit-testable and reusable (e.g. a re-embed job).
"""

from shared.ingestion_core.chunking import (
    ChunkData,
    chunk_document,
    content_hash,
    estimate_tokens,
)
from shared.ingestion_core.content_address import object_key, sha256_hexdigest, sha256_stream
from shared.ingestion_core.language import detect_language
from shared.ingestion_core.parsing import (
    DocumentParser,
    ParsedDocument,
    ParsedPage,
    ParseError,
    ParserFactory,
    PdfParser,
)

__all__ = [  # noqa: RUF022 — grouped by module, not alphabetized
    # parsing
    "DocumentParser",
    "ParsedDocument",
    "ParsedPage",
    "ParseError",
    "ParserFactory",
    "PdfParser",
    # chunking
    "ChunkData",
    "chunk_document",
    "content_hash",
    "estimate_tokens",
    # language
    "detect_language",
    # content addressing
    "object_key",
    "sha256_hexdigest",
    "sha256_stream",
]
