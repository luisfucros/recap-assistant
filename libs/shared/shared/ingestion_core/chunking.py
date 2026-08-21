"""Structure-aware chunking: split a parsed document into page-tagged spans.

Chunks are the unit of retrieval, so each one records the **page range** it was
drawn from (``page_start``/``page_end``) — that is what powers read-range
scoping and spoiler-safety downstream. Chunking is done over whitespace-delimited
words with a configurable window and overlap; words are a stable, tokenizer-free
proxy for token windows, and the overlap keeps a passage split across a boundary
retrievable from either side.

Everything here is pure and deterministic, so chunk boundaries, page attribution,
token estimates, and content hashes are all unit-testable without a document.
"""

from dataclasses import dataclass

from shared.ingestion_core.content_address import sha256_hexdigest
from shared.ingestion_core.parsing import ParsedDocument


@dataclass(slots=True)
class ChunkData:
    """A page-tagged text span ready to be embedded and persisted."""

    ordinal: int
    text: str
    page_start: int
    page_end: int
    token_count: int
    content_hash: str
    chapter: str | None = None
    section: str | None = None


@dataclass(slots=True)
class _Word:
    """A single word carrying the page it came from (for page attribution)."""

    text: str
    page: int


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with the common ~4-chars-per-token rule.

    A deliberate heuristic, not a real tokenizer: the pipeline has no model-
    specific tokenizer dependency, and chunk sizing only needs a stable estimate
    to stay clear of embedder input limits. Returns 0 for empty text.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return (len(stripped) + 3) // 4


def content_hash(text: str) -> str:
    """Hash ``text`` after whitespace normalization, for retrieval de-duplication.

    Normalizing collapses incidental whitespace differences so two chunks that
    differ only in spacing/newlines hash identically and collapse at retrieval.
    """
    normalized = " ".join(text.split())
    return sha256_hexdigest(normalized.encode("utf-8"))


def _flatten_words(parsed: ParsedDocument) -> list[_Word]:
    """Flatten pages into a page-attributed word stream in reading order."""
    words: list[_Word] = []
    for page in parsed.pages:
        for token in page.text.split():
            words.append(_Word(text=token, page=page.number))
    return words


def chunk_document(
    parsed: ParsedDocument, *, chunk_size_words: int, overlap_words: int
) -> list[ChunkData]:
    """Split ``parsed`` into overlapping, page-tagged chunks.

    Args:
        parsed: The parsed document to chunk.
        chunk_size_words: Target words per chunk (the window size).
        overlap_words: Words shared between consecutive chunks (< ``chunk_size_words``).

    Returns:
        Chunks in reading order with contiguous ``ordinal``s. Empty when the
        document has no extractable text.

    Raises:
        ValueError: If the window/overlap sizes are not a valid pairing.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be in [0, chunk_size_words)")

    words = _flatten_words(parsed)
    if not words:
        return []

    # Step forward by (window - overlap) so consecutive windows share `overlap`
    # words; the guard above keeps the step positive (no infinite loop).
    step = chunk_size_words - overlap_words
    chunks: list[ChunkData] = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size_words]
        text = " ".join(word.text for word in window)
        chunks.append(
            ChunkData(
                ordinal=ordinal,
                text=text,
                page_start=window[0].page,
                page_end=window[-1].page,
                token_count=estimate_tokens(text),
                content_hash=content_hash(text),
            )
        )
        # The final window reached the end; stop rather than emit a tail chunk
        # that is wholly contained in the one just produced.
        if start + chunk_size_words >= len(words):
            break
    return chunks
