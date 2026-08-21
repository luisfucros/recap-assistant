"""Document parsing: extract page-tagged text and metadata from raw bytes.

Parsing is a **Strategy** selected by :class:`ParserFactory` on the document's
format, so adding a format is a new parser class plus one registry entry — no
change to the pipeline that calls it. Only PDF is supported today.

The parser's job is narrow: turn bytes into a :class:`ParsedDocument` (per-page
text + best-effort title/author). Chunking, language detection, and embedding
are separate, downstream steps that consume this output.
"""

import io
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shared.core.enums import DocumentFormat


@dataclass(slots=True)
class ParsedPage:
    """One page of a parsed document (page numbers are 1-based)."""

    number: int
    text: str


@dataclass(slots=True)
class ParsedDocument:
    """The text and metadata extracted from a document."""

    pages: list[ParsedPage]
    title: str | None = None
    author: str | None = None

    @property
    def page_count(self) -> int:
        """Number of pages parsed."""
        return len(self.pages)

    def full_text(self) -> str:
        """All page text joined in reading order (used for language detection)."""
        return "\n".join(page.text for page in self.pages if page.text)


class ParseError(Exception):
    """The document could not be parsed (corrupt, unsupported, or unreadable).

    This is a **permanent** failure: the ingestion task marks the document
    ``failed`` rather than retrying, since re-running won't fix bad bytes.
    """


@runtime_checkable
class DocumentParser(Protocol):
    """Turns raw document bytes into a :class:`ParsedDocument`."""

    def parse(self, data: bytes) -> ParsedDocument:
        """Parse ``data`` or raise :class:`ParseError`."""
        ...


class PdfParser:
    """PDF parsing via ``pypdf`` (pure-Python; no native dependencies)."""

    def parse(self, data: bytes) -> ParsedDocument:
        """Extract per-page text and document metadata from PDF ``data``."""
        # Imported lazily so merely importing this module (e.g. in the API) does
        # not require pypdf to be present until a PDF is actually parsed.
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [
                ParsedPage(number=index + 1, text=(page.extract_text() or "").strip())
                for index, page in enumerate(reader.pages)
            ]
            meta = reader.metadata
        except PyPdfError as exc:
            raise ParseError(f"could not read PDF: {exc}") from exc

        if not pages:
            raise ParseError("PDF has no pages")
        # Empty metadata strings are treated as absent.
        return ParsedDocument(
            pages=pages,
            title=(meta.title if meta and meta.title else None),
            author=(meta.author if meta and meta.author else None),
        )


class ParserFactory:
    """Selects a :class:`DocumentParser` by document format."""

    def __init__(self) -> None:
        # Parsers are stateless, so a single shared instance per format is fine.
        self._parsers: dict[DocumentFormat, DocumentParser] = {DocumentFormat.PDF: PdfParser()}

    def for_format(self, document_format: DocumentFormat) -> DocumentParser:
        """Return the parser for ``document_format`` or raise :class:`ParseError`."""
        try:
            return self._parsers[document_format]
        except KeyError as exc:
            raise ParseError(f"no parser registered for format {document_format!r}") from exc
