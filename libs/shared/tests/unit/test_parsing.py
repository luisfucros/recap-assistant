"""Unit tests for PDF parsing and the parser factory.

Text-bearing PDFs are hard to synthesize without a rendering library, so these
cover what can be built deterministically with ``pypdf`` itself — page counting,
metadata extraction, and error handling — leaving full text extraction to the
integration tier (real sample PDFs).
"""

import io

import pytest

from shared.core.enums import DocumentFormat
from shared.ingestion_core.parsing import ParseError, ParserFactory, PdfParser

pytestmark = pytest.mark.unit


def _pdf(*, pages: int, title: str | None = None, author: str | None = None) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    meta: dict[str, str] = {}
    if title is not None:
        meta["/Title"] = title
    if author is not None:
        meta["/Author"] = author
    if meta:
        writer.add_metadata(meta)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_parses_page_count_and_metadata() -> None:
    parsed = PdfParser().parse(_pdf(pages=3, title="Moby Dick", author="Melville"))
    assert parsed.page_count == 3
    assert [p.number for p in parsed.pages] == [1, 2, 3]
    assert parsed.title == "Moby Dick"
    assert parsed.author == "Melville"


def test_missing_metadata_is_none() -> None:
    parsed = PdfParser().parse(_pdf(pages=1))
    assert parsed.title is None
    assert parsed.author is None


def test_garbage_bytes_raise_parse_error() -> None:
    with pytest.raises(ParseError):
        PdfParser().parse(b"this is definitely not a pdf")


def test_factory_returns_pdf_parser() -> None:
    parser = ParserFactory().for_format(DocumentFormat.PDF)
    assert isinstance(parser, PdfParser)
