"""Unit tests for upload boundary validation (type + size)."""

import pytest
from api.uploads import (
    DocumentTooLargeError,
    UnsupportedDocumentFormatError,
    read_within_limit,
    resolve_format,
)

from shared.core.enums import DocumentFormat

pytestmark = pytest.mark.unit


def test_resolve_format_accepts_pdf() -> None:
    assert resolve_format("application/pdf") is DocumentFormat.PDF


def test_resolve_format_ignores_content_type_parameters() -> None:
    # A charset parameter must not defeat the media-type match.
    assert resolve_format("application/pdf; charset=binary") is DocumentFormat.PDF


def test_resolve_format_is_case_insensitive() -> None:
    assert resolve_format("APPLICATION/PDF") is DocumentFormat.PDF


@pytest.mark.parametrize("content_type", ["text/plain", "image/png", "", None])
def test_resolve_format_rejects_unsupported(content_type: str | None) -> None:
    with pytest.raises(UnsupportedDocumentFormatError):
        resolve_format(content_type)


def _reader(data: bytes):
    """Build an async size-taking reader over ``data`` (like UploadFile.read)."""
    position = {"i": 0}

    async def read(size: int) -> bytes:
        start = position["i"]
        chunk = data[start : start + size]
        position["i"] = start + len(chunk)
        return chunk

    return read


async def test_read_within_limit_returns_full_content() -> None:
    data = b"x" * 5000
    result = await read_within_limit(_reader(data), max_bytes=10_000)
    assert result == data


async def test_read_within_limit_allows_content_at_the_limit() -> None:
    data = b"y" * 1000
    result = await read_within_limit(_reader(data), max_bytes=1000)
    assert result == data


async def test_read_within_limit_rejects_oversize() -> None:
    data = b"z" * 1001
    with pytest.raises(DocumentTooLargeError) as excinfo:
        await read_within_limit(_reader(data), max_bytes=1000)
    assert excinfo.value.limit_bytes == 1000


async def test_read_within_limit_handles_empty_upload() -> None:
    assert await read_within_limit(_reader(b""), max_bytes=1000) == b""
