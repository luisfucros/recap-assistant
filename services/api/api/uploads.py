"""Boundary validation for document uploads.

Enforces the security rule that uploaded file *type* and *size* are checked at
the API edge before any bytes are stored or processed. Kept separate from the
route handler so the rules are unit-testable without an HTTP request, and away
from :class:`~api.services.ingestion_service.IngestionService`, which assumes it
is handed already-validated bytes.
"""

from collections.abc import Awaitable, Callable

from shared.core.enums import DocumentFormat

# Accepted upload content types → the document format they map to. PDF only for
# now; adding a format is a one-line change here plus a parser (later milestone).
_CONTENT_TYPE_FORMATS: dict[str, DocumentFormat] = {
    "application/pdf": DocumentFormat.PDF,
}

# Read granularity while enforcing the size cap — large enough to be efficient,
# small enough that a rejected oversize upload is abandoned promptly.
_READ_CHUNK_BYTES = 1024 * 1024


class UnsupportedDocumentFormatError(Exception):
    """The upload's content type is not a supported document format."""


class DocumentTooLargeError(Exception):
    """The upload exceeded the configured size cap."""

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(f"upload exceeds the {limit_bytes}-byte limit")


def resolve_format(content_type: str | None) -> DocumentFormat:
    """Map a request content type to a supported :class:`DocumentFormat`.

    Args:
        content_type: The upload's declared MIME type (may carry parameters like
            ``"application/pdf; charset=..."``; only the media type is compared).

    Raises:
        UnsupportedDocumentFormatError: The type isn't a supported format.
    """
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    try:
        return _CONTENT_TYPE_FORMATS[media_type]
    except KeyError as exc:
        raise UnsupportedDocumentFormatError(f"unsupported content type: {media_type!r}") from exc


async def read_within_limit(read: Callable[[int], Awaitable[bytes]], *, max_bytes: int) -> bytes:
    """Read an upload fully, aborting as soon as it exceeds ``max_bytes``.

    Reads in bounded chunks so an oversize upload is never fully buffered before
    it is rejected. ``read`` is a size-taking async reader (e.g. a Starlette
    ``UploadFile.read``).

    Raises:
        DocumentTooLargeError: Total bytes exceeded ``max_bytes``.
    """
    buffer = bytearray()
    while True:
        chunk = await read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise DocumentTooLargeError(max_bytes)
    return bytes(buffer)
