"""Content-addressing helpers for uploaded documents.

Originals are stored **content-addressed**: the object-storage key is derived
from the SHA-256 of the bytes, so identical content lands at the same key and
the hash doubles as the per-user duplicate key (``documents.content_sha256``).
Keys are namespaced by ``user_id`` to keep isolation intact — the same bytes for
two different users are two different objects, never a shared blob.

These are pure functions (no I/O): the streaming upload computes the digest as
it writes, then calls :func:`object_key` to place the object.
"""

import hashlib
import uuid
from collections.abc import Iterable


def sha256_hexdigest(data: bytes) -> str:
    """Return the hex SHA-256 of ``data`` (whole-buffer form)."""
    return hashlib.sha256(data).hexdigest()


def sha256_stream(chunks: Iterable[bytes]) -> str:
    """Return the hex SHA-256 of an iterable of byte chunks without buffering all.

    Prefer this on the upload path so a large file is never held in memory in
    full just to hash it.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def object_key(user_id: uuid.UUID, content_sha256: str, extension: str) -> str:
    """Build the content-addressed storage key for a user's document.

    Args:
        user_id: Owner of the document (namespaces the key for isolation).
        content_sha256: Hex SHA-256 of the document bytes.
        extension: File extension, with or without a leading dot (case-insensitive).

    Returns:
        A key of the form ``"<user_id>/sha256/<hash>.<ext>"``.
    """
    ext = extension.lower().lstrip(".")
    return f"{user_id}/sha256/{content_sha256}.{ext}"
