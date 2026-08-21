"""Unit tests for the content-addressing helpers (pure, no I/O)."""

import hashlib
import uuid

import pytest

from shared.ingestion_core.content_address import (
    object_key,
    sha256_hexdigest,
    sha256_stream,
)

pytestmark = pytest.mark.unit


def test_sha256_hexdigest_matches_hashlib() -> None:
    data = b"Recap reading assistant"
    assert sha256_hexdigest(data) == hashlib.sha256(data).hexdigest()


def test_sha256_is_stable_across_identical_bytes() -> None:
    # The upload path relies on identical bytes hashing identically — that is what
    # makes (user_id, content_sha256) a reliable duplicate key.
    assert sha256_hexdigest(b"same") == sha256_hexdigest(b"same")


def test_sha256_stream_equals_whole_buffer() -> None:
    data = b"chunked hashing must equal one-shot hashing"
    streamed = sha256_stream([data[:10], data[10:25], data[25:]])
    assert streamed == sha256_hexdigest(data)


def test_sha256_stream_handles_empty_iterable() -> None:
    assert sha256_stream([]) == hashlib.sha256(b"").hexdigest()


def test_object_key_shape_and_namespacing() -> None:
    user_id = uuid.uuid4()
    key = object_key(user_id, "abc123", "pdf")
    assert key == f"{user_id}/sha256/abc123.pdf"


def test_object_key_normalizes_extension() -> None:
    user_id = uuid.uuid4()
    # Leading dot and case are normalized so the key is canonical.
    assert object_key(user_id, "h", ".PDF").endswith(".pdf")
    assert object_key(user_id, "h", "PDF").endswith(".pdf")


def test_object_key_isolates_users_for_identical_content() -> None:
    sha = "deadbeef"
    a, b = uuid.uuid4(), uuid.uuid4()
    # Same content, different owners ⇒ different keys (isolation over dedup).
    assert object_key(a, sha, "pdf") != object_key(b, sha, "pdf")
