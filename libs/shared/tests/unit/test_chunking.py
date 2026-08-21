"""Unit tests for the structure-aware chunker (pure, no I/O)."""

import pytest

from shared.ingestion_core.chunking import (
    chunk_document,
    content_hash,
    estimate_tokens,
)
from shared.ingestion_core.parsing import ParsedDocument, ParsedPage

pytestmark = pytest.mark.unit


def _doc(*pages: tuple[int, str]) -> ParsedDocument:
    return ParsedDocument(pages=[ParsedPage(number=n, text=t) for n, t in pages])


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(_doc((1, "")), chunk_size_words=10, overlap_words=2) == []


def test_single_short_page_is_one_chunk() -> None:
    chunks = chunk_document(_doc((1, "one two three")), chunk_size_words=10, overlap_words=2)
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text == "one two three"
    assert chunks[0].page_start == 1 and chunks[0].page_end == 1


def test_windows_have_contiguous_ordinals_and_expected_count() -> None:
    # 25 words, window 10, overlap 2 → step 8 → windows at 0-9, 8-17, 16-24;
    # the third window reaches the end, so it stops there (3 chunks, no tail gap).
    chunks = chunk_document(_doc((1, _words(25))), chunk_size_words=10, overlap_words=2)
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert chunks[-1].text.split()[-1] == "w24"


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_document(_doc((1, _words(20))), chunk_size_words=10, overlap_words=3)
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:3]
    assert first_tail == second_head


def test_page_attribution_spans_page_boundary() -> None:
    # Page 1 has 6 words, page 2 has 6 words; a 10-word window crosses into page 2.
    chunks = chunk_document(
        _doc((1, _words(6, "a")), (2, _words(6, "b"))),
        chunk_size_words=10,
        overlap_words=0,
    )
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2  # window pulled words from both pages
    # The final window sits wholly on page 2.
    assert chunks[-1].page_start == 2 and chunks[-1].page_end == 2


def test_no_overlap_partitions_words_without_loss() -> None:
    chunks = chunk_document(_doc((1, _words(30))), chunk_size_words=10, overlap_words=0)
    joined = " ".join(c.text for c in chunks).split()
    assert joined == _words(30).split()


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (10, 10), (10, 11), (5, -1)],
)
def test_invalid_window_sizes_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(_doc((1, "x y z")), chunk_size_words=size, overlap_words=overlap)


def test_content_hash_is_whitespace_normalized() -> None:
    assert content_hash("hello   world") == content_hash("hello world")
    assert content_hash("hello\nworld") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("world hello")


def test_estimate_tokens_heuristic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("    ") == 0
    # ~4 chars per token, rounded up.
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
