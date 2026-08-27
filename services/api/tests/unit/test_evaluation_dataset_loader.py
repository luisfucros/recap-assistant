"""Unit tests for the evaluation dataset loader (FR-12.4): file I/O only, no network/DB."""

from pathlib import Path

import pytest
from api.evaluation.datasets.loader import list_datasets, load_dataset

from shared.core.errors import NotFoundError

_VALID_YAML = """
name: fixture_v1
version: v1
documents:
  - key: doc_a
    title: A Test Book
    author: Some Author
    chunks:
      - page: 1
        text: "Page one text."
      - page: 2
        text: "Page two text."
cases:
  - id: case-1
    document: doc_a
    query: What happens on page one?
    expected_chunk_ordinals: [0]
    reference_answer: "Page one describes the opening scene."
    current_page: 2
"""


@pytest.mark.unit
def test_loads_a_valid_dataset(tmp_path: Path) -> None:
    (tmp_path / "fixture_v1.yaml").write_text(_VALID_YAML)

    dataset = load_dataset("fixture_v1", directory=tmp_path)

    assert dataset.name == "fixture_v1"
    assert dataset.version == "v1"
    assert len(dataset.documents) == 1
    assert len(dataset.cases) == 1
    assert dataset.cases[0].expected_chunk_ordinals == [0]


@pytest.mark.unit
def test_document_lookup_returns_the_named_fixture_document(tmp_path: Path) -> None:
    (tmp_path / "fixture_v1.yaml").write_text(_VALID_YAML)
    dataset = load_dataset("fixture_v1", directory=tmp_path)

    doc = dataset.document("doc_a")

    assert doc.title == "A Test Book"
    assert len(doc.chunks) == 2


@pytest.mark.unit
def test_document_lookup_raises_not_found_for_an_unknown_key(tmp_path: Path) -> None:
    (tmp_path / "fixture_v1.yaml").write_text(_VALID_YAML)
    dataset = load_dataset("fixture_v1", directory=tmp_path)

    with pytest.raises(NotFoundError):
        dataset.document("no-such-doc")


@pytest.mark.unit
def test_raises_not_found_for_an_unknown_dataset_name(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_dataset("does_not_exist", directory=tmp_path)


@pytest.mark.unit
def test_a_case_defaults_to_no_expected_chunks_and_no_current_page(tmp_path: Path) -> None:
    minimal = """
name: minimal_v1
version: v1
documents:
  - key: doc_a
    title: A Test Book
    chunks:
      - page: 1
        text: "Some text."
cases:
  - id: case-1
    document: doc_a
    query: A question.
"""
    (tmp_path / "minimal_v1.yaml").write_text(minimal)

    dataset = load_dataset("minimal_v1", directory=tmp_path)

    case = dataset.cases[0]
    assert case.expected_chunk_ordinals == []
    assert case.current_page is None
    assert case.reference_answer == ""


@pytest.mark.unit
def test_the_shipped_sample_dataset_loads_and_has_at_least_one_case() -> None:
    dataset = load_dataset("sample_v1")

    assert dataset.name == "sample_v1"
    assert len(dataset.documents) >= 1
    assert len(dataset.cases) >= 1
    for case in dataset.cases:
        dataset.document(case.document)  # raises if the case references an unknown document


@pytest.mark.unit
def test_list_datasets_returns_the_shipped_sample() -> None:
    datasets = list_datasets()

    names = {d.name for d in datasets}
    assert "sample_v1" in names
    sample = next(d for d in datasets if d.name == "sample_v1")
    assert sample.version
