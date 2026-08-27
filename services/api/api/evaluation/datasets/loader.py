"""Load versioned evaluation datasets (FR-12.4) from the in-repo YAML store.

A dataset is a self-contained fixture: it carries the sample documents' text
directly (page-tagged chunks), rather than referencing real uploads, so a run
never depends on any user's actual library and is reproducible from the
dataset file alone. ``EvaluationService`` seeds these chunks as real
``Document``/``Chunk`` rows (and real vectors) under a dedicated system user
before scoring, so retrieval is genuine, not simulated.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from shared.core.enums import Language
from shared.core.errors import NotFoundError

_DEFAULT_DATASETS_DIR = Path(__file__).parent


class EvalChunk(BaseModel):
    """One fixture chunk: a page-tagged span of a sample document's text."""

    page: int = Field(ge=1)
    chapter: str | None = None
    text: str


class EvalDocument(BaseModel):
    """A sample document a dataset's cases ask questions about."""

    key: str
    title: str
    author: str | None = None
    language: Language = Language.EN
    chunks: list[EvalChunk] = Field(min_length=1)


class EvalCase(BaseModel):
    """One question → expected-chunks/reference-answer evaluation case.

    ``expected_chunk_ordinals`` are 0-based indices into the named document's
    ``chunks`` (the chunk(s) actually relevant to ``query``) — used by the
    retrieval scorers. ``current_page`` simulates the reader's position (read-
    range/spoiler-safe bounding); omitted means the whole document is in read
    range.
    """

    id: str
    document: str
    query: str
    expected_chunk_ordinals: list[int] = Field(default_factory=list)
    reference_answer: str = ""
    current_page: int | None = Field(default=None, ge=1)


class EvalDataset(BaseModel):
    """A versioned, self-contained collection of documents and cases."""

    name: str
    version: str
    documents: list[EvalDocument] = Field(min_length=1)
    cases: list[EvalCase] = Field(min_length=1)

    def document(self, key: str) -> EvalDocument:
        """Return the named fixture document, or raise :class:`NotFoundError`."""
        for doc in self.documents:
            if doc.key == key:
                return doc
        raise NotFoundError(f"Evaluation dataset {self.name!r} has no document {key!r}")


def load_dataset(name: str, *, directory: Path | None = None) -> EvalDataset:
    """Load a dataset by file stem (``name`` -> ``<name>.yaml``).

    Raises :class:`NotFoundError` for an unknown dataset name, so an admin
    triggering a run with a typo'd dataset gets a clean 404 rather than a
    traceback.
    """
    path = (directory or _DEFAULT_DATASETS_DIR) / f"{name}.yaml"
    if not path.is_file():
        raise NotFoundError(f"Unknown evaluation dataset: {name!r}")
    data = yaml.safe_load(path.read_text()) or {}
    return EvalDataset.model_validate(data)


def list_datasets(*, directory: Path | None = None) -> list[EvalDataset]:
    """Return every shipped YAML dataset, sorted by file stem.

    Skips names starting with ``_`` so helper files never appear in the admin UI.
    """
    root = directory or _DEFAULT_DATASETS_DIR
    datasets: list[EvalDataset] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = yaml.safe_load(path.read_text()) or {}
        datasets.append(EvalDataset.model_validate(data))
    return datasets
