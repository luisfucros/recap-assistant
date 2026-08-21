"""Provider interfaces and the value types they exchange.

These capabilities are pluggable behind ``Protocol`` interfaces so the app runs
fully hosted (OpenAI/Voyage, S3, Brave/Tavily) or fully local (HuggingFace,
MinIO) by configuration alone — no code change. Concrete implementations live in
sibling modules; ``build_*`` factories select one from ``Settings``. Depend on
these protocols, never on a concrete provider.

Note: LLM chat models are **not** here — the agent runs on LangChain chat models,
built in the API service (``api.llm``) alongside LangGraph.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SearchResult:
    """A normalized web-search hit, provider-agnostic (Brave/Tavily map onto it)."""

    title: str
    url: str
    snippet: str
    score: float | None = None


@runtime_checkable
class Embedder(Protocol):
    """Turn text into vectors; ``dim`` sizes the Qdrant collection."""

    @property
    def dim(self) -> int:
        """Vector dimension of this embedder's output."""
        ...

    async def embed(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        """Embed ``texts`` in bounded batches (avoids OOM on local models)."""
        ...


@runtime_checkable
class StorageProvider(Protocol):
    """Object storage over the S3 API (MinIO locally, AWS S3 in production)."""

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        """Store ``data`` at ``key``."""
        ...

    async def get(self, key: str) -> bytes:
        """Fetch the object at ``key``."""
        ...

    async def delete(self, key: str) -> None:
        """Delete the object at ``key`` (idempotent)."""
        ...


@runtime_checkable
class WebSearchProvider(Protocol):
    """External web search (Brave / Tavily), normalized to ``SearchResult``."""

    async def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        """Return up to ``count`` results for ``query``."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Audio → text (FR-19): hosted OpenAI Whisper API or local HuggingFace Whisper.

    The agent never reasons over audio directly — ``normalize_input`` transcribes
    it to text first, so the whole pipeline (guardrails, retrieval, memory) stays
    single-modality.
    """

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        """Return the transcript of an audio clip."""
        ...


@runtime_checkable
class ImageDescriber(Protocol):
    """Image → text (FR-19): hosted vision model or a local captioning model.

    Like :class:`Transcriber`, this normalizes non-text input to text before the
    agent sees it; the derived description is what's reasoned over (and optionally
    embedded), never the pixels.
    """

    async def describe(self, image: bytes, *, mime_type: str) -> str:
        """Return a textual description/caption of an image."""
        ...
