"""Embedding providers: OpenAI, Voyage (hosted), HuggingFace (local), and Ollama.

All embed in bounded batches (``embed_batch_size``) to cap memory — essential for
the local sentence-transformers path, which would otherwise OOM on large inputs.
``dim`` is known up front so the Qdrant collection can be provisioned before any
text is embedded.

Ollama speaks the same OpenAI-compatible ``/v1/embeddings`` shape as hosted
OpenAI, so :class:`OpenAIEmbedder` serves both — only its ``base_url``/``api_key``
differ, mirroring how the LLM (``api/llm.py``) and vision
(:mod:`shared.providers.vision`) providers reuse an OpenAI-compatible client for
local Ollama.
"""

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from shared.core.config import Settings
from shared.providers._config import require_secret, resolve_ollama_api_key
from shared.providers.errors import ProviderConfigError

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIEmbedder:
    """Embeddings via the OpenAI API."""

    # Native output dimensions for common models (override via EMBEDDING_DIM) —
    # both hosted OpenAI models and Ollama's own naming share this table since
    # there's no collision risk and both go through the same client.
    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        batch_size: int = 64,
        dim: int | None = None,
        base_url: str | None = None,
        client: "AsyncOpenAI | None" = None,
    ) -> None:
        self._model = model
        self._batch = batch_size
        resolved = dim or self._DIMS.get(model)
        if resolved is None:
            raise ProviderConfigError(
                f"Unknown embedding dimension for model {model!r}; set EMBEDDING_DIM."
            )
        self._dim = resolved
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        size = batch_size or self._batch
        out: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            resp = await self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out


class VoyageEmbedder:
    """Embeddings via the Voyage AI API."""

    _DIMS: ClassVar[dict[str, int]] = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-3-large": 1024,
        "voyage-3.5": 1024,
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        batch_size: int = 64,
        dim: int | None = None,
        client: object | None = None,
    ) -> None:
        self._model = model
        self._batch = batch_size
        resolved = dim or self._DIMS.get(model)
        if resolved is None:
            raise ProviderConfigError(
                f"Unknown embedding dimension for Voyage model {model!r}; set EMBEDDING_DIM."
            )
        self._dim = resolved
        if client is None:
            import voyageai

            client = voyageai.AsyncClient(api_key=api_key)
        self._client = client

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        size = batch_size or self._batch
        out: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            result = await self._client.embed(batch, model=self._model)
            out.extend(result.embeddings)
        return out


class HuggingFaceEmbedder:
    """Local embeddings via sentence-transformers (the `local` extra).

    ``encode`` is CPU/GPU-bound and synchronous, so it is offloaded to a thread to
    avoid blocking the event loop. Requires the optional local dependencies.

    We slice ``texts`` ourselves and call ``encode`` once per slice. Passing the
    whole document with ``encode(..., batch_size=N)`` still tokenizes and holds
    every chunk (and the concatenated output array) at once — the library's
    internal mini-batch only bounds the forward pass, not peak memory, so a
    large PDF OOMs the worker. Hosted embedders already slice at this layer;
    local must too.
    """

    def __init__(
        self,
        *,
        model: str,
        batch_size: int = 64,
        model_obj: object | None = None,
    ) -> None:
        self._batch = batch_size
        if model_obj is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # optional `local` extra not installed
                raise ProviderConfigError(
                    "HuggingFace embeddings require the local extra "
                    "(sentence-transformers); install it or choose a hosted provider."
                ) from exc
            model_obj = SentenceTransformer(model)
        self._model = model_obj
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        size = batch_size or self._batch
        n = len(texts)
        if n == 0:
            return []
        total = (n + size - 1) // size
        logger.info("embeddings.encode: started ({} texts, {} batches of {})", n, total, size)
        out: list[list[float]] = []
        for index, start in enumerate(range(0, n, size), start=1):
            batch = list(texts[start : start + size])
            logger.info("embeddings.encode: batch {}/{} ({} texts)", index, total, len(batch))
            try:
                # One encode() per our slice: the model never sees more than ``size``
                # texts, so tokenizer + output tensors stay bounded. ``batch_size``
                # on encode is the slice length so it does not sub-batch further.
                vectors = await asyncio.to_thread(
                    self._model.encode,
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            except Exception:
                logger.opt(exception=True).error(
                    "embeddings.encode: batch {}/{} failed", index, total
                )
                raise
            out.extend(vec.tolist() for vec in vectors)
            logger.debug("embeddings.encode: batch {}/{} done ({} vectors)", index, total, len(out))
        logger.info("embeddings.encode: finished ({} vectors)", len(out))
        return out


def build_embedder(settings: Settings) -> OpenAIEmbedder | VoyageEmbedder | HuggingFaceEmbedder:
    """Construct the embedder selected by ``EMBEDDINGS_PROVIDER``."""
    provider = settings.embeddings_provider
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=require_secret(settings.openai_api_key, "OPENAI_API_KEY", "openai embeddings"),
            model=settings.embedding_model,
            batch_size=settings.embed_batch_size,
            dim=settings.embedding_dim,
        )
    if provider == "ollama":
        # Local Ollama ignores the key (a placeholder is used, mirroring the
        # LLM and vision providers); OLLAMA_API_KEY lets this same client
        # reach Ollama Cloud's hosted embedding models instead.
        return OpenAIEmbedder(
            api_key=resolve_ollama_api_key(settings.ollama_api_key),
            model=settings.embedding_model_local,
            batch_size=settings.embed_batch_size,
            dim=settings.embedding_dim,
            base_url=settings.ollama_base_url,
        )
    if provider == "voyage":
        return VoyageEmbedder(
            api_key=require_secret(settings.voyage_api_key, "VOYAGE_API_KEY", "voyage embeddings"),
            model=settings.embedding_model,
            batch_size=settings.embed_batch_size,
            dim=settings.embedding_dim,
        )
    if provider == "huggingface":
        return HuggingFaceEmbedder(
            model=settings.embedding_model, batch_size=settings.embed_batch_size
        )
    raise ProviderConfigError(f"Unknown embeddings provider {provider!r}")
