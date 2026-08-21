"""Unit tests for the provider factories and implementations.

No network: external SDK clients are injected as mocks at the boundary. Settings
are built with ``_env_file=None`` so the local ``.env`` never leaks into tests.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.core.config import Settings
from shared.providers import (
    ProviderConfigError,
    build_embedder,
    build_storage_provider,
    build_web_search_provider,
)
from shared.providers.embeddings import OpenAIEmbedder, VoyageEmbedder
from shared.providers.storage import S3StorageProvider
from shared.providers.websearch import BraveSearchProvider, TavilySearchProvider


def _settings(**overrides: object) -> Settings:
    """Build isolated settings (ignoring any on-disk .env)."""
    return Settings(_env_file=None, **overrides)


# --- Embedder factory ------------------------------------------------------- #


@pytest.mark.unit
def test_build_embedder_selects_openai() -> None:
    emb = build_embedder(_settings(embeddings_provider="openai", openai_api_key="k"))
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.dim == 1536  # text-embedding-3-small default


@pytest.mark.unit
def test_build_embedder_selects_voyage() -> None:
    emb = build_embedder(
        _settings(embeddings_provider="voyage", voyage_api_key="k", embedding_model="voyage-3")
    )
    assert isinstance(emb, VoyageEmbedder)
    assert emb.dim == 1024


@pytest.mark.unit
def test_build_embedder_missing_key_raises() -> None:
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_embedder(_settings(embeddings_provider="openai", openai_api_key=None))


@pytest.mark.unit
def test_build_embedder_selects_ollama() -> None:
    emb = build_embedder(
        _settings(
            embeddings_provider="ollama",
            embedding_model_local="nomic-embed-text",
            ollama_base_url="http://ollama:11434/v1",
        )
    )
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.dim == 768  # nomic-embed-text
    assert str(emb._client.base_url).rstrip("/") == "http://ollama:11434/v1"
    assert emb._client.api_key == "ollama"


@pytest.mark.unit
def test_build_embedder_uses_configured_ollama_api_key_when_set() -> None:
    # OLLAMA_API_KEY set ⇒ Ollama Cloud's hosted embedding models, real key used.
    emb = build_embedder(
        _settings(
            embeddings_provider="ollama",
            embedding_model_local="nomic-embed-text",
            ollama_base_url="https://ollama.com/v1",
            ollama_api_key="sk-cloud-key",
        )
    )
    assert emb._client.api_key == "sk-cloud-key"


@pytest.mark.unit
def test_openai_embedder_unknown_model_needs_explicit_dim() -> None:
    with pytest.raises(ProviderConfigError, match="EMBEDDING_DIM"):
        OpenAIEmbedder(api_key="k", model="some-new-model", client=object())


@pytest.mark.unit
def test_build_embedder_huggingface_without_extra_raises() -> None:
    # The optional local extra (sentence-transformers) is not installed by default.
    with pytest.raises(ProviderConfigError, match="local extra"):
        build_embedder(_settings(embeddings_provider="huggingface", embedding_model="x"))


@pytest.mark.unit
async def test_openai_embedder_batches_calls() -> None:
    """5 texts at batch_size=2 → 3 create() calls; embeddings concatenated in order."""

    def _resp(batch: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))]) for t in batch])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock()))
    client.embeddings.create.side_effect = lambda model, input: _resp(input)

    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small", batch_size=2, client=client)
    vectors = await emb.embed(["a", "bb", "ccc", "dddd", "e"])

    assert client.embeddings.create.await_count == 3
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]


# (LLM providers moved to the API service; see services/api/tests/unit/test_llm.py.)


# --- Web search factory ----------------------------------------------------- #


@pytest.mark.unit
def test_build_web_search_selects_provider() -> None:
    assert isinstance(
        build_web_search_provider(_settings(web_search_provider="brave", brave_api_key="k")),
        BraveSearchProvider,
    )
    assert isinstance(
        build_web_search_provider(_settings(web_search_provider="tavily", tavily_api_key="k")),
        TavilySearchProvider,
    )


@pytest.mark.unit
def test_build_web_search_missing_key_raises() -> None:
    with pytest.raises(ProviderConfigError, match="TAVILY_API_KEY"):
        build_web_search_provider(_settings(web_search_provider="tavily", tavily_api_key=None))


@pytest.mark.unit
async def test_brave_search_normalizes_results() -> None:
    payload = {"web": {"results": [{"title": "T", "url": "http://x", "description": "d"}]}}
    resp = SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)
    client = SimpleNamespace(get=AsyncMock(return_value=resp))

    provider = BraveSearchProvider(api_key="k", client=client)
    results = await provider.search("q", count=3)

    assert len(results) == 1
    assert (results[0].title, results[0].url, results[0].snippet) == ("T", "http://x", "d")


# --- Storage factory -------------------------------------------------------- #


@pytest.mark.unit
def test_build_storage_provider() -> None:
    store = build_storage_provider(_settings(s3_bucket="b", s3_endpoint_url="http://minio:9000"))
    assert isinstance(store, S3StorageProvider)
    assert store._bucket == "b"
