"""Unit tests for the ingestion worker's resource container (no infra I/O)."""

import pytest
from ingestion.resources import IngestionResources, get_ingestion_resources

from shared.core.config import Settings
from shared.providers.errors import ProviderConfigError


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.mark.unit
def test_eager_singletons_built_offline() -> None:
    resources = IngestionResources(_settings())
    assert resources.engine is not None
    assert resources.sessionmaker is not None
    assert resources.qdrant is not None


@pytest.mark.unit
def test_embedder_is_lazy_and_cached() -> None:
    resources = IngestionResources(_settings(embeddings_provider="openai", openai_api_key="k"))
    assert resources.embedder is resources.embedder  # cached_property → same instance


@pytest.mark.unit
def test_embedder_missing_key_raises_only_on_access() -> None:
    # Construction succeeds with no key; the error surfaces when the embedder is used.
    resources = IngestionResources(_settings(embeddings_provider="openai", openai_api_key=None))
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        _ = resources.embedder


@pytest.mark.unit
def test_get_ingestion_resources_is_a_singleton() -> None:
    assert get_ingestion_resources() is get_ingestion_resources()


@pytest.mark.unit
async def test_aclose_disposes_without_connections() -> None:
    resources = IngestionResources(_settings())
    await resources.aclose()  # no connections were opened; must not raise
