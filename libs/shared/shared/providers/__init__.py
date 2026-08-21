"""Pluggable provider interfaces and their config-selected factories.

Depend on the protocols (``Embedder``, ``StorageProvider``, ``WebSearchProvider``)
and build concretes with the ``build_*`` factories; never import a concrete
provider directly. Selection is driven entirely by ``Settings``, so the app runs
fully hosted or fully local without code changes.

LLM chat models are not here: the agent runs on LangChain chat models built in
the API service (``api.llm``), so LangChain stays out of the shared library.
"""

from shared.providers.base import (
    Embedder,
    ImageDescriber,
    SearchResult,
    StorageProvider,
    Transcriber,
    WebSearchProvider,
)
from shared.providers.embeddings import build_embedder
from shared.providers.errors import ProviderConfigError, ProviderError
from shared.providers.storage import build_storage_provider
from shared.providers.transcription import build_transcriber
from shared.providers.vision import build_image_describer
from shared.providers.websearch import build_web_search_provider

__all__ = [  # noqa: RUF022 — grouped by kind (protocols, factories, errors), not sorted
    # Protocols + value types
    "Embedder",
    "StorageProvider",
    "WebSearchProvider",
    "Transcriber",
    "ImageDescriber",
    "SearchResult",
    # Factories
    "build_embedder",
    "build_storage_provider",
    "build_web_search_provider",
    "build_transcriber",
    "build_image_describer",
    # Errors
    "ProviderError",
    "ProviderConfigError",
]
