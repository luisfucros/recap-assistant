"""Web-search providers: Brave and Tavily, normalized to ``SearchResult``.

Both are thin HTTP clients; the agent's ``web_search`` tool depends only on the
``WebSearchProvider`` protocol, so switching providers is a config change.
"""

from collections.abc import Sequence

import httpx

from shared.core.config import Settings
from shared.providers._config import require_secret
from shared.providers.base import SearchResult
from shared.providers.errors import ProviderConfigError

_DEFAULT_TIMEOUT = 10.0


class BraveSearchProvider:
    """Web search via the Brave Search API."""

    _URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, *, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        resp = await self._client.get(
            self._URL,
            params={"q": query, "count": count},
            headers={"X-Subscription-Token": self._key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        hits = resp.json().get("web", {}).get("results", [])
        return _to_results(hits, title="title", url="url", snippet="description")


class TavilySearchProvider:
    """Web search via the Tavily API."""

    _URL = "https://api.tavily.com/search"

    def __init__(self, *, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._client = client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        resp = await self._client.post(
            self._URL,
            json={"api_key": self._key, "query": query, "max_results": count},
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [])
        return _to_results(hits, title="title", url="url", snippet="content", score="score")


def _to_results(
    hits: Sequence[dict],
    *,
    title: str,
    url: str,
    snippet: str,
    score: str | None = None,
) -> list[SearchResult]:
    """Map a provider's raw hit dicts onto the common ``SearchResult`` shape."""
    return [
        SearchResult(
            title=hit.get(title, ""),
            url=hit.get(url, ""),
            snippet=hit.get(snippet, ""),
            score=hit.get(score) if score else None,
        )
        for hit in hits
    ]


def build_web_search_provider(settings: Settings) -> BraveSearchProvider | TavilySearchProvider:
    """Construct the web-search provider selected by ``WEB_SEARCH_PROVIDER``."""
    provider = settings.web_search_provider
    if provider == "brave":
        return BraveSearchProvider(
            api_key=require_secret(settings.brave_api_key, "BRAVE_API_KEY", "brave web search")
        )
    if provider == "tavily":
        return TavilySearchProvider(
            api_key=require_secret(settings.tavily_api_key, "TAVILY_API_KEY", "tavily web search")
        )
    raise ProviderConfigError(f"Unknown web search provider {provider!r}")
