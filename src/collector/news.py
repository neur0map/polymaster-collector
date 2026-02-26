"""SearXNG headline fetcher for market context (Phase 2, disabled by default)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class NewsFetcher:
    """Fetch news headlines related to a market question via SearXNG."""

    def __init__(self, searxng_url: str) -> None:
        self.base_url = searxng_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=15)

    async def close(self) -> None:
        await self._http.aclose()

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Return headlines matching *query*."""
        try:
            resp = await self._http.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "news",
                    "language": "en",
                    "pageno": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("News search failed for query=%s", query, exc_info=True)
            return []

        results = data.get("results", [])[:max_results]
        return [
            {
                "headline": r.get("title", ""),
                "source": r.get("engine", ""),
                "url": r.get("url", ""),
            }
            for r in results
        ]
