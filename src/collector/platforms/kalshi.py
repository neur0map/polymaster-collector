"""Kalshi public REST API client.

Resolution logic:
  - Kalshi market objects have a ``status`` field:
    "initialized" | "open" | "closed" | "settled"
  - When ``status == "settled"`` the ``result`` field contains ``"yes"`` or ``"no"``.
  - Prices are included inline in the market response (yes_bid / yes_ask / no_bid / no_ask)
    so no separate price endpoint is needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_PAGE_LIMIT = 200  # Kalshi supports up to 1000, but be conservative
_RATE_DELAY = 0.5  # ~2 req/sec to be safe


class KalshiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30)

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # DISCOVER
    # ------------------------------------------------------------------

    async def discover_markets(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (market_dicts, snapshot_dicts) for all open markets.

        Kalshi includes prices inline, so we capture snapshots during discover.
        """
        raw = await self._fetch_all_markets(status="open")
        markets = []
        snapshots = []
        for m in raw:
            norm = self._normalise(m)
            markets.append(norm)
            # Build snapshot from normalised prices.
            yes_bid = _cents_to_frac(m.get("yes_bid"))
            yes_ask = _cents_to_frac(m.get("yes_ask"))
            if yes_bid is not None and yes_ask is not None:
                yes_price = (yes_bid + yes_ask) / 2
            elif yes_bid is not None:
                yes_price = yes_bid
            else:
                yes_price = yes_ask
            if yes_price is not None:
                no_price = 1.0 - yes_price
                snapshots.append(dict(
                    market_id=norm["market_id"],
                    platform="kalshi",
                    yes_price=yes_price,
                    no_price=no_price,
                    volume=_float(m.get("volume")),
                    liquidity=_float(m.get("open_interest")),
                    spread=abs(yes_price - no_price) if no_price is not None else None,
                ))
        return markets, snapshots

    async def _fetch_all_markets(
        self, status: str | None = None, *, max_pages: int = 50,
    ) -> list[dict]:
        all_markets: list[dict] = []
        cursor: str | None = None
        pages = 0
        while pages < max_pages:
            params: dict[str, Any] = {"limit": _PAGE_LIMIT}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            await asyncio.sleep(_RATE_DELAY)
            resp = await self._http.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("markets", [])
            all_markets.extend(batch)
            cursor = data.get("cursor")
            pages += 1
            if not cursor or not batch:
                break
        if pages >= max_pages:
            log.info("Kalshi: hit max_pages=%d cap at %d markets", max_pages, len(all_markets))
        log.info("Kalshi: fetched %d markets (status=%s, pages=%d)", len(all_markets), status, pages)
        return all_markets

    def _normalise(self, m: dict) -> dict[str, Any]:
        """Map Kalshi market JSON → our DB columns."""
        # Kalshi prices are in cents (0–100).  Normalise to 0.0–1.0.
        yes_bid = _cents_to_frac(m.get("yes_bid"))
        yes_ask = _cents_to_frac(m.get("yes_ask"))
        no_bid = _cents_to_frac(m.get("no_bid"))

        # Use midpoint of yes bid/ask as the canonical yes_price.
        if yes_bid is not None and yes_ask is not None:
            yes_price = (yes_bid + yes_ask) / 2
        elif yes_bid is not None:
            yes_price = yes_bid
        else:
            yes_price = yes_ask
        no_price = 1.0 - yes_price if yes_price is not None else None

        status_map = {"open": "active", "closed": "closed", "settled": "resolved"}

        return dict(
            platform="kalshi",
            market_id=m.get("ticker", ""),
            slug=m.get("ticker", ""),
            title=m.get("title", ""),
            description=m.get("subtitle", ""),
            category=m.get("category", ""),
            outcomes=["Yes", "No"],
            volume=_float(m.get("volume")),
            liquidity=_float(m.get("open_interest")),
            end_date=m.get("close_time") or m.get("expiration_time"),
            status=status_map.get(m.get("status", ""), "active"),
        )

    # ------------------------------------------------------------------
    # SNAPSHOT — prices are inline, just re-fetch the list
    # ------------------------------------------------------------------

    async def fetch_prices(self, markets: list[dict]) -> list[dict]:
        """Fetch current prices for tracked markets.

        We batch-fetch by re-querying open markets since Kalshi includes
        prices in the market response.  For a smaller set we could query
        individual tickers; the bulk approach is simpler.
        """
        raw = await self._fetch_all_markets(status="open")
        tracked_ids = {m["market_id"] for m in markets}
        snapshots: list[dict] = []
        for m in raw:
            ticker = m.get("ticker", "")
            if ticker not in tracked_ids:
                continue

            yes_bid = _cents_to_frac(m.get("yes_bid"))
            yes_ask = _cents_to_frac(m.get("yes_ask"))
            if yes_bid is not None and yes_ask is not None:
                yes_price = (yes_bid + yes_ask) / 2
            elif yes_bid is not None:
                yes_price = yes_bid
            else:
                yes_price = yes_ask
            no_price = 1.0 - yes_price if yes_price is not None else None
            spread = abs(yes_price - no_price) if yes_price is not None and no_price is not None else None

            snapshots.append(dict(
                market_id=ticker,
                platform="kalshi",
                yes_price=yes_price,
                no_price=no_price,
                volume=_float(m.get("volume")),
                liquidity=_float(m.get("open_interest")),
                spread=spread,
            ))
        log.info("Kalshi: captured %d price snapshots", len(snapshots))
        return snapshots

    # ------------------------------------------------------------------
    # RESOLVE — check the ``result`` field
    # ------------------------------------------------------------------

    async def check_resolution(self, market_id: str) -> str | None:
        """Query Kalshi for a single market's resolution.

        Returns:
            "YES", "NO", or None if not settled.
        """
        try:
            await asyncio.sleep(_RATE_DELAY)
            resp = await self._http.get(f"{self.base_url}/markets/{market_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json().get("market", resp.json())
        except Exception:
            log.warning("Kalshi: resolution check failed for %s", market_id, exc_info=True)
            return None

        if data.get("status") != "settled":
            return None

        result = (data.get("result") or "").strip().lower()
        if result == "yes":
            return "YES"
        elif result == "no":
            return "NO"
        return None

    # ------------------------------------------------------------------
    # BACKFILL
    # ------------------------------------------------------------------

    async def fetch_resolved_markets(self) -> list[dict[str, Any]]:
        """Fetch settled markets for backfill."""
        raw = await self._fetch_all_markets(status="settled", max_pages=200)
        results: list[dict] = []
        for m in raw:
            norm = self._normalise(m)
            norm["status"] = "resolved"
            result = (m.get("result") or "").strip().lower()
            if result in ("yes", "no"):
                norm["resolution"] = result.upper()
            results.append(norm)
        log.info("Kalshi: backfill fetched %d settled markets", len(results))
        return results


def _cents_to_frac(val: Any) -> float | None:
    """Convert Kalshi price in cents (0-100) to fraction (0.0-1.0)."""
    if val is None:
        return None
    try:
        v = float(val)
        return v / 100.0 if v > 1 else v
    except (ValueError, TypeError):
        return None


def _float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
