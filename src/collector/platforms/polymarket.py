"""Polymarket Gamma + CLOB API client.

Resolution logic:
  - Gamma API market objects have ``closed`` (bool) and ``resolved`` (bool) fields.
  - When ``resolved=true`` the ``outcomePrices`` array collapses to 1.0/0.0 —
    the outcome at index 0 whose price is "1" is the winning side.
  - We map this to YES / NO for our DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_PAGE_LIMIT = 100  # Gamma API max per page
_RATE_DELAY = 0.5  # 2 req/sec


class PolymarketClient:
    def __init__(self, gamma_url: str, clob_url: str) -> None:
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30)

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # DISCOVER — paginated fetch of all active markets
    # ------------------------------------------------------------------

    async def discover_markets(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (market_dicts, snapshot_dicts) for upsertion.

        Prices are captured from the discovery response so we don't need
        individual per-market API calls for the initial snapshot.
        """
        raw_markets = await self._fetch_all_markets()
        markets = []
        snapshots = []
        for m in raw_markets:
            markets.append(self._normalise(m))
            snap = self._extract_snapshot(m)
            if snap:
                snapshots.append(snap)
        return markets, snapshots

    def _extract_snapshot(self, m: dict) -> dict[str, Any] | None:
        """Extract a price snapshot from a raw Gamma API market object."""
        outcome_prices = m.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                return None
        if not outcome_prices:
            return None
        try:
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else None
        except (ValueError, IndexError):
            return None
        spread = abs(yes_price - no_price) if yes_price is not None and no_price is not None else None
        mid = str(m.get("conditionId") or m.get("id", ""))
        if not mid:
            return None
        return dict(
            market_id=mid,
            platform="polymarket",
            yes_price=yes_price,
            no_price=no_price,
            volume=_float(m.get("volume") or m.get("volumeNum")),
            liquidity=_float(m.get("liquidity") or m.get("liquidityNum")),
            spread=spread,
        )

    async def _fetch_all_markets(self) -> list[dict]:
        all_markets: list[dict] = []
        seen_ids: set[str] = set()
        offset = 0
        max_offset = 5000  # safety cap — Polymarket has ~2-3k active markets
        while offset <= max_offset:
            params = {
                "limit": _PAGE_LIMIT,
                "offset": offset,
                "active": "true",
                "closed": "false",
                "archived": "false",
            }
            await asyncio.sleep(_RATE_DELAY)
            resp = await self._http.get(f"{self.gamma_url}/markets", params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            # Deduplicate — Gamma API can recycle results at high offsets.
            new_count = 0
            for m in batch:
                cid = str(m.get("conditionId") or m.get("id", ""))
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_markets.append(m)
                    new_count += 1
            if new_count == 0:
                log.info("Polymarket: no new markets at offset %d, stopping", offset)
                break
            if len(batch) < _PAGE_LIMIT:
                break
            offset += _PAGE_LIMIT
        log.info("Polymarket: discovered %d unique active markets", len(all_markets))
        return all_markets

    def _normalise(self, m: dict) -> dict[str, Any]:
        """Map Gamma API market object → our DB columns."""
        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = []

        end_date = m.get("endDate") or m.get("endDateIso")

        return dict(
            platform="polymarket",
            market_id=str(m.get("conditionId") or m.get("id", "")),
            slug=m.get("slug", ""),
            title=m.get("question", ""),
            description=(m.get("description") or "")[:2000],
            category=m.get("groupItemTitle", ""),
            outcomes=outcomes,
            volume=_float(m.get("volume") or m.get("volumeNum")),
            liquidity=_float(m.get("liquidity") or m.get("liquidityNum")),
            end_date=end_date,
            status="active",
        )

    # ------------------------------------------------------------------
    # SNAPSHOT — current prices for a list of markets
    # ------------------------------------------------------------------

    async def fetch_prices(self, markets: list[dict], *, max_markets: int = 200) -> list[dict]:
        """Return snapshot rows for the given markets (capped to max_markets)."""
        # Only refresh a subset per cycle — discover already captures initial prices.
        subset = markets[:max_markets]
        snapshots: list[dict] = []
        for mkt in subset:
            slug = mkt.get("slug", "")
            if not slug:
                continue
            try:
                await asyncio.sleep(_RATE_DELAY)
                resp = await self._http.get(
                    f"{self.gamma_url}/markets",
                    params={"slug": slug, "limit": 1},
                )
                if resp.status_code in (404, 422):
                    continue
                resp.raise_for_status()
                data_list = resp.json()
                if not data_list:
                    continue
                data = data_list[0] if isinstance(data_list, list) else data_list
            except Exception:
                log.warning("Polymarket: price fetch failed for %s", slug, exc_info=True)
                continue

            outcome_prices = data.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else None
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else None
            spread = abs(yes_price - no_price) if yes_price is not None and no_price is not None else None

            snapshots.append(dict(
                market_id=mkt["market_id"],
                platform="polymarket",
                yes_price=yes_price,
                no_price=no_price,
                volume=_float(data.get("volume") or data.get("volumeNum")),
                liquidity=_float(data.get("liquidity") or data.get("liquidityNum")),
                spread=spread,
            ))
        log.info("Polymarket: captured %d price snapshots", len(snapshots))
        return snapshots

    # ------------------------------------------------------------------
    # RESOLVE — check if a market has resolved, return resolution
    # ------------------------------------------------------------------

    async def check_resolution(self, market_id: str) -> str | None:
        """Query the Gamma API for resolution status.

        Returns:
            "YES", "NO", or None if not yet resolved.
        """
        try:
            await asyncio.sleep(_RATE_DELAY)
            resp = await self._http.get(f"{self.gamma_url}/markets/{market_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("Polymarket: resolution check failed for %s", market_id, exc_info=True)
            return None

        # The Gamma API exposes:  closed (bool) and a few resolution indicators.
        # When a market resolves the outcomePrices collapse to 1/0.
        # Also check for explicit 'resolved' boolean if present.
        if not (data.get("closed") or data.get("resolved")):
            return None

        outcome_prices = data.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                return None

        if not outcome_prices:
            return None

        # If price is 1.0 (or very close) for the first outcome → YES won.
        try:
            yes_price = float(outcome_prices[0])
        except (ValueError, IndexError):
            return None

        if yes_price >= 0.99:
            return "YES"
        elif yes_price <= 0.01:
            return "NO"

        # Not clearly resolved yet (maybe closed but awaiting oracle).
        return None

    # ------------------------------------------------------------------
    # BACKFILL — fetch resolved/closed markets
    # ------------------------------------------------------------------

    async def fetch_resolved_markets(self) -> list[dict[str, Any]]:
        """Fetch historical resolved markets for backfill."""
        all_markets: list[dict] = []
        seen_ids: set[str] = set()
        offset = 0
        max_offset = 20000  # backfill can go deeper
        while offset <= max_offset:
            params = {
                "limit": _PAGE_LIMIT,
                "offset": offset,
                "closed": "true",
            }
            await asyncio.sleep(_RATE_DELAY)
            resp = await self._http.get(f"{self.gamma_url}/markets", params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            new_count = 0
            for m in batch:
                cid = str(m.get("conditionId") or m.get("id", ""))
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_markets.append(m)
                    new_count += 1
            if new_count == 0:
                log.info("Polymarket: backfill no new markets at offset %d, stopping", offset)
                break
            if len(batch) < _PAGE_LIMIT:
                break
            offset += _PAGE_LIMIT
        log.info("Polymarket: backfill fetched %d unique closed markets", len(all_markets))
        return [self._normalise_resolved(m) for m in all_markets]

    def _normalise_resolved(self, m: dict) -> dict[str, Any]:
        """Normalise a resolved/closed market for backfill upsert."""
        base = self._normalise(m)
        base["status"] = "resolved" if m.get("resolved") else "closed"

        outcome_prices = m.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = []

        if outcome_prices:
            try:
                yes_price = float(outcome_prices[0])
                if yes_price >= 0.99:
                    base["resolution"] = "YES"
                elif yes_price <= 0.01:
                    base["resolution"] = "NO"
            except (ValueError, IndexError):
                pass

        return base


def _float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
