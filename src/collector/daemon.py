"""Main daemon loop: discover → snapshot → resolve on configurable timers."""

from __future__ import annotations

import asyncio
import logging
import time

import aiosqlite

from collector.config import Config
from collector.db import (
    get_markets_by_status,
    get_unresolved_past_end,
    insert_snapshots_bulk,
    mark_closed,
    mark_resolved,
    upsert_market,
)
from collector.platforms.kalshi import KalshiClient
from collector.platforms.polymarket import PolymarketClient

log = logging.getLogger(__name__)

_MAX_BACKOFF = 300  # 5 minutes


class Collector:
    """Orchestrates the three-phase collection loop."""

    def __init__(self, cfg: Config, db: aiosqlite.Connection) -> None:
        self.cfg = cfg
        self.db = db
        self.poly: PolymarketClient | None = None
        self.kalshi: KalshiClient | None = None
        self._backoff: dict[str, float] = {}

    async def start(self) -> None:
        if self.cfg.polymarket.enabled:
            self.poly = PolymarketClient(
                gamma_url=self.cfg.polymarket.gamma_url,
                clob_url=self.cfg.polymarket.base_url,
            )
        if self.cfg.kalshi.enabled:
            self.kalshi = KalshiClient(base_url=self.cfg.kalshi.base_url)
        log.info(
            "Collector started (poly=%s, kalshi=%s)",
            self.cfg.polymarket.enabled,
            self.cfg.kalshi.enabled,
        )

    async def stop(self) -> None:
        if self.poly:
            await self.poly.close()
        if self.kalshi:
            await self.kalshi.close()
        log.info("Collector stopped")

    # ------------------------------------------------------------------
    # Phase runners
    # ------------------------------------------------------------------

    async def run_discover(self) -> int:
        """DISCOVER phase — fetch and upsert active markets.  Returns count."""
        count = 0
        if self.poly:
            try:
                markets = await self.poly.discover_markets()
                for m in markets:
                    await upsert_market(self.db, **m)
                count += len(markets)
                self._reset_backoff("poly_discover")
            except Exception:
                log.error("DISCOVER polymarket failed", exc_info=True)
                await self._sleep_backoff("poly_discover")

        if self.kalshi:
            try:
                markets = await self.kalshi.discover_markets()
                for m in markets:
                    await upsert_market(self.db, **m)
                count += len(markets)
                self._reset_backoff("kalshi_discover")
            except Exception:
                log.error("DISCOVER kalshi failed", exc_info=True)
                await self._sleep_backoff("kalshi_discover")

        await self.db.commit()
        log.info("DISCOVER complete: %d markets upserted", count)
        return count

    async def run_snapshot(self) -> int:
        """SNAPSHOT phase — capture prices for tracked active markets."""
        count = 0
        if self.poly:
            try:
                active = await get_markets_by_status(self.db, "active", "polymarket")
                if active:
                    snaps = await self.poly.fetch_prices(active)
                    count += await insert_snapshots_bulk(self.db, snaps)
                self._reset_backoff("poly_snapshot")
            except Exception:
                log.error("SNAPSHOT polymarket failed", exc_info=True)
                await self._sleep_backoff("poly_snapshot")

        if self.kalshi:
            try:
                active = await get_markets_by_status(self.db, "active", "kalshi")
                if active:
                    snaps = await self.kalshi.fetch_prices(active)
                    count += await insert_snapshots_bulk(self.db, snaps)
                self._reset_backoff("kalshi_snapshot")
            except Exception:
                log.error("SNAPSHOT kalshi failed", exc_info=True)
                await self._sleep_backoff("kalshi_snapshot")

        await self.db.commit()
        log.info("SNAPSHOT complete: %d rows inserted", count)
        return count

    async def run_resolve(self) -> int:
        """RESOLVE phase — check markets past end_date for outcomes."""
        candidates = await get_unresolved_past_end(self.db)
        resolved_count = 0
        for mkt in candidates:
            platform = mkt["platform"]
            mid = mkt["market_id"]
            client = self.poly if platform == "polymarket" else self.kalshi
            if client is None:
                continue
            try:
                resolution = await client.check_resolution(mid)
                if resolution:
                    await mark_resolved(self.db, platform, mid, resolution)
                    resolved_count += 1
                    log.info("RESOLVED %s/%s → %s", platform, mid, resolution)
                else:
                    # Market past end_date but not resolved yet — mark closed.
                    if mkt["status"] == "active":
                        await mark_closed(self.db, platform, mid)
            except Exception:
                log.warning("RESOLVE failed for %s/%s", platform, mid, exc_info=True)

        await self.db.commit()
        log.info("RESOLVE complete: %d markets resolved out of %d candidates", resolved_count, len(candidates))
        return resolved_count

    # ------------------------------------------------------------------
    # Backfill (one-shot)
    # ------------------------------------------------------------------

    async def run_backfill(self) -> int:
        """Pull historical resolved markets from both platforms."""
        count = 0
        if self.poly:
            markets = await self.poly.fetch_resolved_markets()
            for m in markets:
                await upsert_market(self.db, **m)
            count += len(markets)

        if self.kalshi:
            markets = await self.kalshi.fetch_resolved_markets()
            for m in markets:
                await upsert_market(self.db, **m)
            count += len(markets)

        await self.db.commit()
        log.info("BACKFILL complete: %d markets", count)
        return count

    # ------------------------------------------------------------------
    # Backoff helpers
    # ------------------------------------------------------------------

    async def _sleep_backoff(self, key: str) -> None:
        current = self._backoff.get(key, 1.0)
        self._backoff[key] = min(current * 2, _MAX_BACKOFF)
        log.info("Backing off %s for %.0fs", key, current)
        await asyncio.sleep(current)

    def _reset_backoff(self, key: str) -> None:
        self._backoff.pop(key, None)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

async def run_daemon(cfg: Config, db: aiosqlite.Connection) -> None:
    """Run the daemon forever (or until cancelled)."""
    collector = Collector(cfg, db)
    await collector.start()

    poll = cfg.general.poll_interval_minutes * 60
    snap_interval = cfg.general.snapshot_interval_minutes * 60
    resolve_interval = cfg.general.resolve_interval_minutes * 60
    backfill_interval = cfg.general.backfill_interval_hours * 3600

    last_discover = 0.0
    last_snapshot = 0.0
    last_resolve = 0.0
    last_backfill = 0.0  # runs immediately on first cycle

    try:
        while True:
            now = time.monotonic()

            if now - last_discover >= poll:
                await collector.run_discover()
                last_discover = time.monotonic()

            if now - last_snapshot >= snap_interval:
                await collector.run_snapshot()
                last_snapshot = time.monotonic()

            if now - last_resolve >= resolve_interval:
                await collector.run_resolve()
                last_resolve = time.monotonic()

            if now - last_backfill >= backfill_interval:
                await collector.run_backfill()
                last_backfill = time.monotonic()

            # Sleep until the next earliest event.
            next_wake = min(
                last_discover + poll,
                last_snapshot + snap_interval,
                last_resolve + resolve_interval,
                last_backfill + backfill_interval,
            )
            sleep_for = max(1.0, next_wake - time.monotonic())
            await asyncio.sleep(sleep_for)
    finally:
        await collector.stop()
