"""Tests for the collector daemon phases."""

from __future__ import annotations

import httpx
import pytest
import respx

from collector.daemon import Collector
from collector.db import get_markets_by_status, upsert_market

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"


@pytest.mark.asyncio
@respx.mock
async def test_discover_phase(db, cfg):
    """DISCOVER should upsert markets from both platforms."""
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "conditionId": "0xabc",
                    "question": "Test Poly Market",
                    "outcomes": '["Yes", "No"]',
                    "volume": 1000,
                    "endDate": "2026-12-31T00:00:00Z",
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )
    respx.get(f"{KALSHI_URL}/markets").mock(
        return_value=httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "TEST-TICKER",
                        "title": "Test Kalshi Market",
                        "status": "open",
                        "yes_bid": 50,
                        "yes_ask": 52,
                        "volume": 500,
                        "close_time": "2026-12-31T00:00:00Z",
                    }
                ],
                "cursor": "",
            },
        )
    )

    collector = Collector(cfg, db)
    await collector.start()
    count = await collector.run_discover()
    await collector.stop()

    assert count == 2
    active = await get_markets_by_status(db, "active")
    assert len(active) == 2
    platforms = {m["platform"] for m in active}
    assert platforms == {"polymarket", "kalshi"}


@pytest.mark.asyncio
@respx.mock
async def test_resolve_phase(db, cfg):
    """RESOLVE should update markets with resolution from API."""
    # Seed a market past its end_date.
    await upsert_market(
        db,
        platform="kalshi",
        market_id="RESOLVED-TICKER",
        title="Old market",
        status="active",
        end_date="2020-01-01T00:00:00Z",
    )
    await db.commit()

    respx.get(f"{KALSHI_URL}/markets/RESOLVED-TICKER").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "RESOLVED-TICKER",
                    "status": "settled",
                    "result": "yes",
                }
            },
        )
    )

    collector = Collector(cfg, db)
    await collector.start()
    resolved = await collector.run_resolve()
    await collector.stop()

    assert resolved == 1
    rows = await get_markets_by_status(db, "resolved")
    assert len(rows) == 1
    assert rows[0]["resolution"] == "YES"
