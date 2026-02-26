"""Tests for Polymarket and Kalshi API clients using respx mocks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from collector.platforms.polymarket import PolymarketClient
from collector.platforms.kalshi import KalshiClient

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ------------------------------------------------------------------
# Polymarket tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_polymarket_discover():
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "conditionId": "0xabc",
                    "slug": "btc-100k",
                    "question": "Will BTC hit 100k?",
                    "description": "Test",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.65", "0.35"]',
                    "volume": 50000,
                    "liquidity": 10000,
                    "endDate": "2026-12-31T00:00:00Z",
                    "active": True,
                    "closed": False,
                }
            ],
        )
    )

    client = PolymarketClient(GAMMA_URL, CLOB_URL)
    markets = await client.discover_markets()
    await client.close()

    assert len(markets) == 1
    m = markets[0]
    assert m["platform"] == "polymarket"
    assert m["market_id"] == "0xabc"
    assert m["title"] == "Will BTC hit 100k?"
    assert m["status"] == "active"


@pytest.mark.asyncio
@respx.mock
async def test_polymarket_check_resolution_yes():
    respx.get(f"{GAMMA_URL}/markets/0xabc").mock(
        return_value=httpx.Response(
            200,
            json={
                "closed": True,
                "resolved": True,
                "outcomePrices": '["1.0", "0.0"]',
            },
        )
    )

    client = PolymarketClient(GAMMA_URL, CLOB_URL)
    result = await client.check_resolution("0xabc")
    await client.close()

    assert result == "YES"


@pytest.mark.asyncio
@respx.mock
async def test_polymarket_check_resolution_no():
    respx.get(f"{GAMMA_URL}/markets/0xdef").mock(
        return_value=httpx.Response(
            200,
            json={
                "closed": True,
                "resolved": True,
                "outcomePrices": '["0.0", "1.0"]',
            },
        )
    )

    client = PolymarketClient(GAMMA_URL, CLOB_URL)
    result = await client.check_resolution("0xdef")
    await client.close()

    assert result == "NO"


@pytest.mark.asyncio
@respx.mock
async def test_polymarket_check_resolution_not_resolved():
    respx.get(f"{GAMMA_URL}/markets/0xpending").mock(
        return_value=httpx.Response(
            200,
            json={
                "closed": False,
                "outcomePrices": '["0.6", "0.4"]',
            },
        )
    )

    client = PolymarketClient(GAMMA_URL, CLOB_URL)
    result = await client.check_resolution("0xpending")
    await client.close()

    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_polymarket_fetch_prices():
    respx.get(f"{GAMMA_URL}/markets/0xabc").mock(
        return_value=httpx.Response(
            200,
            json={
                "outcomePrices": '["0.72", "0.28"]',
                "volume": 5000,
                "liquidity": 2000,
            },
        )
    )

    client = PolymarketClient(GAMMA_URL, CLOB_URL)
    snaps = await client.fetch_prices([{"market_id": "0xabc"}])
    await client.close()

    assert len(snaps) == 1
    assert snaps[0]["yes_price"] == pytest.approx(0.72)
    assert snaps[0]["no_price"] == pytest.approx(0.28)


# ------------------------------------------------------------------
# Kalshi tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_kalshi_discover():
    respx.get(f"{KALSHI_URL}/markets").mock(
        return_value=httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "FED-25MAR-T4.50",
                        "event_ticker": "FED-25MAR",
                        "title": "Fed rate above 4.50?",
                        "subtitle": "> 4.50%",
                        "category": "Economics",
                        "status": "open",
                        "yes_bid": 65,
                        "yes_ask": 67,
                        "no_bid": 33,
                        "no_ask": 35,
                        "volume": 10000,
                        "open_interest": 5000,
                        "close_time": "2025-03-20T00:00:00Z",
                        "result": "",
                    }
                ],
                "cursor": "",
            },
        )
    )

    client = KalshiClient(KALSHI_URL)
    markets = await client.discover_markets()
    await client.close()

    assert len(markets) == 1
    m = markets[0]
    assert m["platform"] == "kalshi"
    assert m["market_id"] == "FED-25MAR-T4.50"
    assert m["status"] == "active"


@pytest.mark.asyncio
@respx.mock
async def test_kalshi_check_resolution_yes():
    respx.get(f"{KALSHI_URL}/markets/FED-25MAR-T4.50").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "FED-25MAR-T4.50",
                    "status": "settled",
                    "result": "yes",
                }
            },
        )
    )

    client = KalshiClient(KALSHI_URL)
    result = await client.check_resolution("FED-25MAR-T4.50")
    await client.close()

    assert result == "YES"


@pytest.mark.asyncio
@respx.mock
async def test_kalshi_check_resolution_no():
    respx.get(f"{KALSHI_URL}/markets/FED-25MAR-T4.50").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "FED-25MAR-T4.50",
                    "status": "settled",
                    "result": "no",
                }
            },
        )
    )

    client = KalshiClient(KALSHI_URL)
    result = await client.check_resolution("FED-25MAR-T4.50")
    await client.close()

    assert result == "NO"


@pytest.mark.asyncio
@respx.mock
async def test_kalshi_check_resolution_not_settled():
    respx.get(f"{KALSHI_URL}/markets/FED-25MAR-T4.50").mock(
        return_value=httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "FED-25MAR-T4.50",
                    "status": "open",
                    "result": "",
                }
            },
        )
    )

    client = KalshiClient(KALSHI_URL)
    result = await client.check_resolution("FED-25MAR-T4.50")
    await client.close()

    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_kalshi_fetch_prices():
    respx.get(f"{KALSHI_URL}/markets").mock(
        return_value=httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "FED-25MAR-T4.50",
                        "status": "open",
                        "yes_bid": 65,
                        "yes_ask": 67,
                        "no_bid": 33,
                        "no_ask": 35,
                        "volume": 10000,
                        "open_interest": 5000,
                    }
                ],
                "cursor": "",
            },
        )
    )

    client = KalshiClient(KALSHI_URL)
    snaps = await client.fetch_prices([{"market_id": "FED-25MAR-T4.50"}])
    await client.close()

    assert len(snaps) == 1
    assert snaps[0]["yes_price"] == pytest.approx(0.66)  # midpoint of 0.65 and 0.67
    assert snaps[0]["platform"] == "kalshi"
