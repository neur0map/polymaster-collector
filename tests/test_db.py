"""Tests for collector.db module."""

from __future__ import annotations

import pytest

from collector.db import (
    get_markets_by_status,
    get_unresolved_past_end,
    insert_snapshot,
    insert_snapshots_bulk,
    mark_closed,
    mark_resolved,
    stats,
    upsert_market,
)


@pytest.mark.asyncio
async def test_schema_created(db):
    """Tables should exist after init_db."""
    rows = await db.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {r["name"] for r in rows}
    assert "markets" in names
    assert "price_snapshots" in names
    assert "news_context" in names
    assert "_meta" in names


@pytest.mark.asyncio
async def test_upsert_market_insert_and_update(db):
    await upsert_market(
        db,
        platform="polymarket",
        market_id="cond_abc",
        title="Will BTC hit 100k?",
        status="active",
        end_date="2026-06-01T00:00:00Z",
    )
    await db.commit()

    rows = await get_markets_by_status(db, "active")
    assert len(rows) == 1
    assert rows[0]["title"] == "Will BTC hit 100k?"

    # Upsert same market with updated volume.
    await upsert_market(
        db,
        platform="polymarket",
        market_id="cond_abc",
        title="Will BTC hit 100k?",
        status="active",
        volume=50000.0,
    )
    await db.commit()

    rows = await get_markets_by_status(db, "active")
    assert len(rows) == 1
    assert rows[0]["volume"] == 50000.0


@pytest.mark.asyncio
async def test_mark_resolved(db):
    await upsert_market(
        db,
        platform="kalshi",
        market_id="TICKER-123",
        title="Some market",
        status="active",
        end_date="2025-01-01T00:00:00Z",
    )
    await db.commit()

    await mark_resolved(db, "kalshi", "TICKER-123", "YES")
    await db.commit()

    rows = await get_markets_by_status(db, "resolved")
    assert len(rows) == 1
    assert rows[0]["resolution"] == "YES"
    assert rows[0]["resolved_at"] is not None


@pytest.mark.asyncio
async def test_mark_closed(db):
    await upsert_market(
        db,
        platform="polymarket",
        market_id="cond_xyz",
        title="Test",
        status="active",
        end_date="2025-01-01T00:00:00Z",
    )
    await db.commit()

    await mark_closed(db, "polymarket", "cond_xyz")
    await db.commit()

    rows = await get_markets_by_status(db, "closed")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_unresolved_past_end(db):
    # Market with past end_date.
    await upsert_market(
        db,
        platform="polymarket",
        market_id="old1",
        title="Old market",
        status="active",
        end_date="2020-01-01T00:00:00Z",
    )
    # Market with future end_date.
    await upsert_market(
        db,
        platform="polymarket",
        market_id="future1",
        title="Future market",
        status="active",
        end_date="2099-01-01T00:00:00Z",
    )
    await db.commit()

    candidates = await get_unresolved_past_end(db)
    ids = [c["market_id"] for c in candidates]
    assert "old1" in ids
    assert "future1" not in ids


@pytest.mark.asyncio
async def test_insert_snapshot(db):
    await insert_snapshot(
        db,
        market_id="cond_abc",
        platform="polymarket",
        yes_price=0.65,
        no_price=0.35,
        volume=1000.0,
        liquidity=500.0,
        spread=0.30,
    )
    await db.commit()

    rows = await db.execute_fetchall("SELECT * FROM price_snapshots")
    assert len(rows) == 1
    assert rows[0]["yes_price"] == 0.65


@pytest.mark.asyncio
async def test_insert_snapshots_bulk(db):
    snaps = [
        dict(market_id="a", platform="kalshi", yes_price=0.5, no_price=0.5, volume=100, liquidity=50, spread=0.0),
        dict(market_id="b", platform="kalshi", yes_price=0.7, no_price=0.3, volume=200, liquidity=80, spread=0.4),
    ]
    count = await insert_snapshots_bulk(db, snaps)
    await db.commit()
    assert count == 2

    rows = await db.execute_fetchall("SELECT * FROM price_snapshots")
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_stats(db):
    await upsert_market(db, platform="polymarket", market_id="m1", title="A", status="active")
    await upsert_market(db, platform="polymarket", market_id="m2", title="B", status="active")
    await upsert_market(db, platform="kalshi", market_id="m3", title="C", status="resolved")
    await insert_snapshot(db, market_id="m1", platform="polymarket", yes_price=0.5, no_price=0.5)
    await db.commit()

    s = await stats(db)
    assert s["active"] == 2
    assert s["resolved"] == 1
    assert s["snapshots"] == 1
