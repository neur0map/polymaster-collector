"""SQLite schema, migrations, and async query helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS markets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    platform    TEXT NOT NULL,
    market_id   TEXT NOT NULL,
    slug        TEXT,
    title       TEXT,
    description TEXT,
    category    TEXT,
    outcomes    TEXT,
    volume      REAL,
    liquidity   REAL,
    end_date    TEXT,
    status      TEXT DEFAULT 'active',
    resolution  TEXT,
    resolved_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(platform, market_id)
);

CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);
CREATE INDEX IF NOT EXISTS idx_markets_platform ON markets(platform);
CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT NOT NULL,
    platform    TEXT NOT NULL,
    yes_price   REAL,
    no_price    REAL,
    volume      REAL,
    liquidity   REAL,
    spread      REAL,
    snapshot_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market ON price_snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON price_snapshots(snapshot_at);

CREATE TABLE IF NOT EXISTS news_context (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT NOT NULL,
    headline    TEXT,
    source      TEXT,
    captured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_news_market ON news_context(market_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def init_db(db_path: str) -> aiosqlite.Connection:
    """Open (or create) the database and apply schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA_SQL)
    await db.execute(
        "INSERT OR IGNORE INTO _meta (key, value) VALUES (?, ?)",
        ("schema_version", str(_SCHEMA_VERSION)),
    )
    await db.commit()
    log.info("Database ready: %s", path)
    return db


# ---------------------------------------------------------------------------
# Markets helpers
# ---------------------------------------------------------------------------

async def upsert_market(db: aiosqlite.Connection, **kw: Any) -> None:
    """Insert or update a market row.  Pass column names as keyword args."""
    now = _now_iso()
    kw.setdefault("created_at", now)
    kw["updated_at"] = now
    if "outcomes" in kw and not isinstance(kw["outcomes"], str):
        kw["outcomes"] = json.dumps(kw["outcomes"])

    cols = list(kw.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    update_clause = ", ".join(
        f"{c} = excluded.{c}" for c in cols if c not in ("platform", "market_id", "created_at")
    )
    sql = (
        f"INSERT INTO markets ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT(platform, market_id) DO UPDATE SET {update_clause}"
    )
    await db.execute(sql, list(kw.values()))


async def get_markets_by_status(
    db: aiosqlite.Connection, status: str, platform: str | None = None
) -> list[dict]:
    sql = "SELECT * FROM markets WHERE status = ?"
    params: list[Any] = [status]
    if platform:
        sql += " AND platform = ?"
        params.append(platform)
    rows = await db.execute_fetchall(sql, params)
    return [dict(r) for r in rows]


async def get_unresolved_past_end(db: aiosqlite.Connection) -> list[dict]:
    """Markets past end_date that haven't been resolved yet."""
    now = _now_iso()
    sql = (
        "SELECT * FROM markets "
        "WHERE status IN ('active', 'closed') "
        "AND end_date IS NOT NULL AND end_date <= ? "
        "ORDER BY end_date"
    )
    rows = await db.execute_fetchall(sql, [now])
    return [dict(r) for r in rows]


async def mark_resolved(
    db: aiosqlite.Connection, platform: str, market_id: str, resolution: str
) -> None:
    now = _now_iso()
    await db.execute(
        "UPDATE markets SET status = 'resolved', resolution = ?, resolved_at = ?, updated_at = ? "
        "WHERE platform = ? AND market_id = ?",
        (resolution, now, now, platform, market_id),
    )


async def mark_closed(db: aiosqlite.Connection, platform: str, market_id: str) -> None:
    now = _now_iso()
    await db.execute(
        "UPDATE markets SET status = 'closed', updated_at = ? "
        "WHERE platform = ? AND market_id = ?",
        (now, platform, market_id),
    )


# ---------------------------------------------------------------------------
# Price snapshots
# ---------------------------------------------------------------------------

async def insert_snapshot(db: aiosqlite.Connection, **kw: Any) -> None:
    kw.setdefault("snapshot_at", _now_iso())
    cols = list(kw.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT INTO price_snapshots ({col_names}) VALUES ({placeholders})"
    await db.execute(sql, list(kw.values()))


async def insert_snapshots_bulk(
    db: aiosqlite.Connection, rows: Sequence[dict]
) -> int:
    if not rows:
        return 0
    now = _now_iso()
    for r in rows:
        r.setdefault("snapshot_at", now)
    cols = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT INTO price_snapshots ({col_names}) VALUES ({placeholders})"
    await db.executemany(sql, [list(r.values()) for r in rows])
    return len(rows)


# ---------------------------------------------------------------------------
# News context
# ---------------------------------------------------------------------------

async def insert_news(db: aiosqlite.Connection, **kw: Any) -> None:
    kw.setdefault("captured_at", _now_iso())
    cols = list(kw.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT INTO news_context ({col_names}) VALUES ({placeholders})"
    await db.execute(sql, list(kw.values()))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def stats(db: aiosqlite.Connection) -> dict:
    """Return summary counts for the status command."""
    result: dict[str, Any] = {}
    for status in ("active", "closed", "resolved"):
        row = await db.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM markets WHERE status = ?", [status]
        )
        result[status] = row[0]["cnt"] if row else 0

    row = await db.execute_fetchall("SELECT COUNT(*) AS cnt FROM price_snapshots")
    result["snapshots"] = row[0]["cnt"] if row else 0

    row = await db.execute_fetchall("SELECT COUNT(*) AS cnt FROM news_context")
    result["news"] = row[0]["cnt"] if row else 0

    return result
