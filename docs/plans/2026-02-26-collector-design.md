# polymaster-collector

Standalone Python daemon that catalogs prediction markets from Polymarket and Kalshi, tracks prices over time, records resolutions, and exports training-ready datasets for the Prowl Bot ML pipeline.

## Context

- **polymaster/wwatcher** (Rust) watches whale-sized trades → stores in `~/.config/wwatcher/wwatcher.db`
- **polymaster-collector** (Python) watches all markets regardless of trade activity → stores in `collector.db`
- Together they answer: what markets existed, what whales did on them, and how they resolved
- Data feeds into Phase 5 (ONNX models) and Phase 6 (Self-SFT + GRPO) of Prowl Bot roadmap
- Based on Turtel et al. 2025 approach: market questions + news headlines + resolved outcomes as reward signal

## Architecture

```
Polymarket REST API ──┐
                      ├──► polymaster-collector (Python daemon)
Kalshi REST API ──────┘           │
                                  ├──► collector.db (SQLite)
                                  ├──► exports/ (Parquet)
                                  │
wwatcher.db ──────────────────────┘ (linked via market_id at export)
```

Core loop runs three phases on a timer:
1. DISCOVER (every 5 min) — fetch active markets from both platforms, upsert into markets table
2. SNAPSHOT (every 15 min) — capture current prices/volumes for tracked markets
3. RESOLVE (every 30 min) — check closed markets past end_date, record YES/NO outcome

## Database Schema (SQLite)

**markets** — one row per market
- id INTEGER PRIMARY KEY
- platform TEXT (polymarket | kalshi)
- market_id TEXT (condition_id or kalshi ticker) UNIQUE
- slug TEXT
- title TEXT
- description TEXT
- category TEXT
- outcomes TEXT (JSON array)
- volume REAL
- liquidity REAL
- end_date TEXT (ISO)
- status TEXT (active | closed | resolved)
- resolution TEXT (YES | NO | NULL)
- resolved_at TEXT (ISO)
- created_at TEXT
- updated_at TEXT

**price_snapshots** — periodic price captures for time-series
- id INTEGER PRIMARY KEY
- market_id TEXT (FK → markets.market_id)
- platform TEXT
- yes_price REAL
- no_price REAL
- volume REAL
- liquidity REAL
- spread REAL
- snapshot_at TEXT (ISO)

**news_context** — headlines at snapshot time (Phase 2, SearXNG)
- id INTEGER PRIMARY KEY
- market_id TEXT (FK → markets.market_id)
- headline TEXT
- source TEXT
- captured_at TEXT (ISO)

## Project Structure

```
polymaster-collector/
├── pyproject.toml
├── README.md
├── config.example.toml
├── src/
│   └── collector/
│       ├── __init__.py
│       ├── cli.py              # click CLI: run, status, export, backfill
│       ├── config.py           # load config.toml
│       ├── db.py               # schema, migrations, query helpers
│       ├── daemon.py           # main loop: discover → snapshot → resolve
│       ├── platforms/
│       │   ├── __init__.py
│       │   ├── polymarket.py   # CLOB + Gamma API client
│       │   └── kalshi.py       # Kalshi public API client
│       ├── news.py             # SearXNG headline fetcher (disabled by default)
│       └── export.py           # SQLite → Parquet, wwatcher join
├── systemd/
│   └── polymaster-collector.service
└── tests/
```

## CLI Commands

- `collector run` — start daemon (foreground, use systemd for background)
- `collector status` — stats: markets tracked, resolved, pending, disk usage
- `collector export` — dump resolved markets to Parquet, join wwatcher alerts
- `collector export --format=prompts` — export Turtel-style prompt format for GRPO
- `collector backfill` — one-time pull of historical resolved markets

## Platform API Details

**Polymarket:**
- Gamma API `GET /markets` — paginated list of all markets with metadata
- CLOB API `GET /prices` — current yes/no prices by condition_id
- Resolution from market object: `resolved=true`, `resolution` field
- No auth required for public endpoints
- Rate limit: conservative 2 req/sec

**Kalshi:**
- `GET /markets` — paginated, includes prices in response
- `result` field shows YES/NO on resolved markets
- No auth required for public read endpoints
- Rate limit: ~10 req/sec, use 2 req/sec to be safe

## Daemon Loop Timing

- DISCOVER: every 5 min (~2-3k markets across both platforms)
- SNAPSHOT: every 15 min (~2.5k rows per cycle, ~240 cycles/day)
- RESOLVE: every 30 min (only checks markets past end_date)
- Each cycle completes in ~30-60 seconds
- Exponential backoff on 429/5xx errors
- Idempotent: missed cycles catch up on next run

## Export Format

**Parquet (one row per resolved market):**
- market_id, platform, title, category, outcomes
- resolution (YES/NO — the label)
- resolved_at
- volume, liquidity
- price_at_open, price_at_close
- price_history (JSON array of snapshots)
- whale_alerts (JSON array of linked wwatcher alerts)
- whale_count, whale_net_direction, whale_avg_win_rate, whale_max_value
- news_headlines (JSON array, when enabled)
- days_to_resolution

**wwatcher linking:**
1. Exact match on market_id where both sides have it
2. Fuzzy match on market_title (normalized) for the rest
3. Unmatched alerts export without resolution labels

**Prompt export (for GRPO training):**
- question: "Will [title] resolve YES?"
- context: price_history + whale_alerts + news (all pre-resolution, causally masked)
- answer: resolution YES/NO (reward signal)

## Disk Estimates

- Markets table: ~50k rows/year (trivial)
- Price snapshots: ~18M rows/month → ~2GB/month SQLite
- First year total: ~25GB
- Parquet exports: ~100MB per month of resolved data

## Config (config.toml)

```toml
[general]
db_path = "~/.config/polymaster-collector/collector.db"
poll_interval_minutes = 5
snapshot_interval_minutes = 15
resolve_interval_minutes = 30

[polymarket]
enabled = true
base_url = "https://clob.polymarket.com"
gamma_url = "https://gamma-api.polymarket.com"

[kalshi]
enabled = true
base_url = "https://api.elections.kalshi.com/trade-api/v2"

[export]
output_dir = "./exports"
wwatcher_db_path = "~/.config/wwatcher/wwatcher.db"

[news]
enabled = false
searxng_url = "http://localhost:8080"
```

## systemd Service

```ini
[Unit]
Description=polymaster-collector daemon
After=network.target

[Service]
Type=simple
User=neur0map
ExecStart=/home/neur0map/.local/bin/collector run
Restart=always
RestartSec=30
Environment=COLLECTOR_CONFIG=/home/neur0map/.config/polymaster-collector/config.toml

[Install]
WantedBy=multi-user.target
```

## Dependencies

- Python 3.11+
- httpx (async HTTP client)
- click (CLI framework)
- tomli (config parsing)
- pyarrow (Parquet export)
- thefuzz (fuzzy title matching for wwatcher join)
- aiosqlite (async SQLite for daemon)

## Implementation Order

1. Project scaffold: pyproject.toml, config, CLI skeleton
2. db.py: schema creation, migration, query helpers
3. polymarket.py: market discovery + price fetching + resolution checking
4. kalshi.py: same for Kalshi
5. daemon.py: main loop wiring discover → snapshot → resolve
6. cli.py: `run` and `status` commands
7. backfill command: historical resolved markets
8. export.py: Parquet export + wwatcher joining
9. systemd service file
10. Tests + README
