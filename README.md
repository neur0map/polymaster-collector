# polymaster-collector

Standalone Python daemon that catalogs prediction markets from Polymarket and Kalshi, tracks prices over time, records resolutions, and exports training-ready datasets for ML models.

## What it does

- **Discovers** active markets from Polymarket (Gamma API) and Kalshi every 5 minutes
- **Snapshots** prices on every discover cycle + dedicated refresh for a subset
- **Resolves** markets that have closed — records YES/NO outcomes
- **Backfills** historical resolved markets daily (auto on startup)
- **Exports** training data in three formats: Parquet (XGBoost), GRPO, SFT (MLX chat)
- **Joins** whale alerts from wwatcher.db when available
- **Storage guard** — auto-stops if disk hits 90%

## Requirements

- Python 3.11+
- No API keys needed (both APIs are public)

## Setup

```bash
# 1. Clone
git clone https://github.com/neur0map/polymaster-collector.git
cd polymaster-collector

# 2. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Create config
mkdir -p ~/.config/polymaster-collector
cp config.example.toml ~/.config/polymaster-collector/config.toml
```

Edit `~/.config/polymaster-collector/config.toml` if you need to change paths:

```toml
[general]
db_path = "~/.config/polymaster-collector/collector.db"  # change this to put DB on a larger drive

[export]
wwatcher_db_path = "~/.config/wwatcher/wwatcher.db"  # path to wwatcher DB for whale alert joins
```

## Running

```bash
# Start daemon (foreground)
collector run

# Start daemon (background, survives SSH disconnect)
nohup collector run >> ~/.config/polymaster-collector/collector.log 2>&1 &

# Check status
collector status

# List categories
collector categories

# Manual backfill
collector backfill
```

## Exporting training data

```bash
# Parquet (XGBoost features: whale stats, price time-series, market fundamentals)
collector export --format parquet

# GRPO prompts ({"prompt": "...", "outcome": 1} — for reward model training)
collector export --format grpo

# SFT (MLX chat format with <think>/<prediction> tags — for supervised fine-tuning)
collector export --format sft

# Filter by category or platform
collector export --format grpo --category "Crypto"
collector export --format sft --platform polymarket
```

Exports go to `./exports/` by default (configurable in config.toml).

## Moving to another machine

### Move everything (code + data)

```bash
# On the old machine — find your DB
grep db_path ~/.config/polymaster-collector/config.toml

# Copy the DB file (and WAL files if they exist)
scp /path/to/collector.db newhost:/path/to/collector.db
scp /path/to/collector.db-wal newhost:/path/to/collector.db-wal 2>/dev/null
scp /path/to/collector.db-shm newhost:/path/to/collector.db-shm 2>/dev/null

# On the new machine — clone and install
git clone https://github.com/neur0map/polymaster-collector.git
cd polymaster-collector
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Create config pointing to where you put the DB
mkdir -p ~/.config/polymaster-collector
cp config.example.toml ~/.config/polymaster-collector/config.toml
# Edit config.toml → set db_path to where you copied collector.db

# Start
nohup collector run >> ~/.config/polymaster-collector/collector.log 2>&1 &
```

### Move data only (fresh install, keep collected data)

```bash
# Just copy the .db file — it contains everything:
#   - All discovered markets
#   - All price snapshots
#   - All resolutions
#   - Market metadata (categories, volumes, end dates)
scp oldhost:/path/to/collector.db /mnt/storage/polymaster-collector/collector.db
```

The daemon will pick up where it left off — it upserts markets, so no duplicates.

### Putting DB on a separate drive

If your boot drive is small, point the DB to a larger mount:

```toml
[general]
db_path = "/mnt/storage/polymaster-collector/collector.db"
```

Make sure the directory exists and is writable by your user.

## Running as a systemd service

```bash
# Copy service file
sudo cp systemd/polymaster-collector.service /etc/systemd/system/

# Edit the service file to set your username and paths
sudo systemctl edit polymaster-collector

# Enable and start
sudo systemctl enable polymaster-collector
sudo systemctl start polymaster-collector

# Check logs
journalctl -u polymaster-collector -f
```

## Database schema

Single SQLite file with three tables:

- **markets** — market metadata, status (active/closed/resolved), resolution (YES/NO)
- **price_snapshots** — time-series price data (yes_price, no_price, volume, spread)
- **news_context** — headline storage (requires SearXNG, disabled by default)

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `general.db_path` | `~/.config/polymaster-collector/collector.db` | SQLite database location |
| `general.poll_interval_minutes` | `5` | How often to discover new markets |
| `general.snapshot_interval_minutes` | `15` | Dedicated price refresh interval |
| `general.resolve_interval_minutes` | `30` | How often to check for resolutions |
| `general.backfill_interval_hours` | `24` | Auto-backfill historical markets |
| `polymarket.enabled` | `true` | Enable Polymarket collection |
| `kalshi.enabled` | `true` | Enable Kalshi collection |
| `export.output_dir` | `./exports` | Where export files are written |
| `export.wwatcher_db_path` | `~/.config/wwatcher/wwatcher.db` | wwatcher DB for whale alert joins |
| `news.enabled` | `false` | Enable SearXNG headline fetching |

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
