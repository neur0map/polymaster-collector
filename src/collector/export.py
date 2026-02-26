"""Export resolved markets to Parquet and Turtel-style prompt format.

Every whale alert is tagged with:
  - correct (bool): did the whale's side match the market resolution?
  - profit_per_unit (float): how much profit per $1 of exposure
    (e.g. bought YES at 0.60, resolved YES → profit = 0.40)

Per-market aggregates computed for XGBoost:
  - whale_correct_count / whale_incorrect_count / whale_accuracy
  - whale_consensus_correct (bool): did the majority-side match resolution?
  - whale_total_value / whale_avg_value
  - whale_avg_entry_price / whale_consensus_strength (% that agree)
  - price time-series features: mean, std, trend (linear slope), min, max
  - price_move (close − open)
  - market_consensus_at_close (last yes_price before resolution)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from collector.config import Config

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Parquet export
# ------------------------------------------------------------------

async def export_parquet(
    db, cfg: Config, *, category: str | None = None, platform: str | None = None,
) -> str:
    """Export resolved markets + snapshots + wwatcher alerts to Parquet.

    Optional *category* and *platform* filters narrow the export so you can
    produce training sets for specific domains (e.g. weather-only).
    """
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if category:
        suffix += f"_{_slug(category)}"
    if platform:
        suffix += f"_{platform}"
    out_path = str(output_dir / f"resolved_{ts}{suffix}.parquet")

    # Fetch resolved markets with optional filters.
    sql = "SELECT * FROM markets WHERE status = 'resolved' AND resolution IS NOT NULL"
    params: list[Any] = []
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    if platform:
        sql += " AND platform = ?"
        params.append(platform)
    rows = await db.execute_fetchall(sql, params)
    markets = [dict(r) for r in rows]
    if not markets:
        log.warning("No resolved markets to export")
        return out_path

    # Fetch price histories per market.
    price_map: dict[str, list[dict]] = {}
    for mkt in markets:
        mid = mkt["market_id"]
        snap_rows = await db.execute_fetchall(
            "SELECT yes_price, no_price, volume, liquidity, spread, snapshot_at "
            "FROM price_snapshots WHERE market_id = ? ORDER BY snapshot_at",
            [mid],
        )
        price_map[mid] = [dict(r) for r in snap_rows]

    # Fetch news headlines.
    news_map: dict[str, list[dict]] = {}
    for mkt in markets:
        mid = mkt["market_id"]
        news_rows = await db.execute_fetchall(
            "SELECT headline, source, captured_at "
            "FROM news_context WHERE market_id = ? ORDER BY captured_at",
            [mid],
        )
        news_map[mid] = [dict(r) for r in news_rows]

    # Link wwatcher alerts (optional).
    whale_map = _load_wwatcher_alerts(cfg.export.wwatcher_db_path, markets)

    # Build rows.
    records: list[dict[str, Any]] = []
    for mkt in markets:
        mid = mkt["market_id"]
        resolution = mkt["resolution"]
        prices = price_map.get(mid, [])
        first_price = prices[0]["yes_price"] if prices else None
        last_price = prices[-1]["yes_price"] if prices else None

        # --- Enrich whale alerts with correctness + profit ---
        raw_alerts = whale_map.get(mid, [])
        enriched_alerts = _enrich_alerts(raw_alerts, resolution, prices)

        # --- Per-market whale aggregates for XGBoost ---
        whale_stats = _compute_whale_stats(enriched_alerts, resolution)

        # --- Price time-series features ---
        ts_features = _compute_price_features(prices)

        records.append(dict(
            # identifiers
            market_id=mid,
            platform=mkt["platform"],
            title=mkt["title"],
            category=mkt.get("category", ""),
            outcomes=mkt.get("outcomes", ""),

            # label
            resolution=resolution,
            resolved_at=mkt.get("resolved_at"),

            # market fundamentals
            volume=mkt.get("volume"),
            liquidity=mkt.get("liquidity"),
            end_date=mkt.get("end_date"),
            price_at_open=first_price,
            price_at_close=last_price,
            price_move=_safe_sub(last_price, first_price),
            market_consensus_at_close=last_price,
            days_to_resolution=_days_to_resolution(mkt),

            # price time-series features
            price_mean=ts_features["mean"],
            price_std=ts_features["std"],
            price_min=ts_features["min"],
            price_max=ts_features["max"],
            price_trend=ts_features["trend"],
            volume_mean=ts_features["volume_mean"],
            volume_max=ts_features["volume_max"],
            spread_mean=ts_features["spread_mean"],
            snapshot_count=ts_features["count"],

            # whale features (XGBoost-ready)
            whale_count=whale_stats["count"],
            whale_correct_count=whale_stats["correct_count"],
            whale_incorrect_count=whale_stats["incorrect_count"],
            whale_accuracy=whale_stats["accuracy"],
            whale_net_direction=whale_stats["net_direction"],
            whale_consensus_correct=whale_stats["consensus_correct"],
            whale_consensus_strength=whale_stats["consensus_strength"],
            whale_total_value=whale_stats["total_value"],
            whale_avg_value=whale_stats["avg_value"],
            whale_max_value=whale_stats["max_value"],
            whale_avg_entry_price=whale_stats["avg_entry_price"],
            whale_avg_win_rate=whale_stats["avg_win_rate"],
            whale_avg_profit_per_unit=whale_stats["avg_profit_per_unit"],
            whale_unique_wallets=whale_stats["unique_wallets"],
            whale_repeat_actors=whale_stats["repeat_actors"],

            # raw data (JSON blobs for deeper analysis / LLM context)
            price_history=json.dumps(prices),
            whale_alerts=json.dumps(enriched_alerts),
            news_headlines=json.dumps(news_map.get(mid, [])),
        ))

    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)
    log.info("Exported %d resolved markets to %s", len(records), out_path)
    return out_path


# ------------------------------------------------------------------
# Prompt export (Turtel et al. format for GRPO)
# ------------------------------------------------------------------

async def _fetch_causal_context(
    db, mkt: dict, whale_map: dict[str, list[dict]],
) -> dict:
    """Gather causally-masked context for a single market (shared by GRPO & SFT)."""
    mid = mkt["market_id"]
    resolved_at = mkt.get("resolved_at")

    # --- Price history (pre-resolution only) ---
    snap_rows = await db.execute_fetchall(
        "SELECT yes_price, no_price, volume, spread, snapshot_at "
        "FROM price_snapshots WHERE market_id = ? "
        "AND snapshot_at < COALESCE(?, snapshot_at) "
        "ORDER BY snapshot_at",
        [mid, resolved_at],
    )
    price_history = [dict(r) for r in snap_rows]

    # --- News (pre-resolution only) ---
    news_rows = await db.execute_fetchall(
        "SELECT headline, source, captured_at "
        "FROM news_context WHERE market_id = ? "
        "AND captured_at < COALESCE(?, captured_at) "
        "ORDER BY captured_at",
        [mid, resolved_at],
    )
    news = [dict(r) for r in news_rows]

    # --- Whale alerts (strip correctness — model must predict) ---
    raw_alerts = whale_map.get(mid, [])
    masked_alerts = []
    for a in raw_alerts:
        masked = {
            k: v for k, v in a.items()
            if k not in ("correct", "profit_per_unit")
        }
        # Causal mask: skip alerts after resolution.
        alert_ts = a.get("created_at") or a.get("timestamp")
        if alert_ts and resolved_at:
            if _ts_ord(alert_ts) >= _ts_ord(resolved_at):
                continue
        masked_alerts.append(masked)

    # --- Summary stats the model can reason over ---
    ts_features = _compute_price_features(
        [{"yes_price": s.get("yes_price"), "volume": s.get("volume"),
          "spread": s.get("spread")} for s in price_history]
    )

    whale_sides = [a.get("side", "").upper() for a in masked_alerts if a.get("side")]
    yes_whales = sum(1 for s in whale_sides if s == "YES")
    no_whales = len(whale_sides) - yes_whales

    return {
        "platform": mkt["platform"],
        "category": mkt.get("category", ""),
        "volume": mkt.get("volume"),
        "liquidity": mkt.get("liquidity"),
        "current_yes_price": price_history[-1]["yes_price"] if price_history else None,
        "price_trend": ts_features["trend"],
        "price_mean": ts_features["mean"],
        "price_history": price_history,
        "whale_count": len(masked_alerts),
        "whale_yes_count": yes_whales,
        "whale_no_count": no_whales,
        "whale_alerts": masked_alerts,
        "news_headlines": news,
    }


def _resolution_to_int(resolution: str) -> int:
    """Convert resolution string to integer (1=YES, 0=NO)."""
    return 1 if resolution.upper() == "YES" else 0


async def export_prompts(
    db, cfg: Config, *, category: str | None = None, platform: str | None = None,
) -> str:
    """Export GRPO prompt format (matches prowl-bot plan).

    Format: ``{"prompt": "...", "outcome": 1}``
    Context is causally masked: only data from *before* resolution is
    included so the model can't cheat.
    """
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if category:
        suffix += f"_{_slug(category)}"
    if platform:
        suffix += f"_{platform}"
    out_path = str(output_dir / f"grpo_{ts}{suffix}.jsonl")

    sql = "SELECT * FROM markets WHERE status = 'resolved' AND resolution IS NOT NULL"
    params: list[Any] = []
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    if platform:
        sql += " AND platform = ?"
        params.append(platform)
    rows = await db.execute_fetchall(sql, params)
    markets = [dict(r) for r in rows]

    whale_map = _load_wwatcher_alerts(cfg.export.wwatcher_db_path, markets)

    with open(out_path, "w") as fh:
        for mkt in markets:
            ctx = await _fetch_causal_context(db, mkt, whale_map)

            record = {
                "prompt": (
                    f"Market: {mkt['title']}\n"
                    f"Price: {ctx['current_yes_price']}\n"
                    f"Platform: {ctx['platform']}\n"
                    f"Category: {ctx['category']}\n"
                    f"Volume: {ctx['volume']}\n"
                    f"Whale consensus: {ctx['whale_yes_count']} YES / {ctx['whale_no_count']} NO\n"
                    f"Price trend: {ctx['price_trend']}\n"
                    f"Headlines: {json.dumps([h.get('headline', '') for h in ctx['news_headlines']])}"
                ),
                "outcome": _resolution_to_int(mkt["resolution"]),
            }
            fh.write(json.dumps(record) + "\n")

    log.info("Exported %d GRPO prompts to %s", len(markets), out_path)
    return out_path


# ------------------------------------------------------------------
# SFT export (MLX chat format)
# ------------------------------------------------------------------

_SFT_SYSTEM_PROMPT = (
    "You are a prediction market analyst. Given market information, whale "
    "activity, and news headlines, predict the probability that the market "
    "resolves YES. Reason step-by-step inside <think>...</think> tags, then "
    "give your probability inside <prediction>...</prediction> tags."
)


async def export_sft(
    db, cfg: Config, *, category: str | None = None, platform: str | None = None,
) -> str:
    """Export SFT training data in MLX chat format (matches prowl-bot plan).

    Format::

        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "Market: ... Price: ... Headlines: ..."},
            {"role": "assistant", "content": "<think>...</think>\n<prediction>0.72</prediction>"}
        ]}
    """
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if category:
        suffix += f"_{_slug(category)}"
    if platform:
        suffix += f"_{platform}"
    out_path = str(output_dir / f"sft_{ts}{suffix}.jsonl")

    sql = "SELECT * FROM markets WHERE status = 'resolved' AND resolution IS NOT NULL"
    params: list[Any] = []
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    if platform:
        sql += " AND platform = ?"
        params.append(platform)
    rows = await db.execute_fetchall(sql, params)
    markets = [dict(r) for r in rows]

    whale_map = _load_wwatcher_alerts(cfg.export.wwatcher_db_path, markets)

    with open(out_path, "w") as fh:
        for mkt in markets:
            ctx = await _fetch_causal_context(db, mkt, whale_map)
            outcome_int = _resolution_to_int(mkt["resolution"])

            user_content = (
                f"Market: {mkt['title']}\n"
                f"Price: {ctx['current_yes_price']}\n"
                f"Platform: {ctx['platform']}\n"
                f"Category: {ctx['category']}\n"
                f"Volume: {ctx['volume']}\n"
                f"Liquidity: {ctx['liquidity']}\n"
                f"Whale activity: {ctx['whale_count']} trades "
                f"({ctx['whale_yes_count']} YES / {ctx['whale_no_count']} NO)\n"
                f"Price trend: {ctx['price_trend']}\n"
                f"Price mean: {ctx['price_mean']}\n"
                f"Headlines: {json.dumps([h.get('headline', '') for h in ctx['news_headlines']])}"
            )

            # Build a plausible assistant response anchored to the actual outcome.
            prob = 1.0 if outcome_int == 1 else 0.0
            # Use the market's price as a more realistic probability when available.
            if ctx["current_yes_price"] is not None:
                # Nudge toward the true outcome (the model should be more confident
                # than the market, since it has the whale signal).
                mkt_price = ctx["current_yes_price"]
                if outcome_int == 1:
                    prob = max(mkt_price, 0.70)  # confident YES
                else:
                    prob = min(mkt_price, 0.30)  # confident NO

            resolution_word = "YES" if outcome_int == 1 else "NO"
            assistant_content = (
                f"<think>Based on whale activity showing "
                f"{ctx['whale_yes_count']} YES / {ctx['whale_no_count']} NO "
                f"positions and a price trend of {ctx['price_trend']}, "
                f"the market is leaning toward {resolution_word}.</think>\n"
                f"<prediction>{prob:.2f}</prediction>"
            )

            record = {
                "messages": [
                    {"role": "system", "content": _SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            fh.write(json.dumps(record) + "\n")

    log.info("Exported %d SFT examples to %s", len(markets), out_path)
    return out_path


# ------------------------------------------------------------------
# Alert enrichment
# ------------------------------------------------------------------

def _enrich_alerts(
    alerts: list[dict], resolution: str, prices: list[dict]
) -> list[dict]:
    """Tag each whale alert with correctness and estimated profit."""
    enriched = []
    for a in alerts:
        a = dict(a)  # don't mutate original

        side = (a.get("side") or "").strip().upper()
        if side in ("YES", "NO"):
            a["correct"] = side == resolution
        else:
            a["correct"] = None

        # Estimate profit: if whale bought YES at entry_price and it resolved YES,
        # profit per unit = 1.0 - entry_price.  If it resolved NO, loss = -entry_price.
        entry_price = _get_entry_price(a, prices)
        a["entry_price"] = entry_price
        if entry_price is not None and side in ("YES", "NO"):
            if side == "YES":
                a["profit_per_unit"] = (1.0 - entry_price) if resolution == "YES" else -entry_price
            else:  # side == NO
                # Bought NO at (1 - yes_price), pays 1.0 if NO wins.
                no_entry = 1.0 - entry_price
                a["profit_per_unit"] = (1.0 - no_entry) if resolution == "NO" else -no_entry
        else:
            a["profit_per_unit"] = None

        enriched.append(a)
    return enriched


def _get_entry_price(alert: dict, prices: list[dict]) -> float | None:
    """Get the yes_price at the time of the whale's trade.

    Uses the alert's own 'price' field if available, otherwise interpolates
    from the nearest price snapshot.
    """
    # Prefer the alert's own price field (normalised to 0-1).
    raw_price = alert.get("price")
    if raw_price is not None:
        try:
            p = float(raw_price)
            return p / 100.0 if p > 1.0 else p
        except (ValueError, TypeError):
            pass

    # Fall back to nearest snapshot.
    alert_ts = alert.get("created_at") or alert.get("timestamp")
    if not alert_ts or not prices:
        return None

    best = None
    best_diff = float("inf")
    for snap in prices:
        snap_ts = snap.get("snapshot_at", "")
        if not snap_ts:
            continue
        diff = abs(_ts_ord(snap_ts) - _ts_ord(alert_ts))
        if diff < best_diff:
            best_diff = diff
            best = snap
    return best["yes_price"] if best else None


def _ts_ord(value: str | int | float) -> float:
    """Convert a timestamp (ISO string *or* unix int) to seconds-since-epoch."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return datetime.strptime(str(value)[:19] + "Z", fmt).timestamp()
    except Exception:
        return 0.0


# ------------------------------------------------------------------
# Per-market whale aggregates (XGBoost features)
# ------------------------------------------------------------------

def _compute_whale_stats(alerts: list[dict], resolution: str) -> dict[str, Any]:
    """Compute aggregate whale features for one market."""
    if not alerts:
        return {
            "count": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "accuracy": None,
            "net_direction": None,
            "consensus_correct": None,
            "consensus_strength": None,
            "total_value": None,
            "avg_value": None,
            "max_value": None,
            "avg_entry_price": None,
            "avg_win_rate": None,
            "avg_profit_per_unit": None,
            "unique_wallets": 0,
            "repeat_actors": 0,
        }

    n = len(alerts)
    correct = [a for a in alerts if a.get("correct") is True]
    incorrect = [a for a in alerts if a.get("correct") is False]

    # Net direction.
    yes_count = sum(1 for a in alerts if (a.get("side") or "").upper() == "YES")
    no_count = n - yes_count
    if yes_count > no_count:
        net_dir = "YES"
    elif no_count > yes_count:
        net_dir = "NO"
    else:
        net_dir = "MIXED"

    # Consensus: was the majority side correct?
    consensus_correct = net_dir == resolution if net_dir != "MIXED" else None
    consensus_strength = max(yes_count, no_count) / n if n else None

    # Value aggregates.
    values = [a["value"] for a in alerts if a.get("value") is not None]
    total_value = sum(values) if values else None
    avg_value = _safe_mean(values)
    max_value = max(values) if values else None

    # Entry price.
    entries = [a["entry_price"] for a in alerts if a.get("entry_price") is not None]
    avg_entry = _safe_mean(entries)

    # Win rate (from whale profile, not computed).
    win_rates = [a["win_rate"] for a in alerts if a.get("win_rate") is not None]
    avg_wr = _safe_mean(win_rates)

    # Profit.
    profits = [a["profit_per_unit"] for a in alerts if a.get("profit_per_unit") is not None]
    avg_profit = _safe_mean(profits)

    # Unique wallets / repeat actors.
    wallets = [a.get("wallet") or a.get("wallet_address") or a.get("address")
               for a in alerts]
    wallets = [w for w in wallets if w]
    unique = set(wallets)
    repeat = sum(1 for w in unique if wallets.count(w) > 1)

    return {
        "count": n,
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "accuracy": len(correct) / n if n else None,
        "net_direction": net_dir,
        "consensus_correct": consensus_correct,
        "consensus_strength": consensus_strength,
        "total_value": total_value,
        "avg_value": avg_value,
        "max_value": max_value,
        "avg_entry_price": avg_entry,
        "avg_win_rate": avg_wr,
        "avg_profit_per_unit": avg_profit,
        "unique_wallets": len(unique),
        "repeat_actors": repeat,
    }


# ------------------------------------------------------------------
# Price time-series features
# ------------------------------------------------------------------

def _compute_price_features(prices: list[dict]) -> dict[str, Any]:
    """Compute summary statistics over the price snapshot series."""
    empty = {
        "mean": None, "std": None, "min": None, "max": None,
        "trend": None, "volume_mean": None, "volume_max": None,
        "spread_mean": None, "count": 0,
    }
    if not prices:
        return empty

    yes_prices = [p["yes_price"] for p in prices if p.get("yes_price") is not None]
    if not yes_prices:
        return empty

    n = len(yes_prices)
    result: dict[str, Any] = {
        "count": n,
        "mean": statistics.mean(yes_prices),
        "min": min(yes_prices),
        "max": max(yes_prices),
        "std": statistics.stdev(yes_prices) if n >= 2 else 0.0,
    }

    # Linear trend (OLS slope over index 0..n-1).
    if n >= 2:
        x_mean = (n - 1) / 2.0
        y_mean = result["mean"]
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(yes_prices))
        den = sum((i - x_mean) ** 2 for i in range(n))
        result["trend"] = num / den if den else 0.0
    else:
        result["trend"] = 0.0

    # Volume stats.
    volumes = [p["volume"] for p in prices if p.get("volume") is not None]
    result["volume_mean"] = _safe_mean(volumes)
    result["volume_max"] = max(volumes) if volumes else None

    # Spread stats.
    spreads = [p["spread"] for p in prices if p.get("spread") is not None]
    result["spread_mean"] = _safe_mean(spreads)

    return result


# ------------------------------------------------------------------
# wwatcher linking
# ------------------------------------------------------------------

def _normalize_wwatcher_alert(raw: dict) -> dict:
    """Map wwatcher ``alerts`` columns to the names the enrichment pipeline expects.

    wwatcher schema (db.rs):
        outcome  → side  (YES / NO)
        wallet_id → wallet
        created_at INTEGER (unix) → created_at ISO string
        timestamp  TEXT → kept as-is
    """
    a = dict(raw)
    # Side: wwatcher stores "outcome" (YES/NO), we need "side".
    if "side" not in a and "outcome" in a:
        a["side"] = a["outcome"]
    # Wallet: wwatcher uses wallet_id / wallet_hash.
    if "wallet" not in a and "wallet_id" in a:
        a["wallet"] = a["wallet_id"]
    # Timestamp: created_at is a unix integer in wwatcher.
    if "created_at" in a and isinstance(a["created_at"], (int, float)):
        try:
            a["created_at"] = datetime.fromtimestamp(
                int(a["created_at"]), tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OSError, ValueError):
            pass
    return a


def _load_wwatcher_alerts(
    wwatcher_db_path: str, markets: list[dict]
) -> dict[str, list[dict]]:
    """Load whale alerts from wwatcher DB and match to markets."""
    if not os.path.exists(wwatcher_db_path):
        log.debug("wwatcher DB not found at %s, skipping join", wwatcher_db_path)
        return {}

    result: dict[str, list[dict]] = {}
    try:
        conn = sqlite3.connect(wwatcher_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM alerts")
        alerts = [_normalize_wwatcher_alert(dict(r)) for r in cursor.fetchall()]
        conn.close()
    except Exception:
        log.warning("Failed to read wwatcher DB", exc_info=True)
        return {}

    if not alerts:
        return result

    # Build lookup by market_id for exact matching.
    alert_by_mid: dict[str, list[dict]] = {}
    unmatched: list[dict] = []
    for a in alerts:
        mid = a.get("market_id", "")
        if mid:
            alert_by_mid.setdefault(mid, []).append(a)
        else:
            unmatched.append(a)

    # Phase 1: exact match on market_id.
    for mkt in markets:
        mid = mkt["market_id"]
        if mid in alert_by_mid:
            result[mid] = alert_by_mid[mid]

    # Phase 2: fuzzy match on title for unmatched alerts.
    if unmatched:
        try:
            from thefuzz import fuzz

            title_to_mid = {
                _normalise_title(m.get("title", "")): m["market_id"]
                for m in markets
                if m.get("title")
            }
            for a in unmatched:
                alert_title = _normalise_title(a.get("market_title", ""))
                if not alert_title:
                    continue
                best_score = 0
                best_mid = None
                for t, mid in title_to_mid.items():
                    score = fuzz.token_sort_ratio(alert_title, t)
                    if score > best_score:
                        best_score = score
                        best_mid = mid
                if best_score >= 80 and best_mid:
                    result.setdefault(best_mid, []).append(a)
        except ImportError:
            log.debug("thefuzz not available, skipping fuzzy matching")

    return result


def _normalise_title(title: str) -> str:
    return title.strip().lower()


def _slug(text: str) -> str:
    """Turn a category name into a safe filename fragment."""
    return text.strip().lower().replace(" ", "_").replace("/", "_")[:40]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_mean(vals: list) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def _safe_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _days_to_resolution(mkt: dict) -> float | None:
    end = mkt.get("end_date")
    resolved = mkt.get("resolved_at")
    if not end or not resolved:
        return None
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        t0 = datetime.strptime(end[:19] + "Z", fmt)
        t1 = datetime.strptime(resolved[:19] + "Z", fmt)
        return (t1 - t0).total_seconds() / 86400
    except Exception:
        return None
