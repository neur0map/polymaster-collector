"""Tests for export enrichment logic."""

from __future__ import annotations

import pytest

from collector.export import (
    _compute_price_features,
    _compute_whale_stats,
    _enrich_alerts,
    _normalize_wwatcher_alert,
    _resolution_to_int,
    _ts_ord,
)


# ------------------------------------------------------------------
# Alert enrichment
# ------------------------------------------------------------------

class TestEnrichAlerts:
    def test_correct_yes(self):
        alerts = [{"side": "YES", "price": 0.60}]
        result = _enrich_alerts(alerts, "YES", [])
        assert result[0]["correct"] is True
        assert result[0]["profit_per_unit"] == pytest.approx(0.40)

    def test_incorrect_yes(self):
        alerts = [{"side": "YES", "price": 0.60}]
        result = _enrich_alerts(alerts, "NO", [])
        assert result[0]["correct"] is False
        assert result[0]["profit_per_unit"] == pytest.approx(-0.60)

    def test_correct_no(self):
        alerts = [{"side": "NO", "price": 0.60}]
        result = _enrich_alerts(alerts, "NO", [])
        assert result[0]["correct"] is True
        # Bought NO at (1 - 0.60) = 0.40, pays 1.0 → profit = 0.60
        assert result[0]["profit_per_unit"] == pytest.approx(0.60)

    def test_incorrect_no(self):
        alerts = [{"side": "NO", "price": 0.60}]
        result = _enrich_alerts(alerts, "YES", [])
        assert result[0]["correct"] is False
        # Bought NO at 0.40, lost → profit = -0.40
        assert result[0]["profit_per_unit"] == pytest.approx(-0.40)

    def test_no_side(self):
        alerts = [{"value": 1000}]
        result = _enrich_alerts(alerts, "YES", [])
        assert result[0]["correct"] is None
        assert result[0]["profit_per_unit"] is None

    def test_entry_price_from_snapshot(self):
        alerts = [{"side": "YES", "timestamp": "2026-01-15T12:00:00Z"}]
        prices = [
            {"yes_price": 0.50, "snapshot_at": "2026-01-15T11:00:00Z"},
            {"yes_price": 0.55, "snapshot_at": "2026-01-15T12:05:00Z"},
            {"yes_price": 0.60, "snapshot_at": "2026-01-15T13:00:00Z"},
        ]
        result = _enrich_alerts(alerts, "YES", prices)
        # Nearest snapshot is 12:05 → entry_price = 0.55
        assert result[0]["entry_price"] == pytest.approx(0.55)

    def test_does_not_mutate_original(self):
        alerts = [{"side": "YES", "price": 0.70}]
        _enrich_alerts(alerts, "YES", [])
        assert "correct" not in alerts[0]


# ------------------------------------------------------------------
# Whale stats
# ------------------------------------------------------------------

class TestComputeWhaleStats:
    def test_empty(self):
        s = _compute_whale_stats([], "YES")
        assert s["count"] == 0
        assert s["accuracy"] is None

    def test_all_correct(self):
        alerts = [
            {"side": "YES", "correct": True, "value": 1000, "price": 0.60,
             "entry_price": 0.60, "profit_per_unit": 0.40, "win_rate": 0.80,
             "wallet": "0xAAA"},
            {"side": "YES", "correct": True, "value": 2000, "price": 0.55,
             "entry_price": 0.55, "profit_per_unit": 0.45, "win_rate": 0.70,
             "wallet": "0xBBB"},
        ]
        s = _compute_whale_stats(alerts, "YES")
        assert s["count"] == 2
        assert s["correct_count"] == 2
        assert s["incorrect_count"] == 0
        assert s["accuracy"] == pytest.approx(1.0)
        assert s["net_direction"] == "YES"
        assert s["consensus_correct"] is True
        assert s["consensus_strength"] == pytest.approx(1.0)
        assert s["total_value"] == 3000
        assert s["avg_value"] == pytest.approx(1500)
        assert s["max_value"] == 2000
        assert s["avg_win_rate"] == pytest.approx(0.75)
        assert s["unique_wallets"] == 2
        assert s["repeat_actors"] == 0

    def test_mixed_with_repeat(self):
        alerts = [
            {"side": "YES", "correct": True, "value": 500, "wallet": "0xAAA",
             "entry_price": 0.50, "profit_per_unit": 0.50},
            {"side": "NO", "correct": False, "value": 300, "wallet": "0xAAA",
             "entry_price": 0.50, "profit_per_unit": -0.50},
            {"side": "YES", "correct": True, "value": 700, "wallet": "0xBBB",
             "entry_price": 0.60, "profit_per_unit": 0.40},
        ]
        s = _compute_whale_stats(alerts, "YES")
        assert s["count"] == 3
        assert s["correct_count"] == 2
        assert s["incorrect_count"] == 1
        assert s["accuracy"] == pytest.approx(2 / 3)
        assert s["net_direction"] == "YES"
        assert s["consensus_correct"] is True
        assert s["unique_wallets"] == 2
        assert s["repeat_actors"] == 1  # 0xAAA appears twice


# ------------------------------------------------------------------
# Price features
# ------------------------------------------------------------------

class TestComputePriceFeatures:
    def test_empty(self):
        f = _compute_price_features([])
        assert f["count"] == 0
        assert f["mean"] is None

    def test_single_price(self):
        f = _compute_price_features([{"yes_price": 0.50}])
        assert f["count"] == 1
        assert f["mean"] == pytest.approx(0.50)
        assert f["std"] == pytest.approx(0.0)
        assert f["trend"] == pytest.approx(0.0)

    def test_rising_trend(self):
        prices = [
            {"yes_price": 0.40, "volume": 100, "spread": 0.10},
            {"yes_price": 0.50, "volume": 200, "spread": 0.08},
            {"yes_price": 0.60, "volume": 300, "spread": 0.05},
        ]
        f = _compute_price_features(prices)
        assert f["count"] == 3
        assert f["trend"] > 0  # rising
        assert f["min"] == pytest.approx(0.40)
        assert f["max"] == pytest.approx(0.60)
        assert f["volume_mean"] == pytest.approx(200.0)
        assert f["volume_max"] == 300
        assert f["spread_mean"] is not None

    def test_flat_trend(self):
        prices = [
            {"yes_price": 0.50},
            {"yes_price": 0.50},
            {"yes_price": 0.50},
        ]
        f = _compute_price_features(prices)
        assert f["trend"] == pytest.approx(0.0)
        assert f["std"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# wwatcher normalization
# ------------------------------------------------------------------

class TestNormalizeWwatcherAlert:
    def test_outcome_to_side(self):
        raw = {"outcome": "YES", "wallet_id": "0xABC", "created_at": 1740000000}
        norm = _normalize_wwatcher_alert(raw)
        assert norm["side"] == "YES"
        assert norm["wallet"] == "0xABC"
        # created_at should be converted to ISO string
        assert norm["created_at"].endswith("Z")
        assert "T" in norm["created_at"]

    def test_preserves_existing_side(self):
        raw = {"side": "NO", "outcome": "YES", "wallet_id": "0x1"}
        norm = _normalize_wwatcher_alert(raw)
        # side takes precedence over outcome
        assert norm["side"] == "NO"

    def test_no_outcome(self):
        raw = {"value": 1000, "market_id": "abc"}
        norm = _normalize_wwatcher_alert(raw)
        assert "side" not in norm

    def test_iso_timestamp_passthrough(self):
        raw = {"created_at": "2026-01-15T12:00:00Z"}
        norm = _normalize_wwatcher_alert(raw)
        assert norm["created_at"] == "2026-01-15T12:00:00Z"


# ------------------------------------------------------------------
# Timestamp handling
# ------------------------------------------------------------------

class TestTsOrd:
    def test_iso_string(self):
        val = _ts_ord("2026-01-15T12:00:00Z")
        assert val > 0

    def test_unix_int(self):
        val = _ts_ord(1740000000)
        assert val == 1740000000.0

    def test_unix_float(self):
        val = _ts_ord(1740000000.5)
        assert val == 1740000000.5

    def test_bad_string(self):
        val = _ts_ord("not-a-date")
        assert val == 0.0


# ------------------------------------------------------------------
# Resolution conversion
# ------------------------------------------------------------------

class TestResolutionToInt:
    def test_yes(self):
        assert _resolution_to_int("YES") == 1

    def test_no(self):
        assert _resolution_to_int("NO") == 0

    def test_case_insensitive(self):
        assert _resolution_to_int("yes") == 1
        assert _resolution_to_int("No") == 0
