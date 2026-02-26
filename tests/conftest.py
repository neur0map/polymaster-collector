"""Shared test fixtures."""

from __future__ import annotations

import pytest
import pytest_asyncio

from collector.config import Config, GeneralConfig, KalshiConfig, PolymarketConfig, ExportConfig, NewsConfig
from collector.db import init_db


@pytest_asyncio.fixture
async def db(tmp_path):
    """In-memory-like temp DB for tests."""
    db_path = str(tmp_path / "test_collector.db")
    conn = await init_db(db_path)
    yield conn
    await conn.close()


@pytest.fixture
def cfg(tmp_path):
    """Test config with temp paths."""
    return Config(
        general=GeneralConfig(
            db_path=str(tmp_path / "test.db"),
            poll_interval_minutes=1,
            snapshot_interval_minutes=1,
            resolve_interval_minutes=1,
        ),
        polymarket=PolymarketConfig(enabled=True),
        kalshi=KalshiConfig(enabled=True),
        export=ExportConfig(
            output_dir=str(tmp_path / "exports"),
            wwatcher_db_path=str(tmp_path / "wwatcher.db"),
        ),
        news=NewsConfig(enabled=False),
    )
