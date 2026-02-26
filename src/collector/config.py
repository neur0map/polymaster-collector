"""Load and validate collector configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

_DEFAULT_CONFIG_PATH = "~/.config/polymaster-collector/config.toml"
_ENV_CONFIG = "COLLECTOR_CONFIG"


@dataclass
class GeneralConfig:
    db_path: str = "~/.config/polymaster-collector/collector.db"
    poll_interval_minutes: int = 5
    snapshot_interval_minutes: int = 15
    resolve_interval_minutes: int = 30
    backfill_interval_hours: int = 24


@dataclass
class PolymarketConfig:
    enabled: bool = True
    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"


@dataclass
class KalshiConfig:
    enabled: bool = True
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class ExportConfig:
    output_dir: str = "./exports"
    wwatcher_db_path: str = "~/.config/wwatcher/wwatcher.db"


@dataclass
class NewsConfig:
    enabled: bool = False
    searxng_url: str = "http://localhost:8080"


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    news: NewsConfig = field(default_factory=NewsConfig)


def _expand(path: str) -> str:
    return str(Path(os.path.expanduser(path)).resolve())


def _section(raw: dict, cls: type, section: str):
    data = raw.get(section, {})
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_config(path: str | None = None) -> Config:
    """Load config from *path*, env var, or default location."""
    if path is None:
        path = os.environ.get(_ENV_CONFIG, _DEFAULT_CONFIG_PATH)
    resolved = Path(os.path.expanduser(path))

    if resolved.exists():
        with open(resolved, "rb") as fh:
            raw = tomllib.load(fh)
    else:
        raw = {}

    cfg = Config(
        general=_section(raw, GeneralConfig, "general"),
        polymarket=_section(raw, PolymarketConfig, "polymarket"),
        kalshi=_section(raw, KalshiConfig, "kalshi"),
        export=_section(raw, ExportConfig, "export"),
        news=_section(raw, NewsConfig, "news"),
    )
    # Expand tilde / env vars in paths.
    cfg.general.db_path = _expand(cfg.general.db_path)
    cfg.export.output_dir = _expand(cfg.export.output_dir)
    cfg.export.wwatcher_db_path = _expand(cfg.export.wwatcher_db_path)
    return cfg
