"""CLI entry-point for polymaster-collector."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from collector.config import load_config

log = logging.getLogger("collector")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


@click.group()
@click.option("-c", "--config", "config_path", default=None, help="Path to config.toml")
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """polymaster-collector — prediction market cataloger daemon."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Start the collector daemon (foreground)."""
    from collector.daemon import run_daemon
    from collector.db import init_db

    cfg = ctx.obj["cfg"]

    async def _main() -> None:
        db = await init_db(cfg.general.db_path)
        try:
            await run_daemon(cfg, db)
        finally:
            await db.close()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Shutting down.")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show collection statistics."""
    from collector.db import init_db, stats

    cfg = ctx.obj["cfg"]

    async def _main() -> None:
        db = await init_db(cfg.general.db_path)
        try:
            s = await stats(db)
            click.echo(f"Markets:    active={s['active']}  closed={s['closed']}  resolved={s['resolved']}")
            click.echo(f"Snapshots:  {s['snapshots']}")
            click.echo(f"News:       {s['news']}")
        finally:
            await db.close()

    asyncio.run(_main())


@cli.command()
@click.pass_context
def backfill(ctx: click.Context) -> None:
    """One-time pull of historical resolved markets."""
    from collector.daemon import Collector
    from collector.db import init_db

    cfg = ctx.obj["cfg"]

    async def _main() -> None:
        db = await init_db(cfg.general.db_path)
        try:
            c = Collector(cfg, db)
            await c.start()
            count = await c.run_backfill()
            click.echo(f"Backfilled {count} markets.")
            await c.stop()
        finally:
            await db.close()

    asyncio.run(_main())


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["parquet", "grpo", "sft"]), default="parquet")
@click.option("--category", default=None, help="Filter by category (e.g. 'Climate and Weather')")
@click.option("--platform", default=None, type=click.Choice(["polymarket", "kalshi"]),
              help="Filter by platform")
@click.pass_context
def export(ctx: click.Context, fmt: str, category: str | None, platform: str | None) -> None:
    """Export resolved markets to Parquet, GRPO, or SFT format."""
    from collector.db import init_db
    from collector.export import export_parquet, export_prompts, export_sft

    cfg = ctx.obj["cfg"]

    async def _main() -> None:
        db = await init_db(cfg.general.db_path)
        try:
            if fmt == "parquet":
                path = await export_parquet(db, cfg, category=category, platform=platform)
                click.echo(f"Exported to {path}")
            elif fmt == "grpo":
                path = await export_prompts(db, cfg, category=category, platform=platform)
                click.echo(f"Exported GRPO prompts to {path}")
            else:
                path = await export_sft(db, cfg, category=category, platform=platform)
                click.echo(f"Exported SFT data to {path}")
        finally:
            await db.close()

    asyncio.run(_main())


@cli.command()
@click.pass_context
def categories(ctx: click.Context) -> None:
    """List all categories and their market counts."""
    from collector.db import init_db

    cfg = ctx.obj["cfg"]

    async def _main() -> None:
        db = await init_db(cfg.general.db_path)
        try:
            rows = await db.execute_fetchall(
                "SELECT category, COUNT(*) as total, "
                "SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved "
                "FROM markets WHERE category IS NOT NULL AND category != '' "
                "GROUP BY category ORDER BY total DESC"
            )
            if not rows:
                click.echo("No categories found.")
                return
            click.echo(f"{'Category':<40} {'Total':>8} {'Resolved':>10}")
            click.echo("-" * 60)
            for r in rows:
                click.echo(f"{r['category']:<40} {r['total']:>8} {r['resolved']:>10}")
        finally:
            await db.close()

    asyncio.run(_main())
