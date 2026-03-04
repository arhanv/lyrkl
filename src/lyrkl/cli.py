"""Command-line interface for lyrkl.

Usage:
    lyrkl --config configs/default.yaml status
    lyrkl --config configs/default.yaml add-songs songs.yaml
    lyrkl --config configs/default.yaml fetch
    lyrkl --config configs/default.yaml style
    lyrkl --config configs/default.yaml vary
    lyrkl --config configs/default.yaml run-all songs.yaml
    lyrkl --config configs/default.yaml export --out data/prompts.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from lyrkl.config import load_config
from lyrkl.db import Database
from lyrkl.pipeline import LyrkIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.pass_context
def main(ctx: click.Context, config_path: str) -> None:
    """lyrkl -- phonetic lyric variation toolbox."""
    ctx.ensure_object(dict)
    cfg = load_config(config_path)
    db = Database(cfg.db_path)
    ctx.obj["config"] = cfg
    ctx.obj["db"] = db
    ctx.obj["pipeline"] = LyrkIPipeline(cfg, db)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show database summary statistics."""
    db: Database = ctx.obj["db"]
    summary = db.status_summary()
    click.echo(json.dumps(summary, indent=2))


@main.command("add-songs")
@click.argument("songs_file", type=click.Path(exists=True))
@click.pass_context
def add_songs(ctx: click.Context, songs_file: str) -> None:
    """Register songs from a YAML file without fetching lyrics.

    SONGS_FILE should be a YAML file with a top-level "songs" list.
    Each entry is a dict with title/artist/genre or genius_id/genre.
    """
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    with open(songs_file) as f:
        data = yaml.safe_load(f)
    song_list = data.get("songs", [])
    ids = pipeline.add_songs(song_list)
    click.echo(f"Added/updated {len(ids)} songs.")


@main.command()
@click.option(
    "--song", "song_ids", multiple=True, help="Specific song IDs to fetch."
)
@click.pass_context
def fetch(ctx: click.Context, song_ids: tuple[str, ...]) -> None:
    """Fetch lyrics from Genius for songs without lyrics."""
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    fetched = pipeline.fetch_lyrics(list(song_ids) if song_ids else None)
    click.echo(f"Fetched lyrics for {len(fetched)} songs.")


@main.command()
@click.option(
    "--song", "song_ids", multiple=True, help="Specific song IDs to process."
)
@click.option("--overwrite", is_flag=True, default=False, help="Regenerate existing style descriptions.")
@click.pass_context
def style(ctx: click.Context, song_ids: tuple[str, ...], overwrite: bool) -> None:
    """Generate LLM style descriptions for songs."""
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    done = pipeline.generate_style(
        list(song_ids) if song_ids else None, overwrite=overwrite
    )
    click.echo(f"Generated style descriptions for {len(done)} songs.")


@main.command()
@click.option(
    "--song", "song_ids", multiple=True, help="Specific song IDs to process."
)
@click.pass_context
def vary(ctx: click.Context, song_ids: tuple[str, ...]) -> None:
    """Generate phonetic lyric variations for songs."""
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    summary = pipeline.generate_variations(
        list(song_ids) if song_ids else None
    )
    for song_id, counts in summary.items():
        click.echo(
            f"  {song_id}: {counts['accepted']} accepted, {counts['rejected']} rejected"
        )
    click.echo(f"Done. Processed {len(summary)} songs.")


@main.command("run-all")
@click.argument("songs_file", type=click.Path(exists=True))
@click.option(
    "--no-style", is_flag=True, default=False, help="Skip style description generation."
)
@click.pass_context
def run_all(ctx: click.Context, songs_file: str, no_style: bool) -> None:
    """Run all pipeline stages: add, fetch, style, vary.

    SONGS_FILE should be a YAML file with a top-level "songs" list.
    """
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    with open(songs_file) as f:
        data = yaml.safe_load(f)
    song_list = data.get("songs", [])
    pipeline.run_all(song_list, generate_style=not no_style)
    click.echo("Pipeline complete.")


@main.command()
@click.option(
    "--out",
    "out_path",
    default="data/prompts.json",
    show_default=True,
    help="Output JSON path.",
)
@click.option(
    "--song", "song_ids", multiple=True, help="Specific song IDs to export."
)
@click.pass_context
def export(ctx: click.Context, out_path: str, song_ids: tuple[str, ...]) -> None:
    """Export accepted variations as acelm-interp PromptDataset JSON."""
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    n = pipeline.export_prompt_dataset(
        out_path, song_ids=list(song_ids) if song_ids else None
    )
    click.echo(f"Exported {n} records to {out_path}.")


if __name__ == "__main__":
    main()
