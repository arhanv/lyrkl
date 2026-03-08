"""Command-line interface for lyrkl.

Usage:
    lyrkl --config configs/default.yaml status
    lyrkl --config configs/default.yaml add-songs songs.yaml
    lyrkl --config configs/default.yaml fetch
    lyrkl --config configs/default.yaml style
    lyrkl --config configs/default.yaml vary
    lyrkl --config configs/default.yaml run-all songs.yaml

    # Export verbatim full-song lyrics (default):
    lyrkl --config configs/default.yaml export --out data/fullsong_40.json

    # Export verbatim chorus sections only:
    lyrkl --config configs/default.yaml export --section chorus --out data/chorus_only_40.json

    # Export shuffled-line control (deterministic seed):
    lyrkl --config configs/default.yaml export --shuffle-lines --shuffle-seed 1337 --out data/chorus_shuffled_40.json

    # Export accepted phonetic variations (after vary stage):
    lyrkl --config configs/default.yaml export --variations --out data/phonetic.json
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
@click.option(
    "--section",
    default=None,
    help=(
        "Extract a named lyric section (e.g. 'chorus') from each song and export "
        "it as a verbatim record. Also matches common aliases: chorus catches "
        "[Hook] and [Refrain]; omit to export the full clean_lyrics."
    ),
)
@click.option(
    "--variations",
    "use_variations",
    is_flag=True,
    default=False,
    help="Export accepted phonetic variations from the variations table instead of verbatim lyrics.",
)
@click.option(
    "--caption",
    default="",
    show_default=True,
    help=(
        'Caption embedded in every record. Defaults to empty string (caption-free, '
        'recommended for encoder probing). Pass "auto" to use the DB style description.'
    ),
)
@click.option(
    "--shuffle-lines",
    is_flag=True,
    default=False,
    help="Shuffle lyric line order before export (control variant).",
)
@click.option(
    "--shuffle-seed",
    type=int,
    default=0,
    show_default=True,
    help="Base deterministic seed used when --shuffle-lines is enabled.",
)
@click.pass_context
def export(
    ctx: click.Context,
    out_path: str,
    song_ids: tuple[str, ...],
    section: Optional[str],
    use_variations: bool,
    caption: str,
    shuffle_lines: bool,
    shuffle_seed: int,
) -> None:
    """Export lyrics as an acelm-interp PromptDataset JSON.

    By default, exports verbatim lyrics for every song that has lyrics.
    Use --section to extract a named section (e.g. chorus), or --variations
    to export accepted phonetic variations from the vary stage instead.
    """
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]

    if use_variations:
        if shuffle_lines:
            raise click.UsageError(
                "--shuffle-lines applies only to verbatim export. "
                "Use export without --variations."
            )
        n = pipeline.export_prompt_dataset(
            out_path, song_ids=list(song_ids) if song_ids else None
        )
    else:
        n = pipeline.export_verbatim_dataset(
            out_path,
            song_ids=list(song_ids) if song_ids else None,
            section=section,
            caption=caption,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )

    mode = f"[{section}] section" if section else "full song"
    if use_variations:
        src = "variations"
    elif shuffle_lines:
        src = f"shuffled ({mode}, seed={shuffle_seed})"
    else:
        src = f"verbatim ({mode})"
    click.echo(f"Exported {n} records ({src}) to {out_path}.")


if __name__ == "__main__":
    main()
