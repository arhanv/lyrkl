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
from lyrkl.lyrics import ArtistSongFilter
from lyrkl.pipeline import LyrkIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

# Known gender keys that trigger the gender-keyed YAML format.
_GENDER_KEYS = {"female", "male", "nonbinary", "non-binary", "other"}


def _parse_artists_yaml(path: str, default_genre: str = "") -> list[dict]:
    """Parse an artists YAML file in either of two formats.

    Gender-keyed format::

        Female:
          - Taylor Swift
          - name: Ariana Grande
            genre: pop

        Male:
          - Ed Sheeran

    Flat format (existing)::

        artists:
          - name: Eminem
            genre: hip-hop

    In the gender-keyed format the key (e.g. "Female") is stored as
    ``artist_gender`` on each entry. Plain strings and dicts are both
    accepted as list items. Per-artist keys override the top-level defaults.

    Args:
        path: Path to the YAML file.
        default_genre: Fallback genre when none is specified per-artist.

    Returns:
        List of artist spec dicts with at least a ``name`` key.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    top_keys_lower = {k.lower() for k in data}

    if top_keys_lower & _GENDER_KEYS:
        # Gender-keyed format.
        artist_list: list[dict] = []
        for key, entries in data.items():
            gender = key  # preserve original capitalisation, e.g. "Female"
            for entry in (entries or []):
                if isinstance(entry, str):
                    spec: dict = {"name": entry}
                else:
                    spec = dict(entry)
                spec.setdefault("genre", default_genre)
                spec["artist_gender"] = gender
                artist_list.append(spec)
        return artist_list

    # Flat format.
    return data.get("artists", [])


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


@main.command("resolve-artists")
@click.argument("artists", nargs=-1)
@click.option(
    "--file", "artists_file",
    type=click.Path(exists=True),
    default=None,
    help=(
        "YAML file with an 'artists' list. Each entry must have 'name' and may "
        "include genre, max_songs, max_year, sort, include_featured_vocals, "
        "include_collaborations, exclude_remixes, exclude_covers."
    ),
)
@click.option("--genre", default="", show_default=True, help="Genre label for all resolved songs.")
@click.option("--max-songs", type=int, default=20, show_default=True, help="Songs to resolve per artist.")
@click.option(
    "--sort",
    type=click.Choice(["popularity", "title"]),
    default="popularity",
    show_default=True,
    help="Sort order for Genius results.",
)
@click.option("--no-featured-vocals", is_flag=True, default=False, help="Exclude songs with featured vocalists.")
@click.option("--no-collaborations", is_flag=True, default=False, help="Exclude joint-credit releases.")
@click.option("--include-remixes", is_flag=True, default=False, help="Include remixes even when original is present.")
@click.option("--include-covers", is_flag=True, default=False, help="Include cover versions.")
@click.option("--before-year", type=int, default=None, help="Exclude songs released after this year.")
@click.pass_context
def resolve_artists(
    ctx: click.Context,
    artists: tuple[str, ...],
    artists_file: Optional[str],
    genre: str,
    max_songs: int,
    sort: str,
    no_featured_vocals: bool,
    no_collaborations: bool,
    include_remixes: bool,
    include_covers: bool,
    before_year: Optional[int],
) -> None:
    """Resolve top songs for one or more artists via Genius and register them.

    Pass artist names as arguments, or use --file to read from a YAML file.
    The two sources are mutually exclusive. After resolving, run 'lyrkl fetch'
    to download lyrics.

    \b
    Examples:
      lyrkl resolve-artists "Eminem" "Kendrick Lamar" --genre hip-hop
      lyrkl resolve-artists --file artists.yaml
    """
    if artists and artists_file:
        raise click.UsageError("Provide either artist names or --file, not both.")
    if not artists and not artists_file:
        raise click.UsageError("Provide at least one artist name or --file.")

    pipeline: LyrkIPipeline = ctx.obj["pipeline"]

    default_filter = ArtistSongFilter(
        max_songs=max_songs,
        sort=sort,
        include_featured_vocals=not no_featured_vocals,
        include_collaborations=not no_collaborations,
        exclude_remixes=not include_remixes,
        exclude_covers=not include_covers,
        max_year=before_year,
    )

    if artists_file:
        artist_list = _parse_artists_yaml(artists_file, default_genre=genre)
    else:
        artist_list = [{"name": a, "genre": genre} for a in artists]

    ids = pipeline.resolve_artists(artist_list, default_filter=default_filter)
    click.echo(f"Resolved and registered {len(ids)} songs. Run 'lyrkl fetch' to download lyrics.")


@main.command("check-duplicates")
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Minimum Jaccard similarity to flag a pair.",
)
@click.option(
    "--song", "song_ids", multiple=True, help="Specific song IDs to check (defaults to all)."
)
@click.pass_context
def check_duplicates(
    ctx: click.Context,
    threshold: float,
    song_ids: tuple[str, ...],
) -> None:
    """Find songs with suspiciously similar lyrics (post-fetch sanity check).

    Compares word-bag Jaccard similarity between all songs with lyrics.
    Pairs at or above --threshold are printed as potential duplicates.

    Run this after 'lyrkl fetch' to catch accidentally repeated songs.
    """
    pipeline: LyrkIPipeline = ctx.obj["pipeline"]
    pairs = pipeline.check_duplicates(
        song_ids=list(song_ids) if song_ids else None,
        threshold=threshold,
    )
    if pairs:
        click.echo(f"Found {len(pairs)} potential duplicate pair(s) (threshold={threshold}):")
        for a, b, score in pairs:
            click.echo(f"  {a}  <->  {b}  (jaccard={score:.3f})")
    else:
        click.echo(f"No duplicates found (threshold={threshold}).")


@main.command("remove-songs")
@click.argument("song_ids", nargs=-1)
@click.pass_context
def remove_songs(ctx: click.Context, song_ids: tuple[str, ...]) -> None:
    """Delete one or more songs and all their dependent data from the database.

    SONG_IDS are the slug identifiers shown by 'lyrkl status' (e.g.
    eminem__lose_yourself). Each song's variations, style descriptions,
    and LLM response records are removed along with the song row itself.

    \b
    Examples:
      lyrkl remove-songs katy_perry_earth ariana_grande_earth
    """
    if not song_ids:
        raise click.UsageError("Provide at least one song ID to remove.")

    db: Database = ctx.obj["db"]
    removed = 0
    for sid in song_ids:
        if db.delete_song(sid):
            click.echo(f"Removed: {sid}")
            removed += 1
        else:
            click.echo(f"Not found: {sid}")
    click.echo(f"Done. Removed {removed}/{len(song_ids)} songs.")


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
