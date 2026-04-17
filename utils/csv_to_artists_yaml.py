"""Convert a ``fetch_artists.py`` CSV into a lyrkl-compatible artists YAML.

lyrkl's ``resolve-artists --file`` parser accepts two YAML shapes; this script
can emit either:

1. ``gender`` (default): gender-keyed format, matches ``data/artists.yaml``::

       Female:
         - Taylor Swift
         - Ariana Grande
       Male:
         - Ed Sheeran
         - Drake

   By default the pipeline keeps the **top 20 artists per genre** and ensures
   no artist is cross-listed across genres: each artist is assigned to the
   single genre where their ``genre_rank`` is lowest (ties broken by CSV
   order). Entries become dicts instead of plain strings when
   ``--include-genre`` or ``--max-songs`` is passed.

2. ``flat``: flat list, preserves one entry per surviving (artist, genre) pair::

       artists:
         - name: Taylor Swift
           genre: pop
           artist_gender: Female
           max_songs: 20

Usage:
  python utils/csv_to_artists_yaml.py data/artists_raw_20260416.csv
  python utils/csv_to_artists_yaml.py data/artists_raw_20260416.csv \\
      --format flat --max-songs 20
  python utils/csv_to_artists_yaml.py data/artists_raw_20260416.csv \\
      --genres pop rock --top-per-genre 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

_GENDER_BUCKETS = {
    "female": "Female",
    "male": "Male",
    "nonbinary": "Nonbinary",
    "non-binary": "Nonbinary",
    "other": "Other",
    "": "Other",
}


def _bucket_for_gender(raw: str | None) -> str:
    key = (raw or "").strip().lower()
    return _GENDER_BUCKETS.get(key, "Other")


def _rank(row: dict[str, str]) -> int:
    try:
        return int(row.get("genre_rank") or 10**9)
    except ValueError:
        return 10**9


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dedupe_across_genres(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep each artist only in the genre where they rank highest.

    Identity is the artist ``name`` (case-insensitive, trimmed), falling back
    to ``mbid`` when present. The row with the lowest ``genre_rank`` wins;
    on ties, the earlier CSV occurrence wins. The returned list preserves
    the input ordering of the kept rows.
    """
    best: dict[str, tuple[int, int, dict[str, str]]] = {}
    for idx, r in enumerate(rows):
        name = (r.get("name") or "").strip()
        if not name:
            continue
        key = (r.get("mbid") or "").strip() or name.lower()
        rank = _rank(r)
        prev = best.get(key)
        if prev is None or rank < prev[0]:
            best[key] = (rank, idx, r)

    keep_ids = {id(entry[2]) for entry in best.values()}
    return [r for r in rows if id(r) in keep_ids]


def _filter_rows(
    rows: list[dict[str, str]],
    genres: set[str] | None,
    top_per_genre: int | None,
) -> list[dict[str, str]]:
    if genres:
        rows = [r for r in rows if r.get("genre") in genres]
    if top_per_genre is not None:
        # Rank within each genre after cross-genre dedup so a genre that lost
        # its #1 artist still gets its top 20 surviving artists.
        by_genre: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            by_genre.setdefault(r.get("genre") or "", []).append(r)

        kept: set[int] = set()
        for genre_rows in by_genre.values():
            genre_rows.sort(key=_rank)
            for r in genre_rows[:top_per_genre]:
                kept.add(id(r))
        rows = [r for r in rows if id(r) in kept]
    return rows


def _build_gender_yaml(
    rows: list[dict[str, str]],
    include_genre: bool,
    max_songs: int | None,
) -> dict[str, list[Any]]:
    """Group rows into gender buckets. Assumes rows are already deduped."""
    buckets: dict[str, list[Any]] = {}
    extras_mode = include_genre or max_songs is not None

    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue

        bucket = _bucket_for_gender(r.get("gender"))
        buckets.setdefault(bucket, [])

        if extras_mode:
            entry: dict[str, Any] = {"name": name}
            if include_genre and r.get("genre"):
                entry["genre"] = r["genre"]
            if max_songs is not None:
                entry["max_songs"] = max_songs
            buckets[bucket].append(entry)
        else:
            buckets[bucket].append(name)

    preferred = ["Female", "Male", "Nonbinary", "Other"]
    ordered: dict[str, list[Any]] = {k: buckets[k] for k in preferred if k in buckets}
    for k, v in buckets.items():
        ordered.setdefault(k, v)
    return ordered


def _build_flat_yaml(
    rows: list[dict[str, str]],
    max_songs: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """One entry per surviving CSV row."""
    entries: list[dict[str, Any]] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        entry: dict[str, Any] = {"name": name}
        if r.get("genre"):
            entry["genre"] = r["genre"]
        if r.get("gender"):
            entry["artist_gender"] = _bucket_for_gender(r.get("gender"))
        if max_songs is not None:
            entry["max_songs"] = max_songs
        entries.append(entry)
    return {"artists": entries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="CSV produced by fetch_artists.py")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/artists.yaml"),
        help="Output YAML path (default: data/artists.yaml)",
    )
    ap.add_argument(
        "--format",
        choices=["gender", "flat"],
        default="gender",
        help="YAML shape: gender-keyed (default) or flat 'artists:' list",
    )
    ap.add_argument(
        "--genres",
        nargs="+",
        default=None,
        help="Subset of genres to include (default: all genres in CSV)",
    )
    ap.add_argument(
        "--top-per-genre",
        type=int,
        default=20,
        help="Keep only the top N artists per genre after cross-genre "
        "dedup (default: 20; use 0 to disable)",
    )
    ap.add_argument(
        "--no-dedupe-across-genres",
        dest="dedupe_across_genres",
        action="store_false",
        help="Allow the same artist to appear in multiple genres "
        "(default: artists are assigned to their best-ranked genre only)",
    )
    ap.set_defaults(dedupe_across_genres=True)
    ap.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Add a per-entry max_songs hint (forces dict entries in gender mode)",
    )
    ap.add_argument(
        "--include-genre",
        action="store_true",
        help="(gender mode) Emit dict entries that carry the artist's genre",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    rows = _read_rows(args.csv)
    if not rows:
        print(f"No rows in {args.csv}", file=sys.stderr)
        return 2

    if args.dedupe_across_genres:
        rows = _dedupe_across_genres(rows)

    genres = set(args.genres) if args.genres else None
    top_per_genre = args.top_per_genre if args.top_per_genre and args.top_per_genre > 0 else None
    rows = _filter_rows(rows, genres, top_per_genre)
    if not rows:
        print("No rows left after filtering.", file=sys.stderr)
        return 2

    if args.format == "gender":
        payload = _build_gender_yaml(rows, args.include_genre, args.max_songs)
        summary = ", ".join(f"{k}={len(v)}" for k, v in payload.items())
    else:
        payload = _build_flat_yaml(rows, args.max_songs)
        summary = f"artists={len(payload['artists'])}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"wrote {args.out}  ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
