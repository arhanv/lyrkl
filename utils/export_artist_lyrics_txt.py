"""Export one artist's lyrics to a txt file: copy verbatim from DB or JSON (no cleaning).

Each block is: first line = song name (song_id), rest = lyrics. Blocks separated by
LYRICS_FILE_SONG_SEPARATOR. The acelm-interp steering loader uses the first line as
song_id when saving runs.

Usage:
  # From lyrkl DB (config supplies db path):
  python utils/export_artist_lyrics_txt.py --artist Drake --out drake_lyrics.txt [--config configs/default.yaml]

  # From DB path directly:
  python utils/export_artist_lyrics_txt.py --artist Drake --out drake_lyrics.txt --db data/lyrkl.db

  # From 25x10_songs.json / PromptDataset JSON:
  python utils/export_artist_lyrics_txt.py --artist Drake --out drake_lyrics.txt --json path/to/25x10_songs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root: ensure src/ is on path for lyrkl
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lyrkl.config import load_config
from lyrkl.db import Database

# Explicit separator between songs; must not appear in lyrics. Match in acelm-interp loader.
LYRICS_FILE_SONG_SEPARATOR = "\n<<<LYRKL_SONG_SEP>>>\n"


def from_db(db_path: Path, artist: str) -> list[tuple[str, str]]:
    """(song_id, lyrics) for the given artist, copied as-is from the DB."""
    db = Database(db_path)
    songs = [s for s in db.list_songs() if s.clean_lyrics and s.artist.strip().lower() == artist.strip().lower()]
    db.close()
    return [(s.song_id, s.clean_lyrics) for s in songs]


def from_json(json_path: Path, artist: str) -> list[tuple[str, str]]:
    """(song_id, lyrics) for the given artist from PromptDataset / 25x10-style JSON, copied as-is."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of prompt/song objects.")
    artist_lower = artist.strip().lower()
    out: list[tuple[str, str]] = []
    for entry in data:
        meta = entry.get("metadata") or {}
        a = (meta.get("artist") or entry.get("artist") or "").strip().lower()
        if a != artist_lower:
            continue
        ly = entry.get("lyrics", "")
        if ly:
            sid = entry.get("song_id", meta.get("title", "") or "unknown")
            out.append((str(sid), ly))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one artist's lyrics to a txt (verbatim copy from JSON or DB).")
    ap.add_argument("--artist", required=True, help="Artist name (case-insensitive match).")
    ap.add_argument("--out", required=True, type=Path, help="Output .txt path.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--db", type=Path, help="lyrkl SQLite DB path.")
    group.add_argument("--json", type=Path, help="PromptDataset / 25x10_songs.json path.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config for db path when neither --db nor --json given.")
    args = ap.parse_args()

    if args.json is not None:
        if not args.json.exists():
            ap.error(f"JSON not found: {args.json}")
        songs = from_json(args.json, args.artist)
    elif args.db is not None:
        songs = from_db(args.db, args.artist)
    else:
        cfg = load_config(args.config)
        songs = from_db(cfg.db_path, args.artist)

    if not songs:
        print(f"No songs with lyrics found for artist: {args.artist}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [name + "\n" + lyrics for name, lyrics in songs]
    args.out.write_text(LYRICS_FILE_SONG_SEPARATOR.join(blocks), encoding="utf-8")
    print(f"Wrote {len(songs)} songs to {args.out}")


if __name__ == "__main__":
    main()
