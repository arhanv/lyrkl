"""Export a named lyric section (e.g. chorus) from the lyrkl DB as a
verbatim PromptDataset JSON compatible with acelm-interp.

Usage:
    python utils/export_section.py --section chorus --out data/chorus_only_40.json

Options:
    --section   Lyric section to extract (default: chorus). Also matches
                variants like "[Chorus 1]", "[Chorus: Taylor Swift]", and
                "[Hook]" (hook is treated as a chorus alias).
    --out       Output JSON path (default: data/section_export.json).
    --caption   Caption to embed in every record. Pass an empty string (the
                default) to produce caption-free records for encoder probing.
                Use "--caption auto" to fall back to the DB style description.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lyrkl.config import load_config
from lyrkl.db import Database


# Section aliases: "hook" is treated the same as "chorus" in many Genius lyrics.
_SECTION_ALIASES: dict[str, list[str]] = {
    "chorus": ["chorus", "hook", "refrain"],
}


def _build_section_pattern(section: str) -> re.Pattern:
    """Build a regex that matches a lyric section header and its content.

    Handles variants like:
        [Chorus]
        [Chorus 1]
        [Chorus: Taylor Swift]
        [Hook]
    """
    aliases = _SECTION_ALIASES.get(section.lower(), [section.lower()])
    alias_pattern = "|".join(re.escape(a) for a in aliases)
    # Match header like [Chorus], [Chorus 1], [Chorus: anything]
    return re.compile(
        rf"\[(?:{alias_pattern})(?:\s+\d+)?(?::[^\]]+)?\](.*?)(?=\n\[|$)",
        re.IGNORECASE | re.DOTALL,
    )


def extract_section(lyrics: str, section: str) -> str | None:
    """Return the first matching section's text, or None if not found."""
    pattern = _build_section_pattern(section)
    match = pattern.search(lyrics)
    if match:
        return match.group(1).strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", type=str, default="chorus",
                        help="Lyric section name to extract.")
    parser.add_argument("--out", type=str, default="data/section_export.json",
                        help="Output JSON path.")
    parser.add_argument("--caption", type=str, default="",
                        help='Caption for every record. "" = no caption (default). '
                             '"auto" = use DB style description.')
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    db = Database(config.db_path)

    songs = db.list_songs()
    results = []
    skipped = []

    for song in songs:
        if not song.has_lyrics:
            skipped.append(f"{song.song_id} (no lyrics)")
            continue

        section_text = extract_section(song.clean_lyrics, args.section)
        if not section_text:
            skipped.append(f"{song.song_id} (no [{args.section}] found)")
            continue

        if args.caption == "auto":
            caption = db._get_style_text(song.song_id) or f"A {song.genre} song by {song.artist}."
        else:
            caption = args.caption  # "" by default -- caption-free for encoder probing

        results.append({
            "song_id": song.song_id,
            "lyrics": section_text,
            "caption": caption,
            "variant": "verbatim",
            "metadata": {
                "title": song.title,
                "artist": song.artist,
                "genre": song.genre,
                "phi_aggregate": 1.0,
                "phi_scores": {},
                "prompt_hash": "verbatim_baseline",
                "var_id": f"{song.song_id}_verbatim_{args.section}",
                "status": "accepted",
            },
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Exported {len(results)} records containing [{args.section}] to {out_path}.")
    if skipped:
        print(f"Skipped {len(skipped)} songs:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
