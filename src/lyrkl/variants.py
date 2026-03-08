"""Utilities for constructing deterministic lyric control variants."""

from __future__ import annotations

import random


def shuffle_lyrics_lines(lyrics: str, seed: int) -> str:
    """Shuffle lyrics at line level while preserving word order per line.

    Empty lines are preserved in their original positions to maintain stanza
    structure separators.
    """
    lines = lyrics.splitlines()
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if len(non_empty_indices) <= 1:
        return lyrics

    non_empty_lines = [lines[i] for i in non_empty_indices]
    rng = random.Random(seed)
    rng.shuffle(non_empty_lines)

    out = list(lines)
    for idx, line_idx in enumerate(non_empty_indices):
        out[line_idx] = non_empty_lines[idx]
    return "\n".join(out)
