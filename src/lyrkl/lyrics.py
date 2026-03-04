"""Lyrics retrieval from Genius.

Provides a GeniusLyricsFetcher that fetches lyrics by artist/title or
Genius song ID, cleans Genius-specific artifacts, normalizes section
markers, and returns Song objects ready for the lyrkl database.

Requires: pip install lyricsgenius

Usage:
    fetcher = GeniusLyricsFetcher(api_token="your_token")
    song = fetcher.fetch_one("Eminem", "Lose Yourself", genre="hip-hop")
    songs = fetcher.fetch_songs([
        {"artist": "Eminem", "title": "Lose Yourself", "genre": "hip-hop"},
        {"genius_id": 12345, "genre": "pop"},
    ])
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from lyrkl.models import Song

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section marker normalization
# ---------------------------------------------------------------------------

# Genius uses freeform headers like [Verse 1], [Chorus: Drake], etc.
# We normalize these to consistent lowercase markers.
SECTION_NORMALIZE: dict[str, str] = {
    "verse": "verse",
    "chorus": "chorus",
    "pre-chorus": "pre-chorus",
    "prechorus": "pre-chorus",
    "bridge": "bridge",
    "outro": "outro",
    "intro": "intro",
    "hook": "chorus",
    "refrain": "chorus",
    "interlude": "interlude",
    "post-chorus": "post-chorus",
    "postchorus": "post-chorus",
    "breakdown": "breakdown",
    "skit": "skit",
    "spoken": "spoken",
}


def normalize_section_markers(lyrics: str) -> str:
    """Normalize Genius-style section markers to lowercase, clean form.

    Converts headers like "[Verse 1]", "[Chorus: Artist]" to simple
    lowercase markers like "[verse]", "[chorus]". Strips numbered suffixes
    and featured artist annotations.

    Args:
        lyrics: Raw lyrics string with Genius-style section headers.

    Returns:
        Lyrics with normalized section markers.
    """

    def replace_marker(match: re.Match) -> str:
        content = match.group(1).strip()
        content = re.sub(r"\s*\d+\s*$", "", content)
        content = re.sub(r"\s*:.*$", "", content)
        content_lower = content.lower().strip()
        normalized = SECTION_NORMALIZE.get(content_lower, content_lower)
        return f"[{normalized}]"

    return re.sub(r"\[([^\]]+)\]", replace_marker, lyrics)


def clean_genius_lyrics(lyrics: str) -> str:
    """Clean raw Genius lyrics text.

    Removes contributor counts, embed footers, "You might also like" banners,
    and other Genius-specific artifacts. Normalizes section markers.

    Args:
        lyrics: Raw lyrics string from the lyricsgenius library.

    Returns:
        Cleaned lyrics string.
    """
    if not lyrics:
        return ""

    lyrics = re.sub(r"^\d+\s*Contributors?\s*\n?", "", lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r"\d*Embed$", "", lyrics, flags=re.MULTILINE)
    lyrics = re.sub(r"You might also like", "", lyrics)
    lyrics = re.sub(r"See .* LiveGet tickets as low as \$\d+", "", lyrics)

    lyrics = normalize_section_markers(lyrics)

    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    lyrics = lyrics.strip()

    return lyrics


def make_song_id(artist: str, title: str) -> str:
    """Create a URL-safe song slug from artist and title.

    Args:
        artist: Artist name.
        title: Song title.

    Returns:
        A lowercase slug, e.g. "eminem__lose_yourself".
    """
    raw = f"{artist}__{title}"
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class GeniusLyricsFetcher:
    """Fetch and clean lyrics from Genius.

    Args:
        api_token: Genius API access token.
        rate_limit_delay: Seconds to wait between API calls.

    Examples:
        fetcher = GeniusLyricsFetcher(api_token="...")
        song = fetcher.fetch_one("Eminem", "Lose Yourself", genre="hip-hop")
    """

    def __init__(self, api_token: str, rate_limit_delay: float = 1.0) -> None:
        try:
            import lyricsgenius
        except ImportError:
            raise ImportError("lyricsgenius is required: pip install lyricsgenius")

        self._genius = lyricsgenius.Genius(
            api_token,
            verbose=False,
            remove_section_headers=False,
        )
        self._delay = rate_limit_delay

    def fetch_one(
        self,
        artist: str,
        title: str,
        genre: str = "",
        genius_id: Optional[int] = None,
    ) -> Optional[Song]:
        """Fetch lyrics for a single song by artist/title search.

        Args:
            artist: Artist name.
            title: Song title.
            genre: Genre label (not available from Genius; must be supplied).
            genius_id: If provided, used for the Song record but not for lookup.

        Returns:
            A Song with cleaned lyrics, or None if not found or errored.
        """
        try:
            result = self._genius.search_song(title, artist)
        except Exception as e:
            logger.warning("Genius lookup failed for '%s - %s': %s", artist, title, e)
            return None

        if result is None:
            logger.warning("Song not found on Genius: '%s - %s'", artist, title)
            return None

        raw = result.lyrics or ""
        clean = clean_genius_lyrics(raw)
        if not clean:
            logger.warning("Empty lyrics after cleaning: '%s - %s'", artist, title)
            return None

        song_id = make_song_id(artist, title)
        return Song(
            song_id=song_id,
            title=title,
            artist=artist,
            genre=genre,
            raw_lyrics=raw,
            clean_lyrics=clean,
            fetched_at=datetime.utcnow(),
            genius_id=genius_id or getattr(result, "id", None),
        )

    def fetch_by_genius_id(
        self,
        genius_id: int,
        genre: str = "",
    ) -> Optional[Song]:
        """Fetch a song directly by its Genius integer ID.

        Args:
            genius_id: Genius song ID (visible in song URLs).
            genre: Genre label to attach to the Song.

        Returns:
            A Song with cleaned lyrics, or None on failure.
        """
        try:
            result = self._genius.song(genius_id)
        except Exception as e:
            logger.warning("Genius ID lookup failed for id=%d: %s", genius_id, e)
            return None

        if result is None or not hasattr(result, "lyrics"):
            logger.warning("Could not retrieve song for id=%d", genius_id)
            return None

        song_obj = getattr(result, "song", result)
        title = getattr(song_obj, "title", f"genius_{genius_id}")
        artist = getattr(song_obj, "primary_artist", None)
        artist_name = getattr(artist, "name", "Unknown") if artist else "Unknown"

        raw = result.lyrics or ""
        clean = clean_genius_lyrics(raw)
        song_id = make_song_id(artist_name, title)

        return Song(
            song_id=song_id,
            title=title,
            artist=artist_name,
            genre=genre,
            raw_lyrics=raw,
            clean_lyrics=clean,
            fetched_at=datetime.utcnow(),
            genius_id=genius_id,
        )

    def fetch_songs(
        self,
        song_list: list[dict[str, Any]],
    ) -> list[Song]:
        """Fetch lyrics for a list of songs.

        Each entry in song_list is a dict with either:
        - ``title`` and ``artist`` (and optionally ``genre``)
        - ``genius_id`` (and optionally ``genre``)

        Songs that cannot be found are logged and skipped.

        Args:
            song_list: List of song spec dicts.

        Returns:
            List of successfully fetched Song objects.
        """
        results: list[Song] = []
        total = len(song_list)

        for i, spec in enumerate(song_list):
            genre = spec.get("genre", "")
            genius_id: Optional[int] = spec.get("genius_id")

            if genius_id:
                logger.info(
                    "Fetching %d/%d: Genius ID %d", i + 1, total, genius_id
                )
                song = self.fetch_by_genius_id(genius_id, genre=genre)
            else:
                artist = spec.get("artist", "")
                title = spec.get("title", "")
                logger.info(
                    "Fetching %d/%d: %s - %s", i + 1, total, artist, title
                )
                song = self.fetch_one(artist, title, genre=genre)

            if song is not None:
                results.append(song)

            if i < total - 1:
                time.sleep(self._delay)

        logger.info("Fetched %d/%d songs successfully", len(results), total)
        return results
