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
import unicodedata
from dataclasses import dataclass, field
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
# Artist song resolution
# ---------------------------------------------------------------------------

# Patterns that indicate a song is a remix, live version, or alternate edit.
_REMIX_RE = re.compile(
    r"\(.*?\b(remix|rmx|mix|edit|version|live|acoustic|instrumental|demo)\b.*?\)",
    re.IGNORECASE,
)

# Patterns that indicate a song is a cover or tribute.
_COVER_RE = re.compile(r"\(.*?\b(cover|tribute)\b.*?\)", re.IGNORECASE)

# Separators that indicate a joint artist credit (collaborative release).
_COLLAB_RE = re.compile(r"\s(&|×|x)\s", re.IGNORECASE)


def _base_title(title: str) -> str:
    """Strip parenthetical suffixes to get the canonical title.

    For example:
        "Lose Yourself (Remix)"          -> "Lose Yourself"
        "God's Plan (feat. Drake)"       -> "God's Plan"
        "Blinding Lights (Live Version)" -> "Blinding Lights"

    Args:
        title: Raw song title, possibly with parenthetical annotations.

    Returns:
        Title with all parenthetical blocks removed and whitespace stripped.
    """
    return re.sub(r"\s*\(.*?\)", "", title).strip()


def _is_remix(title: str) -> bool:
    """Return True if the title contains a remix/live/edit parenthetical."""
    return bool(_REMIX_RE.search(title))


def _is_cover(title: str) -> bool:
    """Return True if the title contains a cover/tribute parenthetical."""
    return bool(_COVER_RE.search(title))


def _is_collaboration(artist_name: str) -> bool:
    """Return True if the artist credit appears to be a joint release.

    Checks for common separators like ' & ', ' x ', ' × ' in the artist name.

    Args:
        artist_name: The primary artist string from the Genius API.

    Returns:
        True if the name contains a collaboration separator.
    """
    return bool(_COLLAB_RE.search(artist_name))


def _normalize_artist(name: str) -> str:
    """Normalize an artist name for loose equality comparison.

    Lowercases, strips leading/trailing whitespace, and removes diacritics
    via NFKD decomposition so that e.g. "Beyonce" matches "Beyonce". Falls
    back to plain lowercase when the name contains no ASCII characters (e.g.
    CJK or Arabic scripts) to avoid producing an empty comparison string.

    Args:
        name: Raw artist name string.

    Returns:
        Normalized string suitable for equality comparison.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()
    return ascii_only or name.lower().strip()


@dataclass
class ArtistSongFilter:
    """Filtering options for artist song resolution via Genius.

    Attributes:
        max_songs: Maximum number of songs to return after filtering.
        sort: Sort order passed to Genius. Either "popularity" or "title".
        include_featured_vocals: If False, exclude songs that list featured
            artists (i.e. songs with multiple vocalists).
        include_collaborations: If False, exclude songs whose primary artist
            credit is a joint release (e.g. "Artist A & Artist B").
        exclude_remixes: If True, exclude remixes, live versions, and alternate
            edits — but only when the apparent original is also in the results.
        exclude_covers: If True, exclude songs with "(Cover)" or "(Tribute)" in
            their title.
        max_year: If set, exclude songs released strictly after this year.
            Songs with no release date are always included.
    """

    max_songs: int = 20
    sort: str = "popularity"
    include_featured_vocals: bool = True
    include_collaborations: bool = True
    exclude_remixes: bool = True
    exclude_covers: bool = True
    max_year: Optional[int] = None


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

    def __init__(
        self,
        api_token: str,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        try:
            import lyricsgenius
        except ImportError:
            raise ImportError("lyricsgenius is required: pip install lyricsgenius")

        self._genius = lyricsgenius.Genius(
            api_token,
            verbose=False,
            remove_section_headers=False,
            timeout=timeout,
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
        # lyricsgenius 3.x: Genius.song(id) returns API JSON (dict), not an object
        # with .lyrics. search_song(song_id=...) performs the metadata lookup and
        # HTML scrape the same way as title/artist search.
        try:
            result = self._genius.search_song(song_id=genius_id)
        except Exception as e:
            logger.warning("Genius ID lookup failed for id=%d: %s", genius_id, e)
            return None

        if result is None:
            logger.warning("Could not retrieve song for id=%d", genius_id)
            return None

        title = result.title
        artist_name = result.artist
        raw = result.lyrics or ""
        clean = clean_genius_lyrics(raw)
        if not clean:
            logger.warning("Empty lyrics after cleaning for id=%d", genius_id)
            return None

        resolved_id = result.to_dict().get("id", genius_id)
        song_id = make_song_id(artist_name, title)

        return Song(
            song_id=song_id,
            title=title,
            artist=artist_name,
            genre=genre,
            raw_lyrics=raw,
            clean_lyrics=clean,
            fetched_at=datetime.utcnow(),
            genius_id=int(resolved_id) if resolved_id is not None else genius_id,
        )

    def _find_artist_id(self, artist: str) -> Optional[int]:
        """Find a Genius artist ID by searching for the artist name.

        Searches the Genius /search endpoint (which returns song hits) and
        extracts the primary_artist.id from the first result whose artist name
        matches. Falls back to the first result's primary artist if no exact
        match is found.

        Args:
            artist: Artist name to look up.

        Returns:
            Genius integer artist ID, or None if not found.
        """
        try:
            results = self._genius.search(artist)
        except Exception as e:
            logger.warning("_find_artist_id: search failed for '%s': %s", artist, e)
            return None

        hits = (results or {}).get("hits", [])
        artist_lower = artist.lower()

        # Prefer an exact name match.
        for hit in hits:
            if hit.get("type") != "song":
                continue
            primary = hit.get("result", {}).get("primary_artist", {})
            if primary.get("name", "").lower() == artist_lower:
                return primary.get("id")

        # Fall back to the first song hit's primary artist.
        for hit in hits:
            if hit.get("type") == "song":
                primary = hit.get("result", {}).get("primary_artist", {})
                aid = primary.get("id")
                if aid:
                    logger.debug(
                        "_find_artist_id: no exact match for '%s'; using '%s' (id=%d)",
                        artist, primary.get("name", ""), aid,
                    )
                    return aid

        logger.warning("_find_artist_id: no results found for '%s'.", artist)
        return None

    def resolve_artist_songs(
        self,
        artist: str,
        filter: ArtistSongFilter = None,
        genre: str = "",
    ) -> list[dict[str, Any]]:
        """Resolve an artist's top songs from Genius and return song spec dicts.

        Uses the Genius /artists/{id}/songs REST endpoint directly, which
        returns pure metadata without scraping any lyrics HTML pages. This is
        significantly faster than search_artist() and avoids timeout issues
        caused by per-song page loads.

        Pass 1 applies: featured vocals, collaboration, year, and cover filters.
        Pass 2 applies smart remix exclusion: a remix is dropped only if a song
        with the same base title is also present in the candidate set.

        Args:
            artist: Artist name to search for on Genius.
            filter: ArtistSongFilter controlling what is included/excluded.
                Defaults to ArtistSongFilter() with all defaults.
            genre: Genre label to attach to every returned song spec. Genius
                does not provide genre data; this must be supplied by the caller.

        Returns:
            List of song spec dicts with keys: title, artist, genius_id, genre.
            May be shorter than filter.max_songs if not enough songs pass.
        """
        if filter is None:
            filter = ArtistSongFilter()

        fetch_limit = min(filter.max_songs * 4, 100)
        logger.info(
            "resolve_artist_songs: resolving '%s' (want=%d, sort=%s)",
            artist, filter.max_songs, filter.sort,
        )

        artist_id = self._find_artist_id(artist)
        if artist_id is None:
            logger.warning("resolve_artist_songs: artist '%s' not found on Genius.", artist)
            return []

        # Paginate through /artists/{id}/songs — pure API, no HTML scraping.
        raw_songs: list[dict[str, Any]] = []
        page = 1
        per_page = 20
        while len(raw_songs) < fetch_limit:
            try:
                result = self._genius.artist_songs(
                    artist_id, sort=filter.sort, per_page=per_page, page=page
                )
            except Exception as e:
                logger.warning(
                    "resolve_artist_songs: artist_songs API failed for '%s' (page %d): %s",
                    artist, page, e,
                )
                break

            page_songs = (result or {}).get("songs", [])
            if not page_songs:
                break
            raw_songs.extend(page_songs)

            if (result or {}).get("next_page") is None:
                break
            page += 1

        # Pass 1: primary artist check, then featured vocals, collaboration,
        # year, and cover filters.
        candidates: list[dict[str, Any]] = []
        for song in raw_songs:
            title: str = song.get("title", "") or ""
            artist_name: str = (song.get("primary_artist") or {}).get("name", artist) or artist
            featured: list = song.get("featured_artists") or []
            release: Optional[dict] = song.get("release_date_components")
            genius_id: Optional[int] = song.get("id")

            if _normalize_artist(artist_name) != _normalize_artist(artist):
                logger.debug(
                    "resolve_artist_songs: skipping '%s' (primary artist is '%s', not '%s')",
                    title, artist_name, artist,
                )
                continue

            if not filter.include_featured_vocals and featured:
                logger.debug("resolve_artist_songs: skipping '%s' (has featured artists)", title)
                continue

            if not filter.include_collaborations and _is_collaboration(artist_name):
                logger.debug("resolve_artist_songs: skipping '%s' (collaboration)", title)
                continue

            if filter.max_year is not None and release is not None:
                year = release.get("year")
                if year is not None and year > filter.max_year:
                    logger.debug(
                        "resolve_artist_songs: skipping '%s' (year %d > %d)",
                        title, year, filter.max_year,
                    )
                    continue

            if filter.exclude_covers and _is_cover(title):
                logger.debug("resolve_artist_songs: skipping '%s' (cover)", title)
                continue

            candidates.append({"title": title, "artist": artist, "genius_id": genius_id, "genre": genre})

        # Pass 2: smart remix exclusion.
        if filter.exclude_remixes:
            base_titles = {_base_title(c["title"]) for c in candidates}
            filtered: list[dict[str, Any]] = []
            for c in candidates:
                if _is_remix(c["title"]):
                    base = _base_title(c["title"])
                    if base in base_titles and base != c["title"]:
                        logger.debug(
                            "resolve_artist_songs: skipping remix '%s' (original present)",
                            c["title"],
                        )
                        continue
                filtered.append(c)
            candidates = filtered

        result_songs = candidates[: filter.max_songs]

        if len(result_songs) < filter.max_songs:
            logger.warning(
                "resolve_artist_songs: only %d/%d songs passed filters for '%s'.",
                len(result_songs), filter.max_songs, artist,
            )

        logger.info(
            "resolve_artist_songs: resolved %d songs for '%s'.", len(result_songs), artist
        )
        return result_songs

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
