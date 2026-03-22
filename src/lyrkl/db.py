"""SQLite persistence layer for lyrkl.

All database operations go through this module. The schema is created
automatically on first use via `Database.initialize()`. All writes use
INSERT OR IGNORE or INSERT OR REPLACE to guarantee idempotency.

Usage:
    db = Database("data/lyrkl.db")
    db.upsert_song(song)
    variations = db.get_accepted_variations(song_id="eminem__lose_yourself")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from lyrkl.models import (
    LLMResponse,
    PhiScores,
    Song,
    StyleDescription,
    StyleSource,
    Variation,
    VariantStatus,
    VariantType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS songs (
    song_id       TEXT PRIMARY KEY,
    genius_id     INTEGER,
    title         TEXT NOT NULL,
    artist        TEXT NOT NULL,
    genre         TEXT NOT NULL DEFAULT '',
    raw_lyrics    TEXT NOT NULL DEFAULT '',
    clean_lyrics  TEXT NOT NULL DEFAULT '',
    fetched_at    TEXT,
    audio_path    TEXT,
    artist_gender TEXT
);

CREATE TABLE IF NOT EXISTS style_descriptions (
    desc_id      TEXT PRIMARY KEY,
    song_id      TEXT NOT NULL REFERENCES songs(song_id),
    text         TEXT NOT NULL,
    source       TEXT NOT NULL,
    model        TEXT NOT NULL DEFAULT '',
    prompt_hash  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_responses (
    response_id    TEXT PRIMARY KEY,
    song_id        TEXT NOT NULL REFERENCES songs(song_id),
    purpose        TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_hash    TEXT NOT NULL,
    prompt_text    TEXT NOT NULL,
    response_text  TEXT NOT NULL,
    n_candidates   INTEGER NOT NULL DEFAULT 0,
    raw_file_path  TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS variations (
    var_id           TEXT PRIMARY KEY,
    song_id          TEXT NOT NULL REFERENCES songs(song_id),
    variant_type     TEXT NOT NULL,
    lyrics           TEXT NOT NULL,
    phi_scores_json  TEXT NOT NULL,
    phi_aggregate    REAL NOT NULL DEFAULT 0.0,
    prompt_hash      TEXT NOT NULL,
    llm_response_id  TEXT REFERENCES llm_responses(response_id),
    status           TEXT NOT NULL DEFAULT 'pending',
    created_at       TEXT NOT NULL,
    candidate_index  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_variations_song_id
    ON variations(song_id);
CREATE INDEX IF NOT EXISTS idx_variations_status
    ON variations(status);
CREATE INDEX IF NOT EXISTS idx_variations_type
    ON variations(variant_type);
CREATE INDEX IF NOT EXISTS idx_variations_prompt_hash
    ON variations(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_style_song
    ON style_descriptions(song_id, source, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_song_purpose
    ON llm_responses(song_id, purpose);
"""


class Database:
    """SQLite-backed persistence for songs, variations, and LLM responses.

    All public methods are safe to call multiple times; duplicate inserts
    are silently ignored unless explicitly stated otherwise.

    Args:
        path: Path to the SQLite database file. Created if it does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()
        logger.debug("Database opened: %s", self._path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _initialize(self) -> None:
        """Create tables and indexes if they do not already exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        # Migration: add artist_gender column to existing databases that
        # were created before this field was introduced.
        try:
            self._conn.execute("ALTER TABLE songs ADD COLUMN artist_gender TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists.

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a single transaction."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Songs
    # ------------------------------------------------------------------

    def upsert_song(self, song: Song) -> None:
        """Insert or update a song record.

        If the song already exists, any non-null fields in `song` overwrite
        the stored values (allowing incremental enrichment, e.g. adding lyrics
        after an initial add_songs call).

        Args:
            song: The Song to upsert.
        """
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO songs
                    (song_id, genius_id, title, artist, genre,
                     raw_lyrics, clean_lyrics, fetched_at, audio_path,
                     artist_gender)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    genius_id     = COALESCE(excluded.genius_id, genius_id),
                    title         = excluded.title,
                    artist        = excluded.artist,
                    genre         = excluded.genre,
                    raw_lyrics    = CASE WHEN excluded.raw_lyrics != ''
                                         THEN excluded.raw_lyrics
                                         ELSE raw_lyrics END,
                    clean_lyrics  = CASE WHEN excluded.clean_lyrics != ''
                                         THEN excluded.clean_lyrics
                                         ELSE clean_lyrics END,
                    fetched_at    = COALESCE(excluded.fetched_at, fetched_at),
                    audio_path    = COALESCE(excluded.audio_path, audio_path),
                    artist_gender = COALESCE(excluded.artist_gender, artist_gender)
                """,
                (
                    song.song_id,
                    song.genius_id,
                    song.title,
                    song.artist,
                    song.genre,
                    song.raw_lyrics,
                    song.clean_lyrics,
                    song.fetched_at.isoformat() if song.fetched_at else None,
                    song.audio_path,
                    song.artist_gender,
                ),
            )

    def get_song(self, song_id: str) -> Optional[Song]:
        """Retrieve a song by its slug ID, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM songs WHERE song_id = ?", (song_id,)
        ).fetchone()
        return _row_to_song(row) if row else None

    def list_songs(self) -> list[Song]:
        """Return all songs in the database."""
        rows = self._conn.execute("SELECT * FROM songs ORDER BY artist, title").fetchall()
        return [_row_to_song(r) for r in rows]

    def songs_without_lyrics(self) -> list[Song]:
        """Songs that have been added but have no clean lyrics yet."""
        rows = self._conn.execute(
            "SELECT * FROM songs WHERE clean_lyrics = '' OR clean_lyrics IS NULL"
        ).fetchall()
        return [_row_to_song(r) for r in rows]

    def songs_without_variations(self, prompt_hash: str) -> list[Song]:
        """Songs that have lyrics but no accepted variations for prompt_hash."""
        rows = self._conn.execute(
            """
            SELECT s.* FROM songs s
            WHERE s.clean_lyrics != ''
              AND NOT EXISTS (
                  SELECT 1 FROM variations v
                  WHERE v.song_id = s.song_id
                    AND v.prompt_hash = ?
              )
            """,
            (prompt_hash,),
        ).fetchall()
        return [_row_to_song(r) for r in rows]

    def songs_without_style(self, source: StyleSource, prompt_hash: str = "") -> list[Song]:
        """Songs that have lyrics but no style description from `source`.

        Args:
            source: The style source to check (e.g. StyleSource.LLM).
            prompt_hash: If non-empty, also require a specific prompt hash.
        """
        if prompt_hash:
            rows = self._conn.execute(
                """
                SELECT s.* FROM songs s
                WHERE s.clean_lyrics != ''
                  AND NOT EXISTS (
                      SELECT 1 FROM style_descriptions sd
                      WHERE sd.song_id = s.song_id
                        AND sd.source = ?
                        AND sd.prompt_hash = ?
                  )
                """,
                (source.value, prompt_hash),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT s.* FROM songs s
                WHERE s.clean_lyrics != ''
                  AND NOT EXISTS (
                      SELECT 1 FROM style_descriptions sd
                      WHERE sd.song_id = s.song_id
                        AND sd.source = ?
                  )
                """,
                (source.value,),
            ).fetchall()
        return [_row_to_song(r) for r in rows]

    def set_audio_path(self, song_id: str, audio_path: str) -> None:
        """Update the audio_path for a song."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE songs SET audio_path = ? WHERE song_id = ?",
                (audio_path, song_id),
            )

    def delete_song(self, song_id: str) -> bool:
        """Delete a song and all its dependent rows from the database.

        Removes rows in foreign-key dependency order so that the
        ``PRAGMA foreign_keys=ON`` constraint is satisfied:
        variations -> style_descriptions -> llm_responses -> songs.
        All deletes are executed inside a single transaction.

        Args:
            song_id: The song slug to delete (e.g. "eminem__lose_yourself").

        Returns:
            True if the song existed and was deleted; False if not found.
        """
        if self.get_song(song_id) is None:
            return False
        with self._tx() as conn:
            conn.execute("DELETE FROM variations WHERE song_id = ?", (song_id,))
            conn.execute("DELETE FROM style_descriptions WHERE song_id = ?", (song_id,))
            conn.execute("DELETE FROM llm_responses WHERE song_id = ?", (song_id,))
            conn.execute("DELETE FROM songs WHERE song_id = ?", (song_id,))
        logger.info("delete_song: deleted '%s' and all dependent rows.", song_id)
        return True

    # ------------------------------------------------------------------
    # Style descriptions
    # ------------------------------------------------------------------

    def save_style_description(self, desc: StyleDescription) -> None:
        """Insert a style description. Does nothing if desc_id already exists."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO style_descriptions
                    (desc_id, song_id, text, source, model, prompt_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    desc.desc_id,
                    desc.song_id,
                    desc.text,
                    desc.source.value,
                    desc.model,
                    desc.prompt_hash,
                    desc.created_at.isoformat(),
                ),
            )

    def get_latest_style(
        self, song_id: str, source: StyleSource = StyleSource.LLM
    ) -> Optional[StyleDescription]:
        """Return the most recent style description for a song from a given source."""
        row = self._conn.execute(
            """
            SELECT * FROM style_descriptions
            WHERE song_id = ? AND source = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (song_id, source.value),
        ).fetchone()
        return _row_to_style(row) if row else None

    def list_styles(self, song_id: str) -> list[StyleDescription]:
        """Return all style descriptions for a song, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM style_descriptions WHERE song_id = ? ORDER BY created_at DESC",
            (song_id,),
        ).fetchall()
        return [_row_to_style(r) for r in rows]

    # ------------------------------------------------------------------
    # LLM responses
    # ------------------------------------------------------------------

    def save_llm_response(self, response: LLMResponse) -> None:
        """Insert an LLM response record. Ignored if response_id exists."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO llm_responses
                    (response_id, song_id, purpose, model, prompt_hash,
                     prompt_text, response_text, n_candidates, raw_file_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.response_id,
                    response.song_id,
                    response.purpose,
                    response.model,
                    response.prompt_hash,
                    response.prompt_text,
                    response.response_text,
                    response.n_candidates,
                    response.raw_file_path,
                    response.created_at.isoformat(),
                ),
            )

    def get_llm_response(self, response_id: str) -> Optional[LLMResponse]:
        """Retrieve a single LLM response by ID."""
        row = self._conn.execute(
            "SELECT * FROM llm_responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        return _row_to_llm_response(row) if row else None

    def list_llm_responses(self, song_id: str, purpose: str = "") -> list[LLMResponse]:
        """List LLM responses for a song, optionally filtered by purpose."""
        if purpose:
            rows = self._conn.execute(
                "SELECT * FROM llm_responses WHERE song_id = ? AND purpose = ? ORDER BY created_at",
                (song_id, purpose),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM llm_responses WHERE song_id = ? ORDER BY created_at",
                (song_id,),
            ).fetchall()
        return [_row_to_llm_response(r) for r in rows]

    def response_exists(self, song_id: str, prompt_hash: str, purpose: str) -> bool:
        """True if an LLM response with this song+prompt+purpose already exists."""
        row = self._conn.execute(
            """
            SELECT 1 FROM llm_responses
            WHERE song_id = ? AND prompt_hash = ? AND purpose = ?
            LIMIT 1
            """,
            (song_id, prompt_hash, purpose),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Variations
    # ------------------------------------------------------------------

    def save_variation(self, variation: Variation) -> None:
        """Insert a variation. Ignored if var_id already exists."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO variations
                    (var_id, song_id, variant_type, lyrics, phi_scores_json,
                     phi_aggregate, prompt_hash, llm_response_id, status,
                     created_at, candidate_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    variation.var_id,
                    variation.song_id,
                    variation.variant_type.value,
                    variation.lyrics,
                    variation.phi_scores.to_json(),
                    variation.phi_aggregate,
                    variation.prompt_hash,
                    variation.llm_response_id,
                    variation.status.value,
                    variation.created_at.isoformat(),
                    variation.candidate_index,
                ),
            )

    def save_variations(self, variations: list[Variation]) -> None:
        """Bulk-insert variations; duplicates are silently skipped."""
        for v in variations:
            self.save_variation(v)

    def get_variations(
        self,
        song_id: str,
        variant_type: Optional[VariantType] = None,
        status: Optional[VariantStatus] = None,
        prompt_hash: Optional[str] = None,
    ) -> list[Variation]:
        """Query variations with optional filters.

        Args:
            song_id: Required song slug.
            variant_type: Optional type filter.
            status: Optional status filter (accepted / rejected / pending).
            prompt_hash: Optional prompt hash filter.

        Returns:
            List of matching Variation objects, ordered by candidate_index.
        """
        clauses = ["song_id = ?"]
        params: list[str] = [song_id]

        if variant_type is not None:
            clauses.append("variant_type = ?")
            params.append(variant_type.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if prompt_hash is not None:
            clauses.append("prompt_hash = ?")
            params.append(prompt_hash)

        sql = (
            f"SELECT * FROM variations WHERE {' AND '.join(clauses)}"
            " ORDER BY candidate_index"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_variation(r) for r in rows]

    def get_accepted_variations(
        self,
        song_id: str,
        prompt_hash: Optional[str] = None,
    ) -> list[Variation]:
        """Convenience: accepted phonetic variations for a song."""
        return self.get_variations(
            song_id,
            variant_type=VariantType.PHONETIC,
            status=VariantStatus.ACCEPTED,
            prompt_hash=prompt_hash,
        )

    def variation_exists(self, var_id: str) -> bool:
        """True if a variation with this ID already exists."""
        return (
            self._conn.execute(
                "SELECT 1 FROM variations WHERE var_id = ? LIMIT 1", (var_id,)
            ).fetchone()
            is not None
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_prompt_dataset(
        self,
        song_ids: Optional[list[str]] = None,
        variant_types: Optional[list[VariantType]] = None,
        statuses: Optional[list[VariantStatus]] = None,
    ) -> list[dict]:
        """Export variations as a list of dicts compatible with acelm-interp
        PromptDataset JSON format.

        Args:
            song_ids: If provided, only include these songs.
            variant_types: If provided, only include these variant types.
            statuses: If provided, only include these statuses. Defaults to
                ACCEPTED only.

        Returns:
            List of dicts, one per variation, ready to pass to
            ``json.dumps`` or ``PromptDataset.from_dict``.
        """
        statuses = statuses or [VariantStatus.ACCEPTED]
        all_songs = song_ids or [s.song_id for s in self.list_songs()]
        variant_types = variant_types or list(VariantType)

        results = []
        for sid in all_songs:
            song = self.get_song(sid)
            if song is None:
                continue
            for vtype in variant_types:
                for vstatus in statuses:
                    for var in self.get_variations(sid, variant_type=vtype, status=vstatus):
                        results.append(
                            {
                                "song_id": var.song_id,
                                "lyrics": var.lyrics,
                                "caption": self._get_style_text(sid),
                                "variant": var.variant_type.value,
                                "metadata": {
                                    "title": song.title,
                                    "artist": song.artist,
                                    "genre": song.genre,
                                    "artist_gender": song.artist_gender,
                                    "phi_aggregate": var.phi_aggregate,
                                    "phi_scores": var.phi_scores.to_dict(),
                                    "prompt_hash": var.prompt_hash,
                                    "var_id": var.var_id,
                                    "status": var.status.value,
                                },
                            }
                        )
        return results

    def _get_style_text(self, song_id: str) -> str:
        """Return the best available style description text for a song."""
        style = self.get_latest_style(song_id, StyleSource.LLM)
        if style:
            return style.text
        style = self.get_latest_style(song_id, StyleSource.CAPTIONER)
        if style:
            return style.text
        style = self.get_latest_style(song_id, StyleSource.MANUAL)
        if style:
            return style.text
        return ""

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

    def status_summary(self) -> dict:
        """Return a summary dict suitable for display in notebooks or CLI.

        Returns:
            Dict with counts of songs, variations by type/status, etc.
        """
        n_songs = self._conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        n_lyrics = self._conn.execute(
            "SELECT COUNT(*) FROM songs WHERE clean_lyrics != ''"
        ).fetchone()[0]
        n_styles = self._conn.execute("SELECT COUNT(*) FROM style_descriptions").fetchone()[0]
        n_responses = self._conn.execute("SELECT COUNT(*) FROM llm_responses").fetchone()[0]

        var_rows = self._conn.execute(
            "SELECT variant_type, status, COUNT(*) AS n FROM variations "
            "GROUP BY variant_type, status"
        ).fetchall()
        variations: dict[str, dict[str, int]] = {}
        for row in var_rows:
            vtype = row["variant_type"]
            vstatus = row["status"]
            variations.setdefault(vtype, {})[vstatus] = row["n"]

        return {
            "songs_total": n_songs,
            "songs_with_lyrics": n_lyrics,
            "style_descriptions": n_styles,
            "llm_responses": n_responses,
            "variations": variations,
        }


# ---------------------------------------------------------------------------
# Row conversion helpers
# ---------------------------------------------------------------------------


def _row_to_song(row: sqlite3.Row) -> Song:
    d = dict(row)
    if d.get("fetched_at"):
        d["fetched_at"] = datetime.fromisoformat(d["fetched_at"])
    return Song(**d)


def _row_to_style(row: sqlite3.Row) -> StyleDescription:
    d = dict(row)
    d["source"] = StyleSource(d["source"])
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    return StyleDescription(**d)


def _row_to_llm_response(row: sqlite3.Row) -> LLMResponse:
    d = dict(row)
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    return LLMResponse(**d)


def _row_to_variation(row: sqlite3.Row) -> Variation:
    d = dict(row)
    d["variant_type"] = VariantType(d["variant_type"])
    d["status"] = VariantStatus(d["status"])
    phi_json = d.pop("phi_scores_json")
    d["phi_scores"] = PhiScores.from_json(phi_json)
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    return Variation(**d)
