"""Pipeline orchestrator for lyrkl.

Provides LyrkIPipeline, which coordinates all stages of the variation
generation workflow:

  1. add_songs  -- register songs in the database
  2. fetch_lyrics -- pull lyrics from Genius (skips already-fetched songs)
  3. generate_style -- generate LLM style descriptions (incremental)
  4. generate_variations -- call LLM, score with Phi, save accepted/rejected
  5. export_prompt_dataset -- write acelm-interp-compatible JSON

Each stage can be run independently. Idempotency is guaranteed by the
database layer: re-running a stage with the same config is always safe.

Usage:
    from lyrkl.config import load_config
    from lyrkl.db import Database
    from lyrkl.pipeline import LyrkIPipeline

    config = load_config("configs/default.yaml")
    db = Database(config.db_path)
    pipeline = LyrkIPipeline(config, db)

    pipeline.run_all([
        {"title": "Lose Yourself", "artist": "Eminem", "genre": "hip-hop"},
    ])
    pipeline.export_prompt_dataset("data/prompts.json")
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from lyrkl.config import LyrkIConfig
from lyrkl.db import Database
from lyrkl.llm import (
    LLMClient,
    build_apt_prompt,
    build_style_prompt,
    get_client,
    parse_candidates,
)
from lyrkl.lyrics import ArtistSongFilter, GeniusLyricsFetcher, make_song_id
from lyrkl.models import (
    LLMResponse,
    Song,
    StyleDescription,
    StyleSource,
    Variation,
    VariantStatus,
    VariantType,
)
from lyrkl.phi import filter_candidates
from lyrkl.variants import shuffle_lyrics_lines

logger = logging.getLogger(__name__)


class LyrkIPipeline:
    """Orchestrates the full lyrkl workflow.

    Args:
        config: LyrkI configuration object.
        db: Open Database instance. The pipeline does not own nor close the DB.
    """

    def __init__(self, config: LyrkIConfig, db: Database) -> None:
        self._config = config
        self._db = db
        self._llm_client: Optional[LLMClient] = None

    # ------------------------------------------------------------------
    # Stage 1: Add songs
    # ------------------------------------------------------------------

    def add_songs(self, song_list: list[dict[str, Any]]) -> list[str]:
        """Register songs in the database without fetching lyrics.

        Idempotent: already-existing songs are updated with any new fields
        provided (e.g. genre), but lyrics are not overwritten.

        Args:
            song_list: List of song spec dicts. Each dict should have either:
                - "title" and "artist" (and optionally "genre", "genius_id")
                - "genius_id" (and optionally "genre")

        Returns:
            List of song_id slugs that were added or already present.
        """
        ids: list[str] = []
        for spec in song_list:
            spec = spec.copy()
            genius_id = spec.get("genius_id")
            title = spec.get("title", f"genius_{genius_id}")
            artist = spec.get("artist", "Unknown")
            genre = spec.get("genre", "")
            artist_gender = spec.get("artist_gender")
            song_id = spec.get("song_id") or make_song_id(artist, title)

            song = Song(
                song_id=song_id,
                title=title,
                artist=artist,
                genre=genre,
                genius_id=genius_id,
                artist_gender=artist_gender,
            )
            self._db.upsert_song(song)
            ids.append(song_id)
            logger.debug("Registered song: %s", song_id)

        logger.info("add_songs: registered %d songs.", len(ids))
        return ids

    # ------------------------------------------------------------------
    # Artist resolution
    # ------------------------------------------------------------------

    def resolve_artist(
        self,
        artist: str,
        filter: ArtistSongFilter = None,
        genre: str = "",
        artist_gender: Optional[str] = None,
    ) -> list[str]:
        """Resolve an artist's top songs via Genius and register them in the DB.

        Fetches up to filter.max_songs songs for the given artist, applies the
        configured filters, and registers the results with add_songs(). Does not
        fetch lyrics; call fetch_lyrics() after this step.

        Args:
            artist: Artist name to search for on Genius.
            filter: ArtistSongFilter controlling filtering behaviour.
                Defaults to ArtistSongFilter() with all defaults.
            genre: Genre label to attach to every resolved song.
            artist_gender: Optional gender label (e.g. "Female", "Male").

        Returns:
            List of song_id slugs that were registered.
        """
        if filter is None:
            filter = ArtistSongFilter()

        api_key = self._config.genius_api_key()
        if not api_key:
            raise ValueError(
                "No Genius API key found. Set GENIUS_API_KEY in your environment."
            )

        fetcher = GeniusLyricsFetcher(
            api_token=api_key,
            rate_limit_delay=self._config.genius.rate_limit_delay,
            timeout=self._config.genius.timeout,
        )
        song_specs = fetcher.resolve_artist_songs(artist, filter=filter, genre=genre)
        if artist_gender is not None:
            for spec in song_specs:
                spec["artist_gender"] = artist_gender
        return self.add_songs(song_specs)

    def resolve_artists(
        self,
        artist_list: list[dict[str, Any]],
        default_filter: ArtistSongFilter = None,
    ) -> list[str]:
        """Resolve multiple artists and register all their songs in the DB.

        Each entry in artist_list must have a "name" key and may include any
        ArtistSongFilter field as an override (max_songs, sort, max_year,
        include_featured_vocals, include_collaborations, exclude_remixes,
        exclude_covers, genre).

        Args:
            artist_list: List of artist spec dicts. Required key: "name".
                Optional keys mirror ArtistSongFilter fields plus "genre".
            default_filter: Base ArtistSongFilter applied to all artists.
                Per-artist keys in artist_list override these defaults.

        Returns:
            Combined list of song_id slugs registered across all artists.
        """
        if default_filter is None:
            default_filter = ArtistSongFilter()

        all_ids: list[str] = []
        for spec in artist_list:
            spec = spec.copy()
            artist_name = spec.pop("name")
            genre = spec.pop("genre", "")
            artist_gender = spec.pop("artist_gender", None)

            # Build per-artist filter by merging defaults with any overrides.
            filter_kwargs = {
                "max_songs": default_filter.max_songs,
                "sort": default_filter.sort,
                "include_featured_vocals": default_filter.include_featured_vocals,
                "include_collaborations": default_filter.include_collaborations,
                "exclude_remixes": default_filter.exclude_remixes,
                "exclude_covers": default_filter.exclude_covers,
                "max_year": default_filter.max_year,
            }
            for key in list(filter_kwargs):
                if key in spec:
                    filter_kwargs[key] = spec.pop(key)

            per_artist_filter = ArtistSongFilter(**filter_kwargs)
            ids = self.resolve_artist(
                artist_name, filter=per_artist_filter, genre=genre,
                artist_gender=artist_gender,
            )
            all_ids.extend(ids)

        logger.info(
            "resolve_artists: registered %d songs across %d artists.",
            len(all_ids), len(artist_list),
        )
        return all_ids

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def check_duplicates(
        self,
        song_ids: Optional[list[str]] = None,
        threshold: float = 0.8,
    ) -> list[tuple[str, str, float]]:
        """Find pairs of songs with highly similar lyrics.

        Computes pairwise Jaccard similarity on word bags of clean_lyrics.
        Songs without lyrics are skipped. Results are logged as warnings.

        Args:
            song_ids: If provided, only check these songs. Defaults to all
                songs with lyrics.
            threshold: Minimum Jaccard similarity to flag a pair (0.0-1.0).

        Returns:
            List of (song_id_a, song_id_b, jaccard_score) tuples for all pairs
            at or above the threshold, sorted descending by score.
        """
        if song_ids is not None:
            songs = [s for sid in song_ids if (s := self._db.get_song(sid)) is not None]
        else:
            songs = self._db.list_songs()

        # Genius ID collision check: same genius_id registered under multiple
        # song_ids. This can happen when a song appears in multiple artists'
        # catalogs (e.g. a featured appearance). Runs on all songs, not just
        # those with lyrics, so it catches issues before fetch is run.
        gid_map: dict[int, list[str]] = {}
        for s in songs:
            if s.genius_id is not None:
                gid_map.setdefault(s.genius_id, []).append(s.song_id)
        for gid, sids in gid_map.items():
            if len(sids) > 1:
                logger.warning(
                    "check_duplicates: genius_id=%d appears under %d song_ids: %s",
                    gid, len(sids), sids,
                )

        songs = [s for s in songs if s.has_lyrics]

        flagged: list[tuple[str, str, float]] = []

        for i in range(len(songs)):
            words_i = set(songs[i].clean_lyrics.lower().split())
            if not words_i:
                continue
            for j in range(i + 1, len(songs)):
                words_j = set(songs[j].clean_lyrics.lower().split())
                if not words_j:
                    continue
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= threshold:
                    flagged.append((songs[i].song_id, songs[j].song_id, jaccard))

        flagged.sort(key=lambda t: t[2], reverse=True)

        if flagged:
            logger.warning(
                "check_duplicates: found %d potential duplicate pair(s) "
                "(threshold=%.2f):",
                len(flagged), threshold,
            )
            for a, b, score in flagged:
                logger.warning("  %s  <->  %s  (jaccard=%.3f)", a, b, score)
        else:
            logger.info(
                "check_duplicates: no duplicates found (threshold=%.2f).", threshold
            )

        return flagged

    # ------------------------------------------------------------------
    # Stage 2: Fetch lyrics
    # ------------------------------------------------------------------

    def fetch_lyrics(
        self,
        song_ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Fetch lyrics from Genius for songs that don't have them yet.

        Skips songs that already have clean_lyrics in the database.

        Args:
            song_ids: If provided, only fetch these specific songs.
                Defaults to all songs without lyrics.

        Returns:
            List of song_id slugs for which lyrics were successfully fetched.
        """
        api_key = self._config.genius_api_key()
        if not api_key:
            raise ValueError(
                "No Genius API key found. Set GENIUS_API_KEY in your environment."
            )

        if song_ids is not None:
            pending = [
                self._db.get_song(sid)
                for sid in song_ids
                if not (s := self._db.get_song(sid)) or not s.has_lyrics
            ]
            pending = [s for s in pending if s is not None]
        else:
            pending = self._db.songs_without_lyrics()

        if not pending:
            logger.info("fetch_lyrics: all songs already have lyrics.")
            return []

        fetcher = GeniusLyricsFetcher(
            api_token=api_key,
            rate_limit_delay=self._config.genius.rate_limit_delay,
            timeout=self._config.genius.timeout,
        )

        fetched: list[str] = []
        for song in pending:
            if song.genius_id:
                result = fetcher.fetch_by_genius_id(
                    song.genius_id, genre=song.genre
                )
                if result:
                    result.song_id = song.song_id
            else:
                result = fetcher.fetch_one(
                    song.artist, song.title, genre=song.genre
                )

            if result is None:
                logger.warning(
                    "fetch_lyrics: could not fetch '%s - %s'",
                    song.artist, song.title,
                )
                continue

            self._db.upsert_song(result)
            fetched.append(result.song_id)
            logger.info("fetch_lyrics: fetched '%s'", result.song_id)

        logger.info(
            "fetch_lyrics: fetched %d/%d songs.", len(fetched), len(pending)
        )
        return fetched

    # ------------------------------------------------------------------
    # Stage 3: Generate style descriptions
    # ------------------------------------------------------------------

    def generate_style(
        self,
        song_ids: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Generate LLM style descriptions for songs that don't have one yet.

        Uses the LLM's world knowledge (title, artist, genre) to produce a
        rich one-sentence style description. Skips songs that already have
        an LLM-sourced style description for the current prompt hash, unless
        overwrite=True.

        Falls back to a template-based description if no LLM key is available.

        Args:
            song_ids: If provided, only process these songs.
            overwrite: If True, generate new descriptions even if they exist.

        Returns:
            List of song_ids for which a style description was generated.
        """
        client = self._get_llm_client(required=False)
        use_llm = client is not None

        if not use_llm:
            logger.warning(
                "generate_style: no LLM API key found; using template fallback."
            )

        if song_ids is not None:
            songs = [s for sid in song_ids if (s := self._db.get_song(sid)) is not None]
        else:
            songs = self._db.list_songs()

        songs = [s for s in songs if s.has_lyrics]

        completed: list[str] = []
        for song in songs:
            if use_llm:
                prompt_text, prompt_hash = build_style_prompt(song, self._config)
                if not overwrite:
                    existing = self._db.get_latest_style(song.song_id, StyleSource.LLM)
                    if existing and existing.prompt_hash == prompt_hash:
                        logger.debug(
                            "generate_style: style already exists for '%s'",
                            song.song_id,
                        )
                        continue

                logger.info("generate_style: generating for '%s'", song.song_id)
                try:
                    response_text = client.generate(prompt_text)
                except Exception as e:
                    logger.error(
                        "generate_style: LLM call failed for '%s': %s",
                        song.song_id, e,
                    )
                    continue

                # Save raw response
                self._save_raw_response(
                    song.song_id, "style", prompt_text,
                    prompt_hash, response_text, n_candidates=1,
                )

                style_text = response_text.strip()
                desc = StyleDescription(
                    desc_id=str(uuid.uuid4()),
                    song_id=song.song_id,
                    text=style_text,
                    source=StyleSource.LLM,
                    model=self._config.llm.model,
                    prompt_hash=prompt_hash,
                    created_at=datetime.utcnow(),
                )
            else:
                # Template fallback
                if not overwrite:
                    existing = self._db.get_latest_style(
                        song.song_id, StyleSource.TEMPLATE
                    )
                    if existing:
                        continue

                style_text = (
                    f"A {song.genre} song in the style of {song.artist}, "
                    f"featuring characteristic instrumentation and production."
                )
                desc = StyleDescription(
                    desc_id=str(uuid.uuid4()),
                    song_id=song.song_id,
                    text=style_text,
                    source=StyleSource.TEMPLATE,
                    model="template",
                    prompt_hash="",
                    created_at=datetime.utcnow(),
                )

            self._db.save_style_description(desc)
            completed.append(song.song_id)

        logger.info(
            "generate_style: generated style for %d songs.", len(completed)
        )
        return completed

    # ------------------------------------------------------------------
    # Stage 4: Generate variations
    # ------------------------------------------------------------------

    def generate_variations(
        self,
        song_ids: Optional[list[str]] = None,
        variant_type: VariantType = VariantType.PHONETIC,
    ) -> dict[str, dict[str, int]]:
        """Generate LLM lyric variations, score with Phi, and save to DB.

        For each song not yet processed with the current prompt hash,
        calls the LLM candidates_per_song times, parses candidates from
        the response, scores each with the Phi metric, and saves both
        accepted and rejected candidates.

        Args:
            song_ids: If provided, only process these songs. Defaults to all
                songs with lyrics that have not yet been varied with this prompt.
            variant_type: Variant type to generate (default: PHONETIC).

        Returns:
            Dict mapping song_id -> {"accepted": n, "rejected": n}.
        """
        client = self._get_llm_client(required=True)

        # Determine prompt hash for idempotency check
        _dummy_song = Song(song_id="x", title="T", artist="A", genre="g",
                           clean_lyrics="l", raw_lyrics="l")
        _, prompt_hash_template = build_apt_prompt(_dummy_song, self._config)

        if song_ids is not None:
            songs = [s for sid in song_ids if (s := self._db.get_song(sid)) is not None]
        else:
            songs = self._db.songs_without_variations(prompt_hash_template)

        if not songs:
            logger.info("generate_variations: nothing to process.")
            return {}

        summary: dict[str, dict[str, int]] = {}

        for song in songs:
            if not song.has_lyrics:
                logger.warning(
                    "generate_variations: '%s' has no lyrics; skipping.",
                    song.song_id,
                )
                continue

            prompt_text, prompt_hash = build_apt_prompt(song, self._config)

            # Skip if already done with this exact prompt
            if self._db.response_exists(song.song_id, prompt_hash, "apt"):
                logger.debug(
                    "generate_variations: already generated for '%s'",
                    song.song_id,
                )
                continue

            logger.info(
                "generate_variations: calling LLM for '%s'", song.song_id
            )

            try:
                response_text = client.generate(prompt_text)
            except Exception as e:
                logger.error(
                    "generate_variations: LLM call failed for '%s': %s",
                    song.song_id, e,
                )
                continue

            candidates = parse_candidates(response_text)
            if not candidates:
                logger.warning(
                    "generate_variations: no candidates parsed for '%s'. "
                    "Raw response:\n%s",
                    song.song_id, response_text[:500],
                )
                # Still save the response so we don't call again
                candidates = []

            # Save raw LLM response
            response_id = self._save_raw_response(
                song.song_id, "apt", prompt_text,
                prompt_hash, response_text, n_candidates=len(candidates),
            )

            # Score and filter
            accepted, rejected = filter_candidates(
                song.clean_lyrics, candidates, self._config
            )

            variations: list[Variation] = []
            for idx, (lyrics, phi) in enumerate(accepted):
                variations.append(
                    self._make_variation(
                        song, lyrics, phi, prompt_hash,
                        response_id, VariantStatus.ACCEPTED,
                        variant_type, idx,
                    )
                )
            for idx, (lyrics, phi) in enumerate(rejected):
                variations.append(
                    self._make_variation(
                        song, lyrics, phi, prompt_hash,
                        response_id, VariantStatus.REJECTED,
                        variant_type, len(accepted) + idx,
                    )
                )

            self._db.save_variations(variations)

            n_accepted = len(accepted)
            n_rejected = len(rejected)
            summary[song.song_id] = {
                "accepted": n_accepted,
                "rejected": n_rejected,
            }
            logger.info(
                "generate_variations: '%s' -> %d accepted, %d rejected (phi >= %.2f)",
                song.song_id, n_accepted, n_rejected,
                self._config.phi.min_aggregate,
            )

        return summary

    # ------------------------------------------------------------------
    # Convenience: run all stages
    # ------------------------------------------------------------------

    def run_all(
        self,
        song_list: list[dict[str, Any]],
        generate_style: bool = True,
    ) -> None:
        """Run all pipeline stages in order.

        Equivalent to calling:
            add_songs -> fetch_lyrics -> generate_style -> generate_variations

        Args:
            song_list: Song spec dicts (same format as add_songs).
            generate_style: If True (default), generate LLM style descriptions
                before variations.
        """
        self.add_songs(song_list)
        self.fetch_lyrics()
        if generate_style:
            self.generate_style()
        self.generate_variations()

    # ------------------------------------------------------------------
    # Audio linking
    # ------------------------------------------------------------------

    def link_audio(self, song_id: str, audio_path: str) -> None:
        """Register an audio file path for a song.

        Args:
            song_id: Song slug.
            audio_path: Filesystem path to an audio file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the song is not in the database.
        """
        from lyrkl.audio import link_audio as _link_audio
        _link_audio(song_id, audio_path, self._db)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_prompt_dataset(
        self,
        path: str | Path,
        song_ids: Optional[list[str]] = None,
        variant_types: Optional[list[VariantType]] = None,
    ) -> int:
        """Export accepted variations as acelm-interp PromptDataset JSON.

        Args:
            path: Output file path. Parent directories are created.
            song_ids: If provided, only include these songs.
            variant_types: If provided, only include these variant types.

        Returns:
            Number of variation records written.
        """
        records = self._db.to_prompt_dataset(
            song_ids=song_ids, variant_types=variant_types
        )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("Exported %d records to %s", len(records), out)
        return len(records)

    def export_verbatim_dataset(
        self,
        path: str | Path,
        song_ids: Optional[list[str]] = None,
        section: Optional[str] = None,
        caption: str = "",
        shuffle_lines: bool = False,
        shuffle_seed: int = 0,
    ) -> int:
        """Export verbatim lyrics from the songs table as a PromptDataset JSON.

        Reads directly from the songs table (no variations required). Useful
        for verbatim baseline experiments and encoder probing.

        Args:
            path: Output file path. Parent directories are created.
            song_ids: If provided, only include these songs. Defaults to all
                songs that have lyrics.
            section: If provided, extract only this named section from each
                song's lyrics (e.g. "chorus"). Matches variants like
                [Chorus 1], [Chorus: Artist], [Hook], [Refrain]. Songs
                without the requested section are skipped.
            caption: Caption embedded in every record. Defaults to "" (no
                caption), which is recommended for encoder probing experiments.
                Pass "auto" to use the DB style description instead.
            shuffle_lines: If True, export a shuffled-line control variant
                instead of strict verbatim lyrics.
            shuffle_seed: Base deterministic seed for line shuffling. The
                effective per-song seed is `shuffle_seed + song_index`.

        Returns:
            Number of records written.
        """
        import re

        all_songs = self._db.list_songs()
        if song_ids is not None:
            song_id_set = set(song_ids)
            all_songs = [s for s in all_songs if s.song_id in song_id_set]

        # Build section regex if needed. Aliases: chorus also catches hook/refrain.
        _aliases: dict[str, list[str]] = {
            "chorus": ["chorus", "hook", "refrain"],
        }

        def _make_section_re(name: str) -> re.Pattern:
            aliases = _aliases.get(name.lower(), [name.lower()])
            alias_pat = "|".join(re.escape(a) for a in aliases)
            return re.compile(
                rf"\[(?:{alias_pat})(?:\s+\d+)?(?::[^\]]+)?\](.*?)(?=\n\[|$)",
                re.IGNORECASE | re.DOTALL,
            )

        section_re = _make_section_re(section) if section else None

        records = []
        skipped = []

        for song_index, song in enumerate(all_songs):
            if not song.has_lyrics:
                skipped.append(f"{song.song_id} (no lyrics)")
                continue

            if section_re is not None:
                match = section_re.search(song.clean_lyrics)
                if not match:
                    skipped.append(f"{song.song_id} (no [{section}] section found)")
                    continue
                lyrics_text = match.group(1).strip()
            else:
                lyrics_text = song.clean_lyrics

            if shuffle_lines:
                effective_seed = shuffle_seed + song_index
                lyrics_text = shuffle_lyrics_lines(lyrics_text, effective_seed)
                variant_value = "shuffled"
                prompt_hash = f"shuffled_lines_seed_{shuffle_seed}"
                var_suffix = f"shuffled_{effective_seed}_{section or 'full'}"
            else:
                effective_seed = None
                variant_value = "verbatim"
                prompt_hash = "verbatim_baseline"
                var_suffix = f"verbatim_{section or 'full'}"

            if caption == "auto":
                cap = self._db._get_style_text(song.song_id) or f"A {song.genre} song by {song.artist}."
            else:
                cap = caption

            records.append({
                "song_id": song.song_id,
                "lyrics": lyrics_text,
                "caption": cap,
                "variant": variant_value,
                "metadata": {
                    "title": song.title,
                    "artist": song.artist,
                    "genre": song.genre,
                    "artist_gender": song.artist_gender,
                    "phi_aggregate": 1.0,
                    "phi_scores": {},
                    "prompt_hash": prompt_hash,
                    "var_id": f"{song.song_id}_{var_suffix}",
                    "status": "accepted",
                    "source_variant": "verbatim",
                    "shuffle_seed": effective_seed,
                },
            })

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("Exported %d verbatim records to %s", len(records), out)

        if skipped:
            logger.warning("Skipped %d songs:", len(skipped))
            for s in skipped:
                logger.warning("  - %s", s)

        return len(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm_client(self, required: bool = True) -> Optional[LLMClient]:
        """Return (and cache) the LLM client, or None if no key is set."""
        if self._llm_client is not None:
            return self._llm_client
        try:
            self._llm_client = get_client(self._config)
            return self._llm_client
        except ValueError as e:
            if required:
                raise
            logger.debug("LLM client not available: %s", e)
            return None

    def _save_raw_response(
        self,
        song_id: str,
        purpose: str,
        prompt_text: str,
        prompt_hash: str,
        response_text: str,
        n_candidates: int,
    ) -> str:
        """Save an LLM response to DB and to a raw text file.

        Returns:
            The response_id UUID string.
        """
        response_id = str(uuid.uuid4())
        raw_dir = self._config.llm_responses_dir / song_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{response_id}.txt"
        raw_path.write_text(
            f"PROMPT:\n{prompt_text}\n\n---\nRESPONSE:\n{response_text}",
            encoding="utf-8",
        )

        response = LLMResponse(
            response_id=response_id,
            song_id=song_id,
            purpose=purpose,
            model=self._config.llm.model,
            prompt_hash=prompt_hash,
            prompt_text=prompt_text,
            response_text=response_text,
            n_candidates=n_candidates,
            raw_file_path=str(raw_path.resolve()),
            created_at=datetime.utcnow(),
        )
        self._db.save_llm_response(response)
        return response_id

    def _make_variation(
        self,
        song: Song,
        lyrics: str,
        phi,
        prompt_hash: str,
        llm_response_id: str,
        status: VariantStatus,
        variant_type: VariantType,
        candidate_index: int,
    ) -> Variation:
        """Create a content-addressed Variation object."""
        import hashlib
        var_id = hashlib.sha256(
            f"{song.song_id}|{prompt_hash}|{candidate_index}|{lyrics}".encode()
        ).hexdigest()

        return Variation(
            var_id=var_id,
            song_id=song.song_id,
            variant_type=variant_type,
            lyrics=lyrics,
            phi_scores=phi,
            phi_aggregate=phi.aggregate,
            prompt_hash=prompt_hash,
            llm_response_id=llm_response_id,
            status=status,
            created_at=datetime.utcnow(),
            candidate_index=candidate_index,
        )
