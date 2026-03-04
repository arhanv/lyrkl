"""Audio file linking and captioning model stub.

Provides utilities to register audio files in the lyrkl database and
an abstract base for captioning models that generate style descriptions
from audio.

The CaptioningModel ABC is intentionally unimplemented. To use a
real captioner (e.g. LP-MusicCaps, MERT, or any custom model),
subclass CaptioningModel and pass it to caption_and_save().

Usage:
    from lyrkl.audio import link_audio, caption_and_save

    # Register an audio file path for a song
    link_audio("eminem__lose_yourself", "/data/audio/lose_yourself.mp3", db)

    # Use a captioner (implement your own subclass)
    class MyCaptioner(CaptioningModel):
        def caption(self, audio_path: str) -> str:
            ...
    caption_and_save("eminem__lose_yourself", MyCaptioner(), db, config)
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from lyrkl.config import LyrkIConfig
from lyrkl.db import Database
from lyrkl.models import StyleDescription, StyleSource

logger = logging.getLogger(__name__)


class CaptioningModel(ABC):
    """Abstract base class for audio captioning models.

    Subclass this to plug in any audio-to-text captioning model.
    The caption method should return a single descriptive sentence
    suitable as a music generation prompt (style description).

    Args:
        model_name: Identifier string stored in StyleDescription.model.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def caption(self, audio_path: str) -> str:
        """Generate a style description from an audio file.

        Args:
            audio_path: Absolute or relative path to the audio file.

        Returns:
            A descriptive sentence suitable as a style prompt, e.g.:
            "Upbeat hip-hop track with heavy bass, fast-paced verses,
             prominent 808 drums, and aggressive vocal delivery."
        """


def link_audio(song_id: str, audio_path: str, db: Database) -> None:
    """Register an audio file path for a song in the database.

    Does not copy or move the file; only stores the path. Validates
    that the file exists at the given path.

    Args:
        song_id: The song slug to update.
        audio_path: Filesystem path to the audio file (.mp3, .wav, .flac, etc.).
        db: Open Database instance.

    Raises:
        FileNotFoundError: If the audio file does not exist at `audio_path`.
        ValueError: If no song with `song_id` exists in the database.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    song = db.get_song(song_id)
    if song is None:
        raise ValueError(f"No song with song_id '{song_id}' in database.")

    db.set_audio_path(song_id, str(path.resolve()))
    logger.info("Linked audio for '%s': %s", song_id, path.resolve())


def caption_and_save(
    song_id: str,
    captioner: CaptioningModel,
    db: Database,
    config: Optional[LyrkIConfig] = None,
    overwrite: bool = False,
) -> Optional[StyleDescription]:
    """Run a captioning model on a song's audio file and save the result.

    Retrieves the audio_path from the database, runs the captioner, and
    saves the resulting StyleDescription with source=captioner.

    Args:
        song_id: The song slug.
        captioner: A CaptioningModel instance.
        db: Open Database instance.
        config: Optional config (not used currently; reserved for future
            captioner settings).
        overwrite: If True, generate a new caption even if one from this
            captioner model already exists. Defaults to False.

    Returns:
        The saved StyleDescription, or None if skipped (no audio path,
        or description already exists and overwrite=False).
    """
    song = db.get_song(song_id)
    if song is None:
        logger.warning("caption_and_save: no song '%s' in DB", song_id)
        return None

    if not song.audio_path:
        logger.warning(
            "caption_and_save: no audio_path for '%s'. "
            "Call link_audio() first.",
            song_id,
        )
        return None

    if not overwrite:
        existing = db.get_latest_style(song_id, StyleSource.CAPTIONER)
        if existing and existing.model == captioner.model_name:
            logger.info(
                "caption_and_save: caption already exists for '%s' with model '%s'",
                song_id,
                captioner.model_name,
            )
            return existing

    logger.info(
        "Running captioner '%s' on '%s'", captioner.model_name, song.audio_path
    )
    caption_text = captioner.caption(song.audio_path)

    desc = StyleDescription(
        desc_id=str(uuid.uuid4()),
        song_id=song_id,
        text=caption_text,
        source=StyleSource.CAPTIONER,
        model=captioner.model_name,
        prompt_hash="",
        created_at=datetime.utcnow(),
    )
    db.save_style_description(desc)
    logger.info("Saved caption for '%s': %s", song_id, caption_text[:80])
    return desc
