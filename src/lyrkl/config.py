"""Configuration loading for lyrkl.

Config is stored in a YAML file (default: configs/default.yaml) with
optional overrides from environment variables for API keys.

Usage:
    config = load_config("configs/default.yaml")
    client = get_llm_client(config)   # picks provider from config.llm.provider
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeniusConfig:
    """Genius API settings."""

    rate_limit_delay: float = 1.0


@dataclass
class LLMConfig:
    """LLM provider and generation settings."""

    provider: str = "claude"
    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 1.0
    max_tokens: int = 4096
    candidates_per_song: int = 5


@dataclass
class PhiWeights:
    """Per-component weights for the aggregate Phi score.

    Weights are normalized to sum to 1.0 on load.
    """

    phoneme: float = 0.20
    rhyme: float = 0.20
    syllable: float = 0.15
    stress: float = 0.15
    jaccard: float = 0.10
    cv_pattern: float = 0.10
    stressed_vowel: float = 0.10


@dataclass
class PhiConfig:
    """Phi scoring and filtering configuration."""

    min_aggregate: float = 0.70
    weights: PhiWeights = field(default_factory=PhiWeights)


@dataclass
class PromptsConfig:
    """Paths to LLM prompt template files."""

    apt_primary: str = "prompts/apt_primary.txt"
    apt_fallback: str = "prompts/apt_fallback.txt"
    style_description: str = "prompts/style_description.txt"


@dataclass
class LyrkIConfig:
    """Root configuration object.

    Attributes:
        genius: Genius API settings.
        llm: LLM provider + generation settings.
        phi: Phi scoring and filtering settings.
        prompts: Paths to prompt template files.
        data_dir: Directory for the SQLite DB and raw LLM response files.
        api_keys: Dict of provider name -> key (populated from env).
    """

    genius: GeniusConfig = field(default_factory=GeniusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    data_dir: str = "data"
    api_keys: dict[str, str] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        """Absolute path to the SQLite database file."""
        return Path(self.data_dir) / "lyrkl.db"

    @property
    def llm_responses_dir(self) -> Path:
        """Directory for raw LLM response text files."""
        return Path(self.data_dir) / "llm_responses"

    def genius_api_key(self) -> Optional[str]:
        """Return the Genius API key from env or api_keys dict."""
        return self.api_keys.get("genius") or os.environ.get("GENIUS_API_KEY")

    def llm_api_key(self) -> Optional[str]:
        """Return the LLM API key for the configured provider."""
        provider = self.llm.provider.lower()
        env_map = {
            "claude": "ANTHROPIC_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = env_map.get(provider)
        if env_var:
            return self.api_keys.get(provider) or os.environ.get(env_var)
        return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_config(path: str | Path, env_file: str | Path = ".env") -> LyrkIConfig:
    """Load config from a YAML file and overlay API keys from environment.

    Args:
        path: Path to the YAML config file.
        env_file: Optional path to a .env file to load before reading env vars.

    Returns:
        A fully populated LyrkIConfig.
    """
    load_dotenv(env_file, override=False)

    raw: dict[str, Any] = {}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = LyrkIConfig()

    # Genius
    if genius_raw := raw.get("genius"):
        cfg.genius = GeniusConfig(**genius_raw)

    # LLM
    if llm_raw := raw.get("llm"):
        cfg.llm = LLMConfig(**llm_raw)

    # Phi
    if phi_raw := raw.get("phi"):
        weights_raw = phi_raw.pop("weights", {})
        cfg.phi = PhiConfig(**phi_raw)
        if weights_raw:
            cfg.phi.weights = PhiWeights(**weights_raw)
        _normalize_weights(cfg.phi.weights)

    # Prompts
    if prompts_raw := raw.get("prompts"):
        cfg.prompts = PromptsConfig(**prompts_raw)

    # Data dir
    if "data_dir" in raw:
        cfg.data_dir = raw["data_dir"]

    # Inline API keys (not recommended; prefer env vars)
    if api_keys := raw.get("api_keys", {}):
        cfg.api_keys = api_keys

    return cfg


def _normalize_weights(weights: PhiWeights) -> None:
    """Normalize PhiWeights so they sum to 1.0, in-place."""
    total = (
        weights.phoneme
        + weights.rhyme
        + weights.syllable
        + weights.stress
        + weights.jaccard
        + weights.cv_pattern
        + weights.stressed_vowel
    )
    if total <= 0:
        return
    for attr in ("phoneme", "rhyme", "syllable", "stress", "jaccard", "cv_pattern", "stressed_vowel"):
        setattr(weights, attr, getattr(weights, attr) / total)


def default_config() -> LyrkIConfig:
    """Return a default LyrkIConfig without loading any file."""
    load_dotenv(".env", override=False)
    return LyrkIConfig()
