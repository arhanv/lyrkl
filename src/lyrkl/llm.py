"""LLM client abstraction for lyrkl.

Provides a unified interface for calling Claude, Gemini, and OpenRouter
to generate phonetic lyric variations and style descriptions. All clients
accept a prompt string and return the raw response text.

Usage:
    from lyrkl.config import load_config
    from lyrkl.llm import get_client, build_apt_prompt, parse_candidates

    config = load_config("configs/default.yaml")
    client = get_client(config)
    prompt_text, prompt_hash = build_apt_prompt(song, config)
    response_text = client.generate(prompt_text)
    candidates = parse_candidates(response_text)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from lyrkl.config import LyrkIConfig
from lyrkl.models import Song

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract base for LLM provider clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the full response text.

        Args:
            prompt: The complete prompt string to send.

        Returns:
            The raw text content of the model's response.
        """


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------


class ClaudeClient(LLMClient):
    """Anthropic Claude client via the official anthropic SDK.

    Args:
        api_key: Anthropic API key.
        model: Model identifier string.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic is required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        """Call the Anthropic Messages API and return response text."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# Gemini (Google)
# ---------------------------------------------------------------------------


class GeminiClient(LLMClient):
    """Google Gemini client via google-generativeai.

    Args:
        api_key: Google AI Studio API key.
        model: Model identifier string.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required: pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    def generate(self, prompt: str) -> str:
        """Call the Gemini API and return response text."""
        response = self._model.generate_content(prompt)
        return response.text


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------


class OpenRouterClient(LLMClient):
    """OpenRouter client via httpx (OpenAI-compatible chat completions API).

    OpenRouter provides access to many models (GPT-4o, Llama, Mistral, etc.)
    through a single endpoint.

    Args:
        api_key: OpenRouter API key.
        model: Full model identifier (e.g. "meta-llama/llama-3.3-70b-instruct").
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
    """

    _API_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-haiku",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required: pip install httpx")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/lyrkl",
                "X-Title": "lyrkl",
            },
            timeout=120.0,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        """Call OpenRouter chat completions and return response text."""
        response = self._client.post(
            f"{self._API_BASE}/chat/completions",
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_client(config: LyrkIConfig) -> LLMClient:
    """Instantiate the appropriate LLM client from the config.

    Reads API keys from environment variables via config.llm_api_key().

    Args:
        config: LyrkI configuration.

    Returns:
        An LLMClient ready to call.

    Raises:
        ValueError: If the provider is unknown or the API key is missing.
    """
    provider = config.llm.provider.lower()
    api_key = config.llm_api_key()
    if not api_key:
        raise ValueError(
            f"No API key found for provider '{provider}'. "
            "Set ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in your environment."
        )

    if provider in ("claude", "anthropic"):
        return ClaudeClient(
            api_key=api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
        )
    elif provider in ("gemini", "google"):
        return GeminiClient(
            api_key=api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
        )
    elif provider == "openrouter":
        return OpenRouterClient(
            api_key=api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Choose from: claude, gemini, openrouter."
        )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _load_template(path: str) -> str:
    """Load a prompt template from a file.

    Args:
        path: Relative or absolute path to the template file.

    Returns:
        Template string with {placeholders}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return p.read_text(encoding="utf-8")


def _hash_prompt(text: str, model: str) -> str:
    """Compute a SHA-256 hex digest over prompt text + model name.

    Args:
        text: The filled prompt text.
        model: Model identifier string.

    Returns:
        64-character hex string.
    """
    content = f"model:{model}\n{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_apt_prompt(
    song: Song,
    config: LyrkIConfig,
    n_candidates: Optional[int] = None,
) -> tuple[str, str]:
    """Build the APT (Adversarial PhoneTic Prompting) prompt for a song.

    Loads the template from config.prompts.apt_primary and fills it with
    the song's details. Asks the LLM to produce N phonetic variants,
    each separated by the sentinel ``--- VARIANT ---``.

    Args:
        song: The source song.
        config: LyrkI configuration.
        n_candidates: Number of variant candidates to request. Defaults to
            config.llm.candidates_per_song.

    Returns:
        Tuple of (prompt_text, prompt_hash).
    """
    n = n_candidates or config.llm.candidates_per_song
    template = _load_template(config.prompts.apt_primary)
    prompt_text = template.format(
        title=song.title,
        artist=song.artist,
        genre=song.genre,
        lyrics=song.clean_lyrics,
        n_candidates=n,
    )
    return prompt_text, _hash_prompt(prompt_text, config.llm.model)


def build_style_prompt(song: Song, config: LyrkIConfig) -> tuple[str, str]:
    """Build a style description prompt using the LLM's world knowledge.

    The LLM is asked to produce a rich one-sentence style descriptor
    (instrumentation, tempo, mood, production era) for the song, without
    being given the lyrics -- relying on its training knowledge.

    Args:
        song: The source song.
        config: LyrkI configuration.

    Returns:
        Tuple of (prompt_text, prompt_hash).
    """
    template = _load_template(config.prompts.style_description)
    prompt_text = template.format(
        title=song.title,
        artist=song.artist,
        genre=song.genre,
    )
    return prompt_text, _hash_prompt(prompt_text, config.llm.model)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CANDIDATE_SEPARATOR = re.compile(
    r"---\s*VARIANT\s*(?:\d+\s*)?---", re.IGNORECASE
)


def parse_candidates(response_text: str) -> list[str]:
    """Parse multiple lyric candidates from a single LLM response.

    Candidates are separated by the sentinel ``--- VARIANT ---`` (or
    ``--- VARIANT N ---``). Each candidate is stripped of leading/trailing
    whitespace and empty candidates are discarded.

    Args:
        response_text: Raw text returned by the LLM.

    Returns:
        List of non-empty candidate strings in order.
    """
    parts = _CANDIDATE_SEPARATOR.split(response_text)
    candidates: list[str] = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            candidates.append(stripped)
    return candidates
