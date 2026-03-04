"""Full 7-component Phi phonetic similarity metric.

Implements the Phi metric from Bob's Confetti (Roh et al., 2025):

    Phi_agg(a, b) = weighted mean of 7 sub-scores

The 7 components are:
    S_ph  -- phoneme sequence similarity (SequenceMatcher on CMUdict tokens)
    S_rh  -- rhyme: overlap of terminal phoneme suffix
    S_sy  -- syllable count ratio
    S_st  -- stress pattern alignment
    S_jac -- phoneme-set Jaccard similarity
    S_cv  -- consonant-vowel skeleton pattern comparison
    S_vow -- stressed vowel core alignment

Scoring is done at the word level, averaged per line, averaged per song.
Only words with CMUdict entries are compared; function words and OOV words
are skipped gracefully.

Requires: pip install pronouncing>=2.2

Usage:
    from lyrkl.phi import score_phi, filter_candidates
    from lyrkl.config import LyrkIConfig

    cfg = LyrkIConfig()
    scores = score_phi("His palms are sweaty", "His palms are sweaty", cfg)
    accepted, rejected = filter_candidates(
        original_lyrics="His palms are sweaty\\nKnees weak arms are heavy",
        candidates=["His palms are sweaty\\nCheese weak cars are heavy"],
        config=cfg,
    )
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from lyrkl.config import LyrkIConfig, PhiWeights
from lyrkl.models import PhiScores


# ---------------------------------------------------------------------------
# CMUdict helpers
# ---------------------------------------------------------------------------


def _get_phones(word: str) -> Optional[list[str]]:
    """Return the primary CMUdict phoneme list for a word, or None.

    Args:
        word: A single word (case-insensitive).

    Returns:
        List of CMUdict phoneme tokens (e.g. ['P', 'AE1', 'L', 'M', 'Z']),
        or None if the word is not in the dictionary.
    """
    try:
        import pronouncing
    except ImportError:
        raise ImportError("pronouncing>=2.2 is required: pip install 'pronouncing>=2.2'")

    entries = pronouncing.phones_for_word(word.lower())
    if not entries:
        return None
    return entries[0].split()


def _syllables(phones: list[str]) -> int:
    """Count syllables: number of vowel phonemes (those ending with a digit)."""
    return sum(1 for p in phones if p[-1].isdigit())


def _stresses(phones: list[str]) -> list[int]:
    """Extract stress digits from phonemes: 0=unstressed, 1=primary, 2=secondary."""
    return [int(p[-1]) for p in phones if p[-1].isdigit()]


def _strip_stress(phones: list[str]) -> list[str]:
    """Remove stress digits from phoneme tokens."""
    return [re.sub(r"\d", "", p) for p in phones]


def _cv_skeleton(phones: list[str]) -> str:
    """Consonant-vowel skeleton: 'C' for consonants, 'V' for vowels."""
    result = []
    for p in phones:
        result.append("V" if p[-1].isdigit() else "C")
    return "".join(result)


def _terminal_phonemes(phones: list[str], n: int = 4) -> list[str]:
    """Return the last n phonemes (the rhyme nucleus + coda)."""
    return phones[-n:] if len(phones) >= n else phones


def _stressed_vowels(phones: list[str]) -> list[str]:
    """Return only phonemes carrying primary stress (stress digit = 1)."""
    return [re.sub(r"\d", "", p) for p in phones if p.endswith("1")]


# ---------------------------------------------------------------------------
# Per-word component scores
# ---------------------------------------------------------------------------


def _s_ph(a: list[str], b: list[str]) -> float:
    """S_ph: SequenceMatcher ratio on stripped phoneme sequences."""
    sa = _strip_stress(a)
    sb = _strip_stress(b)
    return SequenceMatcher(None, sa, sb).ratio()


def _s_rh(a: list[str], b: list[str]) -> float:
    """S_rh: proportion of terminal phonemes shared."""
    ta = _terminal_phonemes(_strip_stress(a))
    tb = _terminal_phonemes(_strip_stress(b))
    if not ta or not tb:
        return 0.0
    shared = sum(1 for x in ta if x in tb)
    return shared / max(len(ta), len(tb))


def _s_sy(a: list[str], b: list[str]) -> float:
    """S_sy: syllable count similarity as min/max ratio."""
    na = _syllables(a)
    nb = _syllables(b)
    if max(na, nb) == 0:
        return 1.0
    return min(na, nb) / max(na, nb)


def _s_st(a: list[str], b: list[str]) -> float:
    """S_st: proportion of aligned stress positions that match."""
    sta = _stresses(a)
    stb = _stresses(b)
    n = min(len(sta), len(stb))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(sta[:n], stb[:n]) if x == y) / n


def _s_jac(a: list[str], b: list[str]) -> float:
    """S_jac: Jaccard similarity of phoneme sets (ignoring stress)."""
    sa = set(_strip_stress(a))
    sb = set(_strip_stress(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _s_cv(a: list[str], b: list[str]) -> float:
    """S_cv: SequenceMatcher ratio on consonant-vowel skeleton strings."""
    cva = _cv_skeleton(a)
    cvb = _cv_skeleton(b)
    return SequenceMatcher(None, list(cva), list(cvb)).ratio()


def _s_vow(a: list[str], b: list[str]) -> float:
    """S_vow: proportion of primary-stressed vowels shared."""
    va = _stressed_vowels(a)
    vb = _stressed_vowels(b)
    if not va and not vb:
        return 1.0
    if not va or not vb:
        return 0.0
    shared = sum(1 for x in va if x in vb)
    return shared / max(len(va), len(vb))


# ---------------------------------------------------------------------------
# Word-pair aggregate
# ---------------------------------------------------------------------------


def _score_word_pair(
    phones_a: list[str],
    phones_b: list[str],
    weights: PhiWeights,
) -> PhiScores:
    """Compute all 7 component scores for a single word-pair.

    Args:
        phones_a: CMUdict phoneme list for the original word.
        phones_b: CMUdict phoneme list for the modified word.
        weights: PhiWeights from config.

    Returns:
        PhiScores with all components and weighted aggregate.
    """
    ph = _s_ph(phones_a, phones_b)
    rh = _s_rh(phones_a, phones_b)
    sy = _s_sy(phones_a, phones_b)
    st = _s_st(phones_a, phones_b)
    jac = _s_jac(phones_a, phones_b)
    cv = _s_cv(phones_a, phones_b)
    vow = _s_vow(phones_a, phones_b)

    agg = (
        weights.phoneme * ph
        + weights.rhyme * rh
        + weights.syllable * sy
        + weights.stress * st
        + weights.jaccard * jac
        + weights.cv_pattern * cv
        + weights.stressed_vowel * vow
    )

    return PhiScores(
        phoneme=ph,
        rhyme=rh,
        syllable=sy,
        stress=st,
        jaccard=jac,
        cv_pattern=cv,
        stressed_vowel=vow,
        aggregate=agg,
    )


# ---------------------------------------------------------------------------
# Line-level and song-level scoring
# ---------------------------------------------------------------------------


def score_phi_line(
    original_line: str,
    modified_line: str,
    weights: PhiWeights,
) -> PhiScores:
    """Score the phonetic similarity between two lyric lines.

    Words are aligned left-to-right. Only pairs where both words have
    CMUdict entries are included in the average. Short function words
    (length <= 2) are included in alignment but scored if in CMUdict.

    Args:
        original_line: Original lyric line.
        modified_line: Modified lyric line.
        weights: Phi weight configuration.

    Returns:
        PhiScores averaged over all scored word pairs in the line.
        Returns zero-scores if no word pairs can be scored.
    """
    orig_words = re.findall(r"[a-zA-Z']+", original_line)
    mod_words = re.findall(r"[a-zA-Z']+", modified_line)

    if not orig_words or not mod_words:
        return PhiScores()

    pairs = list(zip(orig_words, mod_words))
    scored: list[PhiScores] = []

    for wa, wb in pairs:
        pa = _get_phones(wa)
        pb = _get_phones(wb)
        if pa is None or pb is None:
            continue
        scored.append(_score_word_pair(pa, pb, weights))

    if not scored:
        return PhiScores()

    return _average_scores(scored)


def score_phi(
    original_lyrics: str,
    modified_lyrics: str,
    config: LyrkIConfig,
    skip_section_markers: bool = True,
) -> PhiScores:
    """Score the phonetic similarity between two complete lyric texts.

    Computes Phi per line (skipping section markers and empty lines),
    then averages across all lines.

    Args:
        original_lyrics: The full original lyrics text.
        modified_lyrics: The modified/generated lyrics text.
        config: LyrkI configuration (weights extracted from config.phi.weights).
        skip_section_markers: If True, skip lines matching [section] patterns.

    Returns:
        PhiScores averaged over all scored lines in the lyrics.
    """
    weights = config.phi.weights
    orig_lines = original_lyrics.split("\n")
    mod_lines = modified_lyrics.split("\n")

    line_scores: list[PhiScores] = []

    for orig_line, mod_line in zip(orig_lines, mod_lines):
        orig_stripped = orig_line.strip()
        if not orig_stripped:
            continue
        if skip_section_markers and re.match(r"^\[.*\]$", orig_stripped):
            continue

        s = score_phi_line(orig_stripped, mod_line.strip(), weights)
        if s.phoneme > 0 or s.syllable > 0:  # at least one word was scored
            line_scores.append(s)

    if not line_scores:
        return PhiScores()

    return _average_scores(line_scores)


def _average_scores(scores: list[PhiScores]) -> PhiScores:
    """Average a list of PhiScores component-wise."""
    n = len(scores)
    return PhiScores(
        phoneme=sum(s.phoneme for s in scores) / n,
        rhyme=sum(s.rhyme for s in scores) / n,
        syllable=sum(s.syllable for s in scores) / n,
        stress=sum(s.stress for s in scores) / n,
        jaccard=sum(s.jaccard for s in scores) / n,
        cv_pattern=sum(s.cv_pattern for s in scores) / n,
        stressed_vowel=sum(s.stressed_vowel for s in scores) / n,
        aggregate=sum(s.aggregate for s in scores) / n,
    )


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------


def filter_candidates(
    original_lyrics: str,
    candidates: list[str],
    config: LyrkIConfig,
) -> tuple[list[tuple[str, PhiScores]], list[tuple[str, PhiScores]]]:
    """Score candidates and split into accepted and rejected lists.

    Args:
        original_lyrics: The original lyrics text to compare against.
        candidates: List of candidate variation texts from the LLM.
        config: LyrkI configuration (min_aggregate threshold + weights).

    Returns:
        Tuple of (accepted, rejected), where each is a list of
        (candidate_lyrics, phi_scores) pairs. accepted contains all
        candidates with phi_aggregate >= config.phi.min_aggregate.
        Both lists preserve the original candidate order.
    """
    accepted: list[tuple[str, PhiScores]] = []
    rejected: list[tuple[str, PhiScores]] = []

    threshold = config.phi.min_aggregate

    for candidate in candidates:
        scores = score_phi(original_lyrics, candidate, config)
        if scores.aggregate >= threshold:
            accepted.append((candidate, scores))
        else:
            rejected.append((candidate, scores))

    return accepted, rejected
