# lyrkl

`lyrkl` is a toolbox for lyrics-centered MIR/ML experiments.

Pull lyrics from Genius from a song list or artist discography, filter out common dataset confounds like remixes, covers, and featuring credits, and persist everything in a reusable SQLite database or JSON dataset. The package also includes a lightweight framework for generating systematic lyric variations — LLM-based and deterministic — along with built-in implementations we developed for our own research. Contributions of new variation tools, scoring functions, and data sources are welcome.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `GENIUS_API_KEY` in `.env` to fetch lyrics. Set one LLM key only if you want style generation or lyric variation:

- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`

Get a Genius API token at [genius.com/api-clients](https://genius.com/api-clients).

## Core workflow

Register songs from YAML, fetch lyrics, inspect status, and export a dataset:

```bash
lyrkl add-songs songs.example.yaml
lyrkl fetch
lyrkl status
lyrkl export --out data/prompts.json
```

Or resolve songs directly from one or more artists:

```bash
lyrkl resolve-artists "Taylor Swift" "Drake" --max-songs 20
lyrkl fetch
```

If you want the built-in LLM features:

```bash
lyrkl style
lyrkl vary
```

Each stage is incremental and safe to rerun.

## Lyrics retrieval and dataset building

`lyrkl` fetches lyrics from Genius and does the cleanup work that otherwise gets repeated across projects:

- **Normalization:** section markers (`[Verse 1]`, `[Chorus: Drake]`) are standardized; Genius-specific artifacts like contributor footers and "You might also like" banners are stripped.
- **Artist resolution:** `resolve-artists` pulls an artist's top songs from Genius up to a configurable limit and registers them automatically.
- **Confound filtering:** when building datasets from artist discographies, `lyrkl` can filter out remixes, alternate versions, covers, tribute tracks, collaborations, featured appearances, and songs outside a year range.
- **Duplicate detection:** `check-duplicates` flags near-duplicate lyrics by Jaccard similarity on word bags.

Example song list YAML (`songs.example.yaml`):

```yaml
songs:
  - title: Bohemian Rhapsody
    artist: Queen
    genre: rock
  - title: Lose Yourself
    artist: Eminem
    genre: hip-hop
```

Example artist list YAML:

```yaml
- name: Radiohead
  genre: alt-rock
  max_songs: 10
- name: Kendrick Lamar
  genre: hip-hop
  max_songs: 10
```

## Phonetic similarity

`lyrkl` includes a seven-component phonetic similarity metric (Phi) in [`src/lyrkl/phi.py`](src/lyrkl/phi.py), implemented using CMUdict-based features:

| Component | What it measures |
|-----------|-----------------|
| S_ph | Phoneme sequence similarity (SequenceMatcher) |
| S_rh | Rhyme overlap (terminal phoneme match) |
| S_sy | Syllable count ratio |
| S_st | Stress pattern alignment |
| S_jac | Phoneme-set Jaccard similarity |
| S_cv | Consonant-vowel skeleton pattern |
| S_vow | Stressed-vowel core alignment |

Weights and the aggregate threshold are configurable. This implementation follows the feature family described in Roh et al. (2025); see [Citation](#citation).

```python
from lyrkl.phi import score_phi, filter_candidates
```

## Variation generation

The built-in pipeline supports:

- **LLM-based phonetic variations** via the APT prompting strategy
- **Verbatim baselines** for control experiments
- **Section-only exports** (e.g. chorus only)
- **Deterministic line-shuffled controls** (preserves phonetics, destroys semantics)

These are the methods we built for our own research. The package is intended to be extended with new variation generators, scoring functions, and export paths.

## Database and pipeline

`lyrkl` uses a single SQLite file (usually `data/lyrkl.db`) as the source of truth for all pipeline artifacts.

Main tables:

- `songs` — song metadata and cleaned lyrics
- `style_descriptions` — LLM, captioner, template, or manual style text
- `variations` — generated lyric variants and Phi scores
- `llm_responses` — raw prompts and model outputs

Every LLM call is keyed by a SHA-256 hash of the filled prompt text and model name. Re-running with the same config never re-calls the LLM; changing config automatically triggers new calls.

## Configuration

Edit `configs/default.yaml` to choose the LLM provider, prompt templates, and Phi threshold:

```yaml
llm:
  provider: claude
  model: claude-haiku-4-5-20251001
  candidates_per_song: 5

phi:
  min_aggregate: 0.70
```

API keys come from environment variables or `.env`, not from committed config files.

## Extending lyrkl

Useful extension points:

- new lyrics sources or metadata enrichers
- new variation generators
- alternate phonetic or structural scoring functions
- audio captioning backends via `CaptioningModel` in `src/lyrkl/audio.py`
- new export formats

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

- `src/lyrkl/` — installable Python package
- `configs/` — user-editable YAML configuration
- `prompts/` — LLM prompt templates (examples and defaults)
- `utils/` — small research utilities and export helpers
- `experiments/` — project-specific scripts and one-off analyses
- `notebooks/` — exploratory notebooks
- `data/` — local database, exports, and raw responses; gitignored

## Citation

The phonetic feature family in `phi.py` is based on the APT method described in:

> Roh et al. (2025). *Bob's Confetti*. arXiv:2507.17937.

Phonetic lookups use the CMU Pronouncing Dictionary via the [`pronouncing`](https://pypi.org/project/pronouncing/) library.

`lyrkl` is an independent implementation; it is not the official code release for that paper.

## Data and copyright

This repository is code, not a lyrics dataset. Fetched lyrics, LLM responses, and exported lyric text are local research artifacts and should stay out of version control. The `data/` directory is gitignored for this reason.
