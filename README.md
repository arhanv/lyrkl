# lyrkl

Curation toolkit for building **controlled lyric prompt datasets** for music generation model research.

Experiments that probe how music generation models respond to lyric content -- whether they track
phonetic patterns, semantic meaning, or structural form -- require matched sets of lyric variants
that vary along a single axis while holding style constant. lyrkl automates the three steps that
make this tractable: fetching clean lyrics at scale from Genius, generating LLM-driven variants
filtered by a 7-component phonetic similarity metric, and persisting everything in a structured
SQLite database so no API call ever needs to repeat.

The output is a `PromptDataset` JSON compatible with [acelm-interp](../acelm-interp) and adaptable
to any music generation pipeline.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # fill in GENIUS_API_KEY and one LLM key
lyrkl run-all songs.yaml      # fetch lyrics, generate style + variants
lyrkl status                  # check coverage across all pipeline stages
lyrkl export --out data/prompts.json
```

## Environment variables

| Variable | Required for |
|---|---|
| `GENIUS_API_KEY` | Lyrics fetching |
| `ANTHROPIC_API_KEY` | Claude provider |
| `GEMINI_API_KEY` | Gemini provider |
| `OPENROUTER_API_KEY` | OpenRouter provider |

Get a Genius API token at [genius.com/api-clients](https://genius.com/api-clients).

## Pipeline stages

Each stage is independently re-runnable. Re-running is safe: content-addressed IDs and
`INSERT OR IGNORE` ensure no work is duplicated.

```bash
lyrkl add-songs songs.yaml   # register songs (title, artist, genre)
lyrkl fetch                   # pull lyrics from Genius; attach genius_id + raw/clean text
lyrkl style                   # generate LLM style descriptions per song
lyrkl vary                    # generate variants, score with Phi, save accepted + rejected
lyrkl export --out data/p.json
```

## Database and dataset

lyrkl maintains a single SQLite file (`data/lyrkl.db`) with four tables:

| Table | Contents |
|---|---|
| `songs` | Core records: title, artist, genre, raw and cleaned lyrics, Genius ID, optional audio path |
| `style_descriptions` | One or more style descriptions per song (LLM-generated, captioner-generated, or manual) |
| `variations` | Lyric variants with full Phi scores, acceptance status, variant type, and the prompt hash that produced them |
| `llm_responses` | Complete record of every LLM API call: prompt text, response text, model, timestamp |

Raw LLM responses are also written as flat `.txt` files under `data/llm_responses/{song_id}/` for
inspection without querying the database.

### Exported dataset format

`lyrkl export` writes a JSON array of `PromptDataset` records:

```json
{
  "song_id": "eminem__lose_yourself",
  "lyrics": "...",
  "caption": "Aggressive hip-hop with dense rhyme schemes ...",
  "variant": "phonetic",
  "metadata": {
    "title": "Lose Yourself",
    "artist": "Eminem",
    "genre": "hip-hop",
    "phi_aggregate": 0.74,
    "phi_scores": { "phoneme": 0.81, "rhyme": 0.70, "syllable": 0.95, ... }
  }
}
```

The `export` command supports flags for exporting verbatim lyrics (`--variations=verbatim`),
extracting a specific section (`--section verse`), and shuffling line order (`--shuffle-lines`)
for structural control conditions.

## Phi: phonetic similarity filtering

Phonetic variant generation follows the APT (Adversarial PhoneTic Prompting) method introduced by
[Roh et al., 2025](https://arxiv.org/abs/2507.17937). Each candidate variant is scored against the
original on seven dimensions using CMUdict pronunciations: phoneme overlap, rhyme scheme, syllable
count, stress pattern, Jaccard token similarity, consonant-vowel pattern, and stressed vowel
identity. The aggregate Phi score is a weighted sum (weights configurable and auto-normalized in
`configs/default.yaml`).

Variants below the `phi.min_aggregate` threshold are stored as `rejected` rather than discarded,
so lowering the threshold and re-exporting never requires re-calling the LLM.

## Extending the dataset

The pipeline is designed for incremental expansion:

- **Add songs:** append to `songs.yaml` and re-run `lyrkl fetch` + `lyrkl vary`
- **Change the prompt:** edit `prompts/apt_primary.txt`; the next `lyrkl vary` detects the new
  prompt hash and generates fresh variants while keeping old ones
- **Add style from audio:** implement the `CaptioningModel` ABC in `audio.py` to generate style
  descriptions from reference audio files
- **Add variant types:** `VariantType` supports `phonetic`, `verbatim`, `semantic`, `shuffled`,
  `random`, and `style_only`; new generation paths can be added to `pipeline.py`
- **Adjust Phi weights:** per-component weights in `configs/default.yaml` are auto-normalized;
  adjust them to prioritize specific phonetic dimensions

## Configuration

Edit `configs/default.yaml`:

```yaml
llm:
  provider: gemini
  model: gemini-2.0-flash
  candidates_per_song: 3

phi:
  min_aggregate: 0.75
```

## Connection to acelm-interp

```python
from acelm_interp.prompts import PromptDataset
ds = PromptDataset.load("data/prompts.json")
```
