# Contributing to `lyrkl`

Thanks for contributing. This repo is meant to be a practical research toolbox
for lyrics-related MIR/ML work, so the best contributions usually improve
reusability, clarity, and reproducibility.

## Good contribution areas

- new dataset-building utilities
- new variation generators or controls
- alternate scoring or filtering methods
- docs, examples, and tests
- exporters for downstream research workflows
- bug fixes in lyrics cleaning, artist resolution, or persistence

## General guidelines

- Keep public functions type-annotated
- Use Google-style docstrings
- Prefer small, composable utilities over project-specific glue
- Avoid hardcoding credentials or local paths
- Treat copyrighted lyrics and generated outputs as local data, not repository
  assets

## Research-specific features

Features that are tightly tied to one project are still welcome, but they
should be introduced as optional utilities or clearly labeled experiment
scaffolds rather than core assumptions of the package.

## Paper-derived methods

If you add or modify an implementation based on a paper:

- include a citation in the relevant docstring or README section
- distinguish clearly between the paper's description and this repo's concrete
  implementation choices
- avoid implying that the repo is the official implementation unless it truly is

## Development

Typical local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run tests and checks before opening a PR when possible.

## Scope

The repository aims to stay useful to a broad set of MIR/ML researchers. When
in doubt, prefer abstractions and docs that make a feature easier for someone
outside the original project to understand and reuse.
