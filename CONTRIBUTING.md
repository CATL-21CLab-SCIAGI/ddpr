# Contributing

## Development setup

Use Python 3.12 or newer and install from the project root:

```bash
python -m pip install -e '.[dev]'
```

Install training backends separately. See the [Slime guide](ddpr/backends/slime/README.md)
for its environment and integration requirements. Prefer ModelScope for model
artifacts; tokenizer checks do not require weights.

## Code layout

- `ddpr/data/`: framework-independent schemas and export normalization.
- `ddpr/backends/`: adapters for each training backend.
- `configs/jobs/`: launch configurations.
- `tests/`: data and backend contract checks.

Keep data normalization independent of backend imports. Backends own training
and optimization; ddpr adapts the exported data. Implement current requirements
without speculative registries or placeholder RL abstractions.

## Coding conventions

Follow the sibling ddsr-bench project's style and compare related code before
introducing patterns or renaming APIs.

- Name public functions for domain operations, such as `load_sample` and
  `generate_rollout`. Use one entry point per operation.
- Keep construction helpers private, such as `_messages`. Extract meaningful
  work rather than forwarding wrappers.
- Use small functions, frozen slotted dataclasses, and named constructor args.
  Preserve shared field names and explain schema differences.
- Validate external input: `TypeError` for wrong types, `ValueError` for invalid
  values. Use concrete types after normalization; reserve `Any` for boundaries.
- Keep docstrings concise and comments focused on non-obvious decisions.
- Test observable behavior: context, supervision, IDs, metadata and invalid
  input. Use small fixtures and public entry points.

## Checks and changes

For code changes, run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

For documentation-only changes, verify links and command examples.
Slime-dependent checks are opt-in; see [verification](ddpr/backends/slime/README.md#verification).
Report local tests, tokenizer checks and GPU training separately. Keep changes
focused and describe the behavior changed, validation performed and limitations.

## Local and DSW files

Keep code and small run artifacts at matching relative paths locally and on
DSW. Preserve exports and trial logs, and verify hashes after transfer. The
`data/` directory is excluded from Git; environments, weights and caches remain
machine-specific. Do not delete unrelated destination files during sync.
