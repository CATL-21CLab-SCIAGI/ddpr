# Contributing

## Development setup

Use Python 3.12 or newer and install from the project root:

```bash
python -m pip install -e '.[dev]'
```

For local CPU development with Swift and Slime's data utilities:

```bash
mamba env create -f environment.yml
mamba activate ddpr
python scripts/setup_dev.py
python scripts/check_environment.py dev
```

The setup installs both frameworks at pinned revisions in the active environment.
Slime includes only dependencies needed for data utilities; its full CLI is
unavailable here. The checker reports those omitted training dependencies.
See [environment profiles](configs/environments/README.md) for the Mac dependency
snapshot, clean rebuilds, other development platforms and separate GPU setups.
See the [Slime guide](ddpr/backends/slime/README.md) and
[Swift guide](ddpr/backends/swift/README.md) for backend requirements.
Prefer ModelScope for model artifacts; tokenizer checks do not require weights.

Run the local backend checks with an existing Qwen3.8 tokenizer directory:

```bash
DDPR_INSTALLED_SWIFT=1 \
DDPR_QWEN38_TOKENIZER=/path/to/local/tokenizer python -m pytest -q
```

Leave `DDPR_INSTALLED_SLIME` unset locally; it enables full Slime CLI checks.
Set `DDPR_SWIFT_CPU_TRAINING=1` to also run Swift's tiny SFT and GRPO jobs.

## Code layout

- `ddpr/data/`: framework-independent schemas and export normalization.
- `ddpr/backends/`: adapters for each training backend.
- `configs/jobs/`: launch configurations.
- `tests/`: data and backend contract checks.

Use the same conventions within each backend:

- `sft.py` and `rl.py`: task adapters, only for implemented tasks.
- `configs/jobs/<backend>/sft.sh` and `rl.sh`: sourceable argument fragments.
- `<model>_<task>.sh`: standalone launchers; append `_smoke` for smoke jobs.
- `test_sft.py` and `test_rl.py`: task behavior; `test_masks.py`: token supervision;
  `test_integration.py`: installed-framework interfaces and opt-in training
  checks.
- Backend README sections: input and supervision, launch integration,
  model-specific commands, verification and tested revisions.

Add shared data helpers and framework registration files only where needed.
Both use `plugin.py` for framework integration: Swift registers datasets and
Slime provides the shared SFT/RL data-source hook. SFT also uses Slime's direct
rollout hook to prepare supervised tokens.
Do not add empty counterparts for unsupported tasks or interfaces.

Before adding a backend feature, compare both directory trees and README
headings together. Check SFT, SFT smoke and RL launchers, update every command
and link affected by a rename, and keep differences limited to framework behavior.

Keep data normalization independent of backend imports. Backends own training
and optimization; ddpr adapts the exported data. Implement current requirements
without speculative registries or placeholder RL abstractions.
Import framework dependencies directly in backend modules that use them;
keep `ddpr/data` independent of training frameworks.

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
Slime backend tests use the installed package; full CLI checks remain opt-in.
See [verification](ddpr/backends/slime/README.md#verification).
Report local tests, tokenizer checks and GPU training separately. Keep changes
focused and describe the behavior changed, validation performed and limitations.

## Local and DSW files

Keep code and small run artifacts at matching relative paths locally and on
DSW. Preserve exports and trial logs, and verify hashes after transfer. The
`data/` directory is excluded from Git; environments, weights and caches remain
machine-specific. Do not delete unrelated destination files during sync.
