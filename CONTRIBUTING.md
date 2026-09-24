# Contributing

## Development setup

Use Python 3.12 or newer and install from the project root:

```bash
python -m pip install -e '.[dev]'
```

The CPU environment is for local debugging and tests only. With Mamba and Git
installed, set up Swift and Slime's data utilities:

```bash
mamba env create -f envs/cpu/env.yml
mamba activate ddpr
python scripts/envs/setup.py cpu
python scripts/envs/check.py cpu
```

The setup installs both frameworks at pinned revisions in the active environment.
Slime includes only dependencies needed for data utilities; its full CLI is
unavailable here. The checker reports those omitted training dependencies.
The Python 3.12 setup and [dependency snapshot](envs/cpu/requirements-macos-arm64-py312.txt)
were tested on macOS arm64. On another CPU platform, use
`python scripts/envs/setup.py cpu --resolve` inside the activated environment,
then run the checks below. `cpu_revision` in [frameworks.json](envs/frameworks.json)
pins the local-debug framework sources. For GPU training, see
[DSW/DLC environments](envs/README.md).
Inside a ready Slime or Swift GPU container, use `python scripts/envs/setup.py slime`
or `python scripts/envs/setup.py swift`; these preserve the container's dependencies.
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

To check an actual ddsr-bench export, set `DDPR_TEST_SFT_DATA` to its path and
`DDPR_TEST_SFT_LIMIT` to the number of leading records (default three):

```bash
DDPR_TEST_SFT_DATA=/path/to/sft.jsonl DDPR_TEST_SFT_LIMIT=3 \
DDPR_INSTALLED_SWIFT=1 DDPR_SWIFT_CPU_TRAINING=1 \
DDPR_QWEN38_TOKENIZER=/path/to/local/tokenizer \
python -m pytest -q tests/backends -k export
```

These checks verify IDs, metadata, SFT supervision and RL prompt/reference
separation, reward fields and Slime cursor restoration. Swift CPU tests also
verify SFT updates and checkpoint resume, and an RL update with a synthetic
reward. Selected records are copied intact to temporary test files; originals
are unchanged; model templates apply their own whitespace normalization.
Both tasks skip empty completion text and report filtering counts. Samples must
fit the tests' 8192-token limit; no content is truncated. This is a tiny-model
integration check, not Qwen3.8-27B GPU training.

## Code layout

- `ddpr/data/`: framework-independent schemas and export normalization.
- `ddpr/rewards/`: shared reward contract, loader and reward implementations.
- `ddpr/backends/`: adapters for each training backend.
- `envs/`: environment definitions, dependency pins and setup guidance.
- `scripts/envs/`: environment installation and validation commands.
- `scripts/train/`: backend launchers and sourceable argument fragments.
- `tests/`: data and backend contract checks.

Use the same conventions within each backend:

- `sft.py` and `rl.py`: task adapters, only for implemented tasks.
- `scripts/train/<backend>/sft.sh` and `rl.sh`: sourceable argument fragments.
- `<model>_<task>.sh`: standalone launchers; append `_smoke` for smoke jobs.
  Smoke scripts wrap the regular launcher and override only workload and checks.
  Keep container requirements and hardware validation in the backend README.
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

Before stopping or replacing a DSW instance, reconcile local and remote changes
in both directions and verify content hashes for code, documentation, and small
run artifacts. Preserve conflicting edits before resolving them; do not stop
until verification completes. Keep large checkpoints on persistent remote
storage and record their paths locally.
