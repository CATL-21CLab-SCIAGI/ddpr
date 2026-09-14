# ddpr

Training for data-driven physics reasoning.

ddpr adapts exported scientific examples to training frameworks. Its first
integration uses ddsr-bench's existing `sft.jsonl` directly for offline SFT in
Slime. Conversion happens in memory; no intermediate dataset is written and
no teacher or policy answers are generated during batch preparation.

## Installation

Use Python 3.12 or newer. From this project's root:

```bash
python -m pip install -e '.[dev]'
```

Install Slime and its model/backend dependencies separately using its own
instructions. ddpr does not depend on ddsr-bench's Python package, and does
not install the similarly named PyPI package `slime`. Slime imports occur
only when its SFT adapter initializes a tokenizer.

See the [Slime SFT guide](ddpr/backends/slime/README.md) for training setup.

## Structure

```text
ddpr/data/schemas.py               framework-independent example
ddpr/data/adapters/ddsr_bench.py   exported prompt/completion contract
ddpr/backends/slime/sft.py         tokenization and completion-only loss masks
configs/jobs/slime/sft.sh          sourceable SFT arguments
tests/                            data and backend contract checks
```

The data adapter preserves separate context and completion fields. The Slime
adapter owns message loss masks and training sample fields. Future RL adapters
can use the same input boundary to obtain prompts, but must implement fresh
policy generation and an explicit reward contract. A teacher completion is
not automatically a verified physics reference. RL is not implemented here.

## Verification

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Optional upstream mask checks require `transformers`, `tokenizers`, and a
Slime source checkout:

```bash
DDPR_SLIME_ROOT=/path/to/slime python -m pytest -q
```

These checks execute Slime's real masking implementation with a synthetic
in-memory chat tokenizer. They do not validate a student checkpoint or run
Ray, Megatron, GPU training, or an optimizer update.
