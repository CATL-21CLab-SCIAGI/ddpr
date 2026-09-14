# Coding conventions

ddpr follows the style of the sibling ddsr-bench project. Before introducing a
new pattern, read its closest existing counterpart; do not infer style from
names alone. Useful references are:

- `../ddsr-bench/ddsr_bench/benchmarks/critpt/data/loader.py`
- `../ddsr-bench/ddsr_bench/benchmarks/scicode/data/loader.py`
- `../ddsr-bench/ddsr_bench/benchmarks/cmphysbench/trajectory.py`
- `../ddsr-bench/ddsr_bench/training/schemas.py`

## Structure and interfaces

- Expose domain operations: `load_sample`, `generate_rollout`, `normalize`.
  Use one public entry point for one operation, rather than parallel APIs for
  records and separate fields when a small mapping at the caller suffices.
- Keep construction and validation helpers private, with concrete names such
  as `_messages`, `_spec`, or `_step`. Add a helper when it isolates a meaningful
  piece of work; do not add forwarding layers merely to follow a naming pattern.
- Use small functions and plain frozen, slotted dataclasses. Prefer named
  constructor arguments for records with multiple fields.
- Preserve shared domain vocabulary and exported field names. Document real
  differences between local and upstream schemas rather than inventing names.
- Keep dataset normalization independent of the training backend. Slime owns
  tokenization and training mechanics; ddpr owns the adaptation boundary.
- Implement present requirements. Leave RL extensible through these boundaries
  without placeholder registries, base classes, or speculative configuration.

## Validation and readability

- Validate external data at the boundary. Use `TypeError` for wrong types and
  `ValueError` for invalid values or violated domain constraints.
- Use concrete annotations after normalization; reserve `Any` for external
  records and framework interfaces whose types are not locally available.
- Keep docstrings short and about the operation. Comments should explain a
  non-obvious contract or decision, not narrate individual statements.
- Keep framework imports optional where practical; ordinary data operations
  must not require the full training stack.
- Test observable contracts: preserved context, supervision boundaries, IDs,
  metadata and invalid input. Use small fixtures and public entry points.
- Keep changes reviewable. Do not rename APIs repeatedly based on isolated
  wording preferences; compare the surrounding design with existing code first.

Run pytest, Ruff lint and formatting checks for changes. Upstream mask checks
use `DDPR_SLIME_ROOT` as documented in README.md. Clearly distinguish these
checks from student-tokenizer validation and actual GPU training.
