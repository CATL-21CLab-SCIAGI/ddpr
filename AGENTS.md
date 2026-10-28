# Agent guidance

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for code layout, conventions and checks.
Before introducing a pattern or renaming an API, read the closest counterpart
in the sibling ddsr-bench project; do not infer style from names alone:

- `../ddsr-bench/ddsr_bench/benchmarks/critpt/data/loader.py`
- `../ddsr-bench/ddsr_bench/benchmarks/scicode/data/loader.py`
- `../ddsr-bench/ddsr_bench/benchmarks/cmphysbench/trajectory.py`
- `../ddsr-bench/ddsr_bench/training/schemas.py`

## Maintenance

CritPt-AI grading follows DDSR-Bench's pinned Auto_CritPt_Grader behavior.
Do not change verdicts, fallback rules or failures during integration. DDPR only
maps the shared report to training rewards; upstream scoring fixes need separate
approval and differential tests in DDSR-Bench.

Keep handwritten changes small enough to review (about 200 lines when practical).
Search for equivalent operations and inspect callers before adding helpers;
preserve scoring and failure semantics when consolidating code. Avoid speculative
abstractions and compatibility layers without an actual consumer.

After refactors, remove obsolete imports, modules, references and empty folders,
including directories left behind by bytecode caches. Preserve unrelated work,
private data and run artifacts. Follow CONTRIBUTING.md's checks before committing;
preserve contributor history and commit only when authorized.

Review affected documents in full and check incoming links after moving sections.
Keep shared contracts in the root README, backend behavior in backend guides,
and implementation layout in CONTRIBUTING.md. Update existing guidance instead
of adding documents for each change.

When an agreed rule changes, update its concise repository guidance and align the
shared `ddsr-bench-maintenance` skill when available. Repository instructions must
remain usable without personal skill files; report any blocked skill updates.
