# Environments

Use separate environments for CPU development, Slime GPU training and Swift GPU
training. The development package versions must not be applied to an existing
GPU image: they can replace its CUDA-compatible libraries.

| Profile | Setup | Validation status |
| --- | --- | --- |
| Development | Pinned Python, framework revisions and Mac package snapshot | Clean rebuild passed on macOS arm64 / Python 3.12.14 |
| Slime GPU | Compatible upstream Slime image or configured DSW | Earlier DSW SFT smoke passed; current GPU stack/rebuild remains unverified |
| Swift GPU | Separate compatible upstream Swift environment | CPU integration passed; GPU stack/rebuild remains unverified |

## Development

From the project root, with Mamba and Git installed:

```bash
mamba env create -f environment.yml
mamba activate ddpr
python scripts/setup_dev.py
```

`environment.yml` installs only Python and pip. The script fetches exact revisions
from [frameworks.json](frameworks.json), installs dependencies, then installs
both frameworks and ddpr editably. Framework checkouts live under the active
environment's `src/ddpr/` directory. Existing checkouts must be clean and match
the requested revision; the script does not reset them.

On macOS arm64, [the package snapshot](dev-macos-arm64-py312.txt) pins direct and
transitive runtime dependencies. This is a version snapshot, not a hash-locked
artifact or a Conda build lock. Build tools, package indexes and system libraries
can still affect installation. Slime's full GPU dependencies are intentionally
omitted; actual data utilities are imported during verification.

For another CPU development platform, create a fresh Python 3.12 environment
and explicitly resolve dependencies:

```bash
python scripts/setup_dev.py --resolve
```

This uses [dev.txt](dev.txt) and the pinned Swift source's requirements. It is an
unverified platform until the checks below pass. Do not copy the Mac snapshot
to declare another platform supported.

```bash
python scripts/check_environment.py dev
DDPR_INSTALLED_SWIFT=1 DDPR_SWIFT_CPU_TRAINING=1 \
DDPR_QWEN38_TOKENIZER=/path/to/local/tokenizer python -m pytest -q
```

The checker prints JSON and exits nonzero on missing imports, unknown framework
revisions or dependency conflicts outside Slime's intentionally partial dev
installation. It reports omitted Slime requirements. The tests additionally
check real tokenizer masks, tiny CPU updates and checkpoint resume. Prefer
ModelScope when downloading tokenizer files; no model weights are needed.

To test a rebuild without changing an existing environment:

```bash
mamba env create -f environment.yml -p "$PWD/.cache/envs/dev-rebuild"
mamba activate "$PWD/.cache/envs/dev-rebuild"
python scripts/setup_dev.py
# Run the verification commands above in this environment.
```

On 2026-09-23, a clean rebuild matched all 152 package pins and passed 91 tests,
including Swift CPU SFT/RL and checkpoint resume. Two full Slime CLI tests were
skipped because this development profile omits the GPU stack.

## Slime GPU

Start from a Slime image compatible with the target GPU and driver, or reuse the
configured DSW image. Follow the [upstream build instructions at our reviewed
revision](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/docker/README.md).
Keep its PyTorch, Megatron, SGLang and compiled libraries together. Record the
image's immutable digest when validating a new host; no portable GPU image has
yet been validated by ddpr.

Inside that environment, from the ddpr root:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_environment.py slime
```

The check requires Linux/CUDA, runs a small CUDA operation, checks dependencies
and imports the Slime CLI, Megatron and Transformer Engine. It accepts the local
reviewed revision and the previously tested DSW revision listed in
`frameworks.json`. A passing check does not establish model support, memory
capacity or distributed training correctness. Follow it with the
[Slime interface and smoke checks](../../ddpr/backends/slime/README.md#verification).

## Swift GPU

Use a separate Linux/CUDA environment with a compatible PyTorch build, following
[Swift's installation instructions at our reviewed revision](https://github.com/modelscope/ms-swift/tree/ae700468052af1e461857b6f41eee65aa1641f45#installation).
Install the pinned Swift revision while preserving the chosen CUDA stack, then
install ddpr:

```bash
python -m pip install --no-deps --no-build-isolation 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@ae700468052af1e461857b6f41eee65aa1641f45'
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_environment.py swift
```

The base environment must already contain Swift's dependencies. `--no-deps`
preserves that stack; resolve reported incompatibilities before proceeding.
The check requires Linux/CUDA, runs a small CUDA operation and checks framework
provenance, dependencies and dataset/template/reward imports. Follow it with the
[Swift interface and smoke checks](../../ddpr/backends/swift/README.md#verification).
Pin the resulting environment or image digest only after these checks and an
actual GPU update succeed. Co-installation with Slime's full GPU stack is not
validated.
