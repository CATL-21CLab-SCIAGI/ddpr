# ddpr

Training for data-driven physics reasoning.

ddpr adapts ddsr-bench's `sft.jsonl` in memory for supervised fine-tuning (SFT)
with Slime, preserving context and supervising only the completion. This path
has been validated with a Qwen3.8-27B optimizer update on DSW.

## Installation

Use Python 3.12 or newer. From this project's root:

```bash
python -m pip install -e .
```

Install Slime and its model/backend dependencies separately using its own
instructions. ddpr does not depend on ddsr-bench's Python package, and does
not install the similarly named PyPI package `slime`.

See the [Slime SFT guide](ddpr/backends/slime/README.md) for training setup.

## Run the SFT smoke job

This example assumes the configured DSW, SSH alias and smoke data already
exist. The data is not included in Git. Connect from your local terminal:

```bash
ssh dsw-0n3jist2qzjxzgaxqc
```

Install ddpr in the existing DSW environment, then run:

```bash
cd /mnt/workspace/Projects/ddpr
python -m pip install --no-deps --no-build-isolation -e .
export DDPR_SFT_DATA="$PWD/data/sft-smoke/cmphysbench-aliyun-20260914/export/sft.jsonl"
bash configs/jobs/slime/qwen38_27b_smoke.sh
```

This runs one supervised Adam update on eight samples using four GPUs and
the mounted `Qwen/Qwen3.8-27B` model. It prints training metrics to the terminal
and does not save a checkpoint. No Conda or Mamba activation is needed on this
DSW image; repeat the installation after an instance restart if needed.

To retain the terminal output, run the command in Bash with:

```bash
set -o pipefail
bash configs/jobs/slime/qwen38_27b_smoke.sh 2>&1 | tee "data/sft-smoke/sft-$(date +%Y%m%d-%H%M%S).log"
```

`DDPR_MODEL` and `DDPR_SLIME_ROOT` override the checkpoint and Slime paths;
the launcher still assumes the Qwen3.8-27B architecture and `/root/Megatron-LM`.
For another export, set `DDPR_SFT_DATA` to its path. This is a fixed smoke job; see the
[Slime SFT guide](ddpr/backends/slime/README.md#one-step-qwen38-smoke-test)
for the tested configuration and results.

## Future work

Planned extensions, not yet implemented:

- **Longer SFT jobs:** configurable sequence lengths, batches and duration,
  with checkpoint saving and resumption.
- **Reinforcement learning (RL) with Slime:** generate fresh responses from
  exported prompts and score them with physics verification and rewards.
  Teacher completions are not automatically verified references.
- **Additional backends such as Miles:** reuse shared data adapters and add
  backend-specific launch configurations and integration tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code layout,
conventions and checks.
