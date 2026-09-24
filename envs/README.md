# Environments

Run commands from the project root on a DSW instance or DLC job using the
matching custom image.

## Images by GPU

These custom images provide the runtime for Alibaba Cloud PAI DSW instances
and DLC jobs. Select the image for the backend and GPU below.
Obtain image access and private registry details from the project maintainer.

| Backend | GPU | Built tag | Verification |
| --- | --- | --- | --- |
| Slime | H20 | `slime-cu129-codex-20260924` | Selected-layer SFT/RL updates, RL resume and selected rollout-weight checks passed |
| Slime | B300 | `slime-cu130-codex-20260923` | Built candidate; B300 validation pending |
| Swift | H20 | `ms-swift-cu130-codex-20260923` | LoRA SFT/RL updates passed, vLLM disabled |
| Swift | B300 | `ms-swift-cu130-codex-20260923` | Candidate; B300 validation pending |

H20 results cover the limited training configurations above. B300 validation
and GPU reruns of the refactored smoke wrappers remain pending.

## DSW and DLC setup

- **DSW:** select the custom image, start the instance, then use its terminal
  or SSH for development and interactive training.
- **DLC:** select the custom image in the job configuration, make the repository,
  model and data available, and set the training launcher as the job command.
  Match the job's GPU allocation and distributed settings to the launcher.

Use persistent storage for data, logs and checkpoints. The images provide the
backend and CUDA dependencies; no additional Conda/Mamba environment or manual
Docker launch is needed. The recorded GPU checks ran on DSW; DLC job execution
has not yet been verified.

## Install ddpr

From the repository root, run the matching command in the DSW terminal or
as part of the DLC job command before training:

```bash
python scripts/envs/setup.py slime
# Or, in a Swift container:
python scripts/envs/setup.py swift
```

GPU setup installs only ddpr, preserving the container's framework and CUDA
dependencies. All setup profiles finish by checking the environment. To repeat
the check without installing, run `python scripts/envs/check.py <profile>`,
where `<profile>` is `slime` or `swift`.

A passing environment check still needs a training smoke test. See the
[Slime](../ddpr/backends/slime/README.md#verification) and
[Swift](../ddpr/backends/swift/README.md#verification) guides for commands and
validation results. The tested Swift image runs the LoRA training checks but
still fails strict dependency and Git-provenance checks; see its guide.

## Framework revisions

[frameworks.json](frameworks.json) records:

- `gpu_revision`: added only after testing that commit in the target GPU
  container, including an actual training update. Swift's tested image does not
  expose its Git commit, so its revision remains unrecorded.

The recorded Slime GPU revision targets the tested H20 CUDA 12.9 image.
GPU checks require a matching revision when recorded. Otherwise, they report
validation as pending and continue runtime checks. They never record revisions
automatically. Keep image provenance in the internal test records; a matching
framework commit alone does not validate another container.
