"""Check recomputation arguments without starting training workers."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def launcher_env(tmp_path):
    models = tmp_path / "scripts" / "models"
    models.mkdir(parents=True)
    (models / "qwen3.5-27B.sh").write_text("MODEL_ARGS=(--num-layers 64)\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DDPR_")}
    env.update(
        PATH=f"{binary}:{env['PATH']}",
        DDPR_SLIME_ROOT=str(tmp_path),
        DDPR_SFT_DATA=str(tmp_path / "sft.jsonl"),
        DDPR_OUTPUT_DIR=str(tmp_path / "checkpoints"),
        DDPR_LOAD=str(tmp_path / "saved-checkpoint"),
        DDPR_ACTOR_GPUS="2",
        DDPR_ROLLOUT_GPUS="2",
        DDPR_ROLLOUT_GPUS_PER_ENGINE="2",
        DDPR_NUM_ROLLOUT="1",
        DDPR_REWARD_FUNCTION="rewards.score",
    )
    return env


@pytest.mark.parametrize("task", ["sft", "rl"])
@pytest.mark.parametrize("granularity", [None, "selective"])
def test_recomputation_arguments(launcher_env, task, granularity):
    env = launcher_env
    if granularity is not None:
        env["DDPR_RECOMPUTE_GRANULARITY"] = granularity
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/train/slime" / f"qwen38_27b_{task}.sh"),
            "--override-opt_param-scheduler",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--load") + 1] == env["DDPR_LOAD"]
    assert args[-1] == "--override-opt_param-scheduler"
    assert args[args.index("--recompute-granularity") + 1] == (granularity or "full")
    if granularity == "selective":
        assert "--recompute-method" not in args
        assert "--recompute-num-layers" not in args
    else:
        assert args[args.index("--recompute-method") + 1] == "uniform"
        assert args[args.index("--recompute-num-layers") + 1] == "1"


def _arguments(env, task, smoke=False):
    suffix = "_smoke" if smoke else ""
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/train/slime" / f"qwen38_27b_{task}{suffix}.sh"),
            "--lr",
            "2e-5",
            "--only-train-params-name-list",
            "decoder.layers.31.",
            "--num-rollout",
            "9",
            "--override-opt_param-scheduler",
        ],
        env=env,
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    )
    # Slime's argparse uses the last occurrence for these scalar options.
    arguments = {}
    key = None
    for value in result.stdout.splitlines()[1:]:
        if value.startswith("--"):
            key = value
            arguments[key] = []
        elif key is not None:
            arguments[key].append(value)
    return arguments


@pytest.mark.parametrize("task", ["sft", "rl"])
@pytest.mark.parametrize("granularity", ["full", "selective"])
def test_smoke_inherits_training_configuration(launcher_env, task, granularity):
    launcher_env["DDPR_RECOMPUTE_GRANULARITY"] = granularity
    regular = _arguments(launcher_env, task)
    smoke = _arguments(launcher_env, task, smoke=True)
    expected = {
        "--num-rollout": ["1"],
        "--rollout-batch-size": ["2" if task == "sft" else "1"],
        "--global-batch-size": ["2"],
        "--micro-batch-size": ["1"],
        "--seq-length": ["2048"],
        "--save-interval": ["1"],
        "--custom-megatron-before-train-step-hook-path": ["smoke_checks.before_step"],
    }
    if task == "rl":
        expected.update(
            {
                "--n-samples-per-prompt": ["2"],
                "--rollout-max-prompt-len": ["1536"],
                "--rollout-max-response-len": ["16"],
                "--custom-rm-path": ["ddpr.backends.slime.reward.reward"],
                "--group-rm": [],
                "--ddpr-reward": ["ddpr.rewards.SmokeReward"],
            }
        )
    assert smoke == regular | expected
    assert smoke["--lr"] == ["2e-5"]
    assert smoke["--recompute-granularity"] == [granularity]
    assert smoke["--num-gpus-per-node"] == (["2"] if task == "sft" else ["4"])
