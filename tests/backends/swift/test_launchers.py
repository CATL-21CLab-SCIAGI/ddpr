"""Check smoke wrappers inherit the regular Swift launch configuration."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("task", ["sft", "rl"])
def test_smoke_inherits_training_configuration(tmp_path, task):
    swift = tmp_path / "swift"
    swift.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    swift.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DDPR_")}
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        DDPR_SFT_DATA=str(tmp_path / "sft.jsonl"),
        DDPR_OUTPUT_DIR=str(tmp_path / "output"),
        DDPR_REWARD_FUNCS="test_reward",
        DDPR_REWARD="ddpr.rewards.SmokeReward",
    )
    arguments = []
    for suffix in ("", "_smoke"):
        result = subprocess.run(
            [
                "bash",
                str(ROOT / f"scripts/train/swift/qwen38_27b_{task}{suffix}.sh"),
                "--tuner_type",
                "lora",
                "--learning_rate",
                "2e-5",
            ],
            env=env,
            cwd="/",
            check=True,
            capture_output=True,
            text=True,
        )
        arguments.append(result.stdout.splitlines())
    regular, smoke = arguments
    assert smoke[: len(regular)] == regular
    expected = ["--max_steps", "1", "--max_length", "2048"]
    if task == "rl":
        expected += [
            "--max_completion_length",
            "16",
            "--num_generations",
            "2",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--overlong_filter",
            "false",
        ]
    expected += ["--save_strategy", "no"]
    assert smoke[len(regular) :] == expected
