"""Optional checks against real MS-Swift; no model weights or GPU required."""

import json
import os
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ddpr.backends.swift import rl, sft


@pytest.mark.skipif(
    os.environ.get("DDPR_INSTALLED_SWIFT") != "1",
    reason="set DDPR_INSTALLED_SWIFT=1 in an MS-Swift environment",
)
@pytest.mark.parametrize("keep_valid", [True, False])
@pytest.mark.parametrize("task", ["sft", "rl"])
def test_registered_dataset_filters_empty_completions(
    tmp_path, monkeypatch, record, keep_valid, task
):
    from swift.dataset import load_dataset
    from swift.dataset.dataset_meta import DATASET_MAPPING

    records = []
    for content in ("", " \n\t"):
        empty = deepcopy(record)
        empty["completion"][0]["content"] = content
        records.append(empty)
    if keep_valid:
        records.append(record)
    path = tmp_path / "sft.jsonl"
    content = "".join(json.dumps(row) + "\n" for row in records)
    path.write_text(content)
    monkeypatch.setenv("DDPR_SFT_DATA", str(path))
    before = dict(DATASET_MAPPING)
    try:
        plugin = Path(__file__).parents[3] / "ddpr/backends/swift/plugin.py"
        runpy.run_path(str(plugin))
        if keep_valid:
            train, _ = load_dataset(
                [f"ddpr_{task}"],
                split_dataset_ratio=0,
                strict=True,
                remove_unused_columns=False,
            )
            assert len(train) == 1
            row = dict(train[0])
            row.pop("dataset")
            preprocess = sft.preprocess if task == "sft" else rl.preprocess
            assert row == preprocess(record)
        else:
            with pytest.raises(ValueError, match="no usable samples remain"):
                load_dataset([f"ddpr_{task}"], split_dataset_ratio=0, strict=True)
        assert path.read_text() == content
    finally:
        DATASET_MAPPING.clear()
        DATASET_MAPPING.update(before)


@pytest.mark.skipif(
    os.environ.get("DDPR_INSTALLED_SWIFT") != "1",
    reason="set DDPR_INSTALLED_SWIFT=1 in an MS-Swift environment",
)
@pytest.mark.parametrize("use_export", [False, True], ids=["synthetic", "export"])
@pytest.mark.parametrize("task", ["sft", "rl"])
def test_registered_dataset_views(
    tmp_path, monkeypatch, record, request, use_export, task
):
    from swift.dataset import load_dataset
    from swift.dataset.dataset_meta import DATASET_MAPPING

    if use_export:
        path, records = request.getfixturevalue("exported_sft")
    else:
        records = [record]
        path = tmp_path / "sft.jsonl"
        path.write_text(json.dumps(record) + "\n")
    monkeypatch.setenv("DDPR_SFT_DATA", str(path))
    # Restore Swift's global registry after exercising the external plugin.
    before = dict(DATASET_MAPPING)
    try:
        plugin = Path(__file__).parents[3] / "ddpr/backends/swift/plugin.py"
        runpy.run_path(str(plugin))
        train, val = load_dataset(
            [f"ddpr_{task}"],
            split_dataset_ratio=0,
            remove_unused_columns=False,
            strict=True,
            shuffle=False,
        )
        assert val is None
        preprocess = sft.preprocess if task == "sft" else rl.preprocess
        expected = [
            row for record in records if (row := preprocess(record)) is not None
        ]
        assert len(train) == len(expected)
        for loaded, row in zip(train, expected, strict=True):
            assert loaded.pop("dataset") == str(path)
            assert loaded == row
    finally:
        DATASET_MAPPING.clear()
        DATASET_MAPPING.update(before)


@pytest.mark.skipif(
    os.environ.get("DDPR_INSTALLED_SWIFT") != "1",
    reason="set DDPR_INSTALLED_SWIFT=1 in an MS-Swift environment",
)
@pytest.mark.parametrize("use_export", [False, True], ids=["synthetic", "export"])
def test_grpo_reward_columns(record, request, use_export):
    import torch
    from swift.rl_core.data import GRPOSample
    from swift.rl_core.grpo_algorithm import compute_rewards_per_func

    records = request.getfixturevalue("exported_sft")[1] if use_export else [record]
    records = [r for r in records if r["completion"][0]["content"].strip()]
    assert records, "reward check needs a nonempty reference"
    samples = []
    for original in records:
        row = rl.preprocess(original)
        assert row["messages"] == [
            {"role": m["role"], "content": m["content"]} for m in original["prompt"]
        ]
        sample = GRPOSample.from_row(row)
        sample.messages.append({"role": "assistant", "content": "Fresh response"})
        samples.append(sample)

    def reward(completions, sample_id, metadata, reference_completion, **kwargs):
        assert completions == ["Fresh response"] * len(records)
        assert sample_id == [r["id"] for r in records]
        assert metadata == [r["metadata"] for r in records]
        assert reference_completion == [r["completion"][0]["content"] for r in records]
        return [0.5] * len(records)

    scores = compute_rewards_per_func(samples, [reward], None, torch.device("cpu"))
    assert scores.tolist() == [[0.5]] * len(records)


@pytest.mark.skipif(
    os.environ.get("DDPR_SWIFT_CPU_TRAINING") != "1",
    reason="set DDPR_SWIFT_CPU_TRAINING=1 for a tiny CPU optimizer/checkpoint test",
)
@pytest.mark.parametrize(
    "task,use_export",
    [
        ("sft", False),
        ("sft_smoke", False),
        ("rl", False),
        ("rl_smoke", False),
        ("sft", True),
        ("rl", True),
    ],
    ids=["sft", "sft_smoke", "rl", "rl_smoke", "sft-export", "rl-export"],
)
def test_launcher_training(tmp_path, task, use_export, request):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tokenizer_path = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not tokenizer_path:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to local tokenizer files")
    exported = request.getfixturevalue("exported_sft") if use_export else None
    max_length = 8192 if use_export else 1024
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=max_length,
            tie_word_embeddings=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    checkpoint = tmp_path / "model"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    original = model.model.embed_tokens.weight.detach().clone()
    data = tmp_path / "sft.jsonl"
    records = [
        {
            "id": "synthetic:cpu-smoke",
            "prompt": [{"role": "user", "content": "Find kinetic energy."}],
            "completion": [{"role": "assistant", "content": "E = mv²/2."}],
            "metadata": {"synthetic_test": True},
        }
    ]
    if task == "sft":
        # Distinct rows make restarting at the wrong data position observable.
        records.extend(
            {
                "id": f"synthetic:{index}",
                "prompt": [{"role": "user", "content": question}],
                "completion": [{"role": "assistant", "content": answer}],
                "metadata": {"synthetic_test": True},
            }
            for index, (question, answer) in enumerate(
                [("Find momentum.", "p = mv."), ("Find the force.", "F = ma.")]
            )
        )
    if exported is not None:
        data, records = exported
    else:
        data.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "DDPR_MODEL": str(checkpoint),
        "DDPR_SFT_DATA": str(data),
        "DDPR_OUTPUT_DIR": str(output),
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    }
    root = Path(__file__).parents[3]
    extra_args = ["--max_steps", "2"] if task == "sft" else []
    if task in ("rl", "rl_smoke"):
        (tmp_path / "reward_records.json").write_text(
            json.dumps({r["id"]: r for r in records})
        )
        reward_plugin = tmp_path / "reward.py"
        reward_plugin.write_text(
            "import json\n"
            "from pathlib import Path\n"
            "from swift.rewards import ORM, orms\n"
            "records = json.loads(Path(__file__).with_name('reward_records.json').read_text())\n"
            "class SmokeReward(ORM):\n"
            "    def __call__(self, completions, sample_id, metadata, "
            "reference_completion, **kwargs):\n"
            "        assert len(completions) == 2\n"
            "        assert sample_id[0] == sample_id[1]\n"
            "        for key, meta, ref in zip(sample_id, metadata, reference_completion, strict=True):\n"
            "            assert meta == records[key]['metadata']\n"
            "            assert ref == records[key]['completion'][0]['content']\n"
            "        return [0.0, 1.0]\n"
            "orms['ddpr_cpu_smoke'] = SmokeReward\n"
        )
        env["DDPR_REWARD_PLUGIN"] = str(reward_plugin)
        env["DDPR_REWARD_FUNCS"] = "ddpr_cpu_smoke"
        extra_args = [
            "--truncation_strategy",
            "delete",
            "--num_generations",
            "2",
            "--per_device_train_batch_size",
            "2",
            "--max_completion_length",
            "4",
            "--beta",
            "0",
        ]
    command = [
        "bash",
        str(root / f"scripts/train/swift/qwen38_27b_{task}.sh"),
        "--model_type",
        "qwen3",
        "--template",
        "qwen3",
        "--torch_dtype",
        "float32",
        "--use_cpu",
        "true",
        "--device_map",
        "cpu",
        "--attn_impl",
        "eager",
        "--max_length",
        str(max_length),
        "--max_steps",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--save_steps",
        "1",
        "--add_version",
        "false",
        "--dataloader_num_workers",
        "0",
        "--dataloader_pin_memory",
        "false",
        *extra_args,
    ]
    result = subprocess.run(
        command,
        env=env,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    if task.endswith("_smoke"):
        logs = [
            json.loads(line)
            for line in (output / "logging.jsonl").read_text().splitlines()
        ]
        assert logs[-1]["global_step"] == 1
        assert not list(output.glob("checkpoint-*"))
        assert any("loss" in entry for entry in logs)
        return
    saved = output / "checkpoint-1"
    state = json.loads((saved / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    assert any("loss" in entry for entry in state["log_history"])
    weights = load_file(saved / "model.safetensors")
    assert not torch.equal(original, weights["model.embed_tokens.weight"])

    if task == "sft":
        resumed_output = tmp_path / "resumed"
        resumed = subprocess.run(
            [
                *command,
                "--resume_from_checkpoint",
                str(saved),
                "--output_dir",
                str(resumed_output),
            ],
            env=env,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert resumed.returncode == 0, resumed.stdout[-3000:] + resumed.stderr[-6000:]
        resumed_checkpoint = resumed_output / "checkpoint-2"
        resumed_state = json.loads(
            (resumed_checkpoint / "trainer_state.json").read_text()
        )
        assert resumed_state["global_step"] == 2
        actual = load_file(resumed_checkpoint / "model.safetensors")
        expected = load_file(output / "checkpoint-2/model.safetensors")
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        optimizer = torch.load(
            resumed_checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
        )
        assert optimizer["state"]
        assert all(state["step"].item() == 2 for state in optimizer["state"].values())
