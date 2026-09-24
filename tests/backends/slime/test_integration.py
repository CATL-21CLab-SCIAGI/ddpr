"""Exercise the installed Slime loader and ddpr adapter without model weights."""

import asyncio
import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("slime")

from ddpr.backends.slime.sft import generate_rollout


def test_selective_recomputation_with_frozen_prefix():
    if os.environ.get("DDPR_INSTALLED_SLIME") != "1":
        pytest.skip("requires the Slime GPU container")
    import torch
    from megatron.core.tensor_parallel.random import checkpoint

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    prefix = torch.nn.Linear(4, 4, device="cuda").requires_grad_(False)
    projection = torch.nn.Linear(4, 4, device="cuda")
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)
    inputs = prefix(torch.ones(2, 4, device="cuda"))
    assert not inputs.requires_grad
    before = projection.weight.detach().clone()
    # Selective recomputation starts after the trainable projection, so the
    # checkpoint has a gradient-enabled input even with a frozen prefix.
    output = checkpoint(torch.sin, False, projection(inputs))
    output.square().sum().backward()
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad) > 0
    optimizer.step()
    assert not torch.equal(before, projection.weight)
    assert all(parameter.grad is None for parameter in prefix.parameters())


def test_exported_rl_batch_and_rewards(exported_sft, args, monkeypatch, tmp_path):
    from slime.rollout.rm_hub import batched_async_rm
    from slime.utils.types import Sample
    from transformers import AutoTokenizer

    from ddpr.backends.slime import rl
    from ddpr.backends.slime.plugin import RolloutDataSource

    checkpoint = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not checkpoint:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to local tokenizer files")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    monkeypatch.setattr(rl, "load_tokenizer", lambda *a, **kw: tokenizer)
    path, records = exported_sft
    args.prompt_data = str(path)
    args.rollout_shuffle = False
    args.rollout_max_prompt_len = 8192
    source = RolloutDataSource(args)
    records = [r for r in records if r["completion"][0]["content"].strip()]
    groups = source.get_samples(len(records))
    for group, record in zip(groups, records, strict=True):
        expected = tokenizer.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True
        )
        assert len(group) == args.n_samples_per_prompt
        for sample in group:
            assert sample.prompt == expected
            assert sample.label is None
            assert sample.sample_id == record["id"]
            assert sample.reference_completion == record["completion"][0]["content"]
            assert sample.metadata == record["metadata"]
            assert Sample.from_dict(sample.to_dict()).to_dict() == sample.to_dict()
            sample.response = "Fresh response"
    (tmp_path / "reward_records.json").write_text(
        json.dumps({r["id"]: r for r in records})
    )
    (tmp_path / "export_reward.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "records = json.loads(Path(__file__).with_name('reward_records.json').read_text())\n"
        "async def score(args, samples, **kwargs):\n"
        "    for sample in samples:\n"
        "        record = records[sample.sample_id]\n"
        "        assert sample.metadata == record['metadata']\n"
        "        assert sample.reference_completion == record['completion'][0]['content']\n"
        "        assert sample.response == 'Fresh response'\n"
        "        assert sample.label is None\n"
        "    return [0.5] * len(samples)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args.custom_rm_path = "export_reward.score"
    samples = [sample for group in groups for sample in group]
    assert asyncio.run(batched_async_rm(args, samples)) == [0.5] * len(samples)
    source.save(0)
    restored = RolloutDataSource(args)
    restored.load(0)
    assert [[s.to_dict() for s in g] for g in restored.get_samples(2)] == [
        [s.to_dict() for s in g] for g in source.get_samples(2)
    ]


def test_exported_sft_batch(exported_sft, args):
    from transformers import AutoTokenizer

    from ddpr.backends.slime.plugin import RolloutDataSource

    checkpoint = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not checkpoint:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to local tokenizer files")
    path, records = exported_sft
    args.prompt_data = str(path)
    args.hf_checkpoint = checkpoint
    args.loss_type = "sft_loss"
    args.n_samples_per_prompt = 1
    args.compute_advantages_and_returns = False
    args.apply_chat_template = False
    args.loss_mask_type = "qwen3_5"
    args.rollout_shuffle = False
    source = RolloutDataSource(args)
    records = [r for r in records if r["completion"][0]["content"].strip()]
    args.rollout_batch_size = len(records)
    args.seq_length = 8192
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    batch = generate_rollout(args, 0, source)
    assert len(batch) == len(records)
    for group, record in zip(batch, records, strict=True):
        sample = group[0]
        assert sample.sample_id == record["id"]
        assert sample.metadata == record["metadata"]
        assert sample.prompt == record["prompt"]
        assert sample.label == record["completion"]
        assert sample.tokens == tokenizer.apply_chat_template(
            record["prompt"] + record["completion"], tokenize=True, return_dict=False
        )
        supervised = tokenizer.decode(
            [
                token
                for token, keep in zip(
                    sample.tokens[-sample.response_length :],
                    sample.loss_mask,
                    strict=True,
                )
                if keep
            ]
        )
        assert (
            supervised
            == "\n\n</think>\n\n"
            + record["completion"][0]["content"].strip()
            + "<|im_end|>\n"
        )


def test_native_rl_data_source(tmp_path, monkeypatch):
    data_source = pytest.importorskip("slime.rollout.data_source")
    from transformers import AutoTokenizer

    checkpoint = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not checkpoint:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to local tokenizer files")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    monkeypatch.setattr(data_source, "load_tokenizer", lambda *a, **kw: tokenizer)
    # Text-only contract; loading a vision processor requires unrelated extras.
    monkeypatch.setattr(data_source, "load_processor", lambda *a, **kw: None)
    record = {
        "id": "physics:rl-contract",
        "prompt": [
            {"role": "user", "content": "Find the energy."},
            {"role": "assistant", "content": "Use the force."},
            {"role": "user", "content": "Give the final answer."},
        ],
        "completion": [{"role": "assistant", "content": "TEACHER_ONLY: E = mv²/2."}],
        "metadata": {"benchmark": "cmphysbench"},
    }
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps(record) + "\n")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=str(path),
        hf_checkpoint=checkpoint,
        dump_details=None,
        rollout_max_prompt_len=1024,
        input_key="prompt",
        multimodal_keys=None,
        label_key="completion",
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=True,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
    )
    source = data_source.RolloutDataSourceWithBuffer(args)
    group = source.get_samples(1)[0]
    expected = tokenizer.apply_chat_template(
        record["prompt"], tokenize=False, add_generation_prompt=True
    )
    assert [sample.index for sample in group] == [0, 1]
    assert all(sample.group_index == 0 for sample in group)
    assert all(sample.prompt == expected for sample in group)
    assert "TEACHER_ONLY" not in expected
    assert "Use the force." in expected
    assert group[0].label == record["completion"]
    assert group[0].metadata == record["metadata"]
    assert "id" not in group[0].to_dict()
    assert record["id"] not in json.dumps(group[0].to_dict())
    group[0].metadata["attempt"] = 1
    assert "attempt" not in group[1].metadata
    assert "attempt" not in source.dataset[0].metadata

    group[0].status = group[0].Status.ABORTED
    group[0].response = "Partial response"
    source.add_samples([group])
    resumed = source.get_samples(1)[0]
    assert resumed[0] is group[0]
    assert resumed[0].response == "Partial response"
    source.save(0)
    restored = data_source.RolloutDataSourceWithBuffer(args)
    restored.load(0)
    assert [s.index for s in restored.get_samples(1)[0]] == [2, 3]

    from ddpr.backends.slime.plugin import RolloutDataSource

    args.loss_type = "policy_loss"
    args.advantage_estimator = "grpo"
    args.compute_advantages_and_returns = True
    args.custom_rm_path = "example.reward"
    adapted = RolloutDataSource(args).get_samples(1)[0][0]
    assert adapted.prompt == expected
    assert adapted.label is None
    assert adapted.sample_id == record["id"]
    assert adapted.reference_completion == record["completion"][0]["content"]
    assert adapted.metadata == record["metadata"]

    args.loss_type = "sft_loss"
    args.n_samples_per_prompt = 1
    args.apply_chat_template = False
    args.compute_advantages_and_returns = False
    args.loss_mask_type = "qwen3_5"
    args.rollout_batch_size = 8
    args.seq_length = 4096
    batch = generate_rollout(args, 0, RolloutDataSource(args))
    assert len(batch) == 8
    for group in batch:
        sample = group[0]
        assert sample.sample_id == record["id"]
        assert sample.tokens == tokenizer.apply_chat_template(
            record["prompt"] + record["completion"], tokenize=True, return_dict=False
        )


def test_native_rl_reward_dispatch(tmp_path, monkeypatch):
    rm_hub = pytest.importorskip("slime.rollout.rm_hub")
    from slime.utils.types import Sample

    plugin = tmp_path / "ddpr_test_reward.py"
    plugin.write_text(
        "async def score(args, sample, **kwargs):\n"
        "    assert sample.response == 'Fresh response'\n"
        "    assert sample.label == [{'role': 'assistant', 'content': 'Teacher'}]\n"
        "    assert sample.metadata == {'problem_id': '1'}\n"
        "    return 0.5\n"
        "async def score_batch(args, samples, **kwargs):\n"
        "    return [await score(args, sample, **kwargs) for sample in samples]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sample = Sample(
        prompt="Policy prompt",
        response="Fresh response",
        label=[{"role": "assistant", "content": "Teacher"}],
        metadata={"problem_id": "1"},
    )
    args = SimpleNamespace(custom_rm_path="ddpr_test_reward.score")
    assert asyncio.run(rm_hub.async_rm(args, sample)) == 0.5
    args.custom_rm_path = "ddpr_test_reward.score_batch"
    assert asyncio.run(rm_hub.batched_async_rm(args, [sample, sample])) == [0.5, 0.5]


@pytest.mark.parametrize("history", [False, True])
def test_installed_slime_sft(tmp_path, history):
    if os.environ.get("DDPR_INSTALLED_SLIME") != "1":
        pytest.skip("set DDPR_INSTALLED_SLIME=1 in the Slime environment")
    checkpoint = os.environ["DDPR_QWEN38_TOKENIZER"]
    from slime.utils.arguments import get_slime_extra_args_provider
    from slime.utils.data import Dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    prompt = [
        {"role": "system", "content": "Solve physics problems."},
        {"role": "user", "content": "Find the kinetic energy."},
    ]
    if history:
        prompt += [
            {"role": "assistant", "content": "HISTORY: Integrate the force."},
            {"role": "user", "content": "Give the final expression."},
        ]
    completion = [{"role": "assistant", "content": "E = mv²/2."}]
    metadata = {"benchmark": "critpt", "synthetic_test": True}
    path = tmp_path / "synthetic-sft.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "completion": completion,
                "metadata": metadata,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = load_tokenizer(checkpoint, trust_remote_code=True)
    dataset = Dataset(
        str(path),
        tokenizer=tokenizer,
        processor=load_processor(checkpoint, trust_remote_code=True),
        max_length=None,
        prompt_key="prompt",
        label_key="completion",
        metadata_key="metadata",
        apply_chat_template=False,
    )
    source = dataset[0]
    original = deepcopy(source)
    buffer = SimpleNamespace(get_samples=lambda count: [[source]])
    parser = get_slime_extra_args_provider()(ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--disable-compute-advantages-and-returns",
            "--hf-checkpoint",
            checkpoint,
            "--loss-mask-type",
            "qwen3_5",
            "--input-key",
            "prompt",
            "--label-key",
            "completion",
            "--tool-key",
            "",
            "--loss-type",
            "sft_loss",
            "--n-samples-per-prompt",
            "1",
        ]
    )
    assert args.compute_advantages_and_returns is False
    args.seq_length = 4096
    sample = generate_rollout(args, 0, buffer)[0][0]
    assert sample.prompt == source.prompt == original.prompt == prompt
    assert sample.label == source.label == original.label == completion
    assert sample.metadata == metadata
    assert source.tokens == original.tokens
    assert sample.tokens == tokenizer.apply_chat_template(
        prompt + completion, tokenize=True, return_dict=False
    )
    response = sample.tokens[-sample.response_length :]
    assert len(response) == len(sample.loss_mask)
    supervised = tokenizer.decode(
        [token for token, keep in zip(response, sample.loss_mask) if keep]
    )
    assert supervised == "\n\n</think>\n\nE = mv²/2.<|im_end|>\n"
    assert sample.reward == 0
