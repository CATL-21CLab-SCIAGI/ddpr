from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("slime")

from ddpr.backends.slime import sft
from ddpr.data.adapters.ddsr_bench import load_sample


def test_source_preserves_ids_and_fills_small_dataset(args, record, monkeypatch):
    pytest.importorskip("slime.rollout.data_source")
    from slime.utils.types import Sample

    from ddpr.backends.slime.plugin import RolloutDataSource

    args.loss_type = "sft_loss"
    args.n_samples_per_prompt = 1
    args.compute_advantages_and_returns = False
    args.apply_chat_template = False
    args.loss_mask_type = "qwen3_5"
    args.rollout_batch_size = 8
    args.seq_length = 4096
    monkeypatch.setattr(sft, "_mask_generator", lambda *a: CharacterMask())
    source = RolloutDataSource(args)
    batch = sft.generate_rollout(args, 0, source)
    assert len(batch) == 8
    assert [group[0].index for group in batch] == list(range(8))
    for group in batch:
        sample = group[0]
        assert sample.sample_id == record["id"]
        assert sample.prompt == record["prompt"]
        assert sample.label == record["completion"]
        assert sample.metadata == record["metadata"]
        assert Sample.from_dict(sample.to_dict()).sample_id == record["id"]
    batch[0][0].metadata["quality"]["verified"] = True
    assert not batch[1][0].metadata["quality"]["verified"]
    assert not source.dataset.samples[0].metadata["quality"]["verified"]

    source.save(0)
    restored = RolloutDataSource(args)
    restored.load(0)
    resumed = sft.generate_rollout(args, 1, restored)
    assert len(resumed) == 8
    assert [group[0].index for group in resumed] == list(range(8, 16))
    assert all(group[0].sample_id == record["id"] for group in resumed)


def arguments(**overrides):
    return SimpleNamespace(
        **{
            "rollout_global_dataset": True,
            "input_key": "prompt",
            "label_key": "completion",
            "metadata_key": "metadata",
            "loss_type": "sft_loss",
            "n_samples_per_prompt": 1,
            "compute_advantages_and_returns": False,
            "apply_chat_template": False,
            "hf_checkpoint": "student",
            "loss_mask_type": "qwen3",
            "rollout_batch_size": 2,
            "seq_length": 4096,
            **overrides,
        }
    )


def example(benchmark="critpt", history=False):
    prompt = [
        {"role": "system", "content": "Solve physics problems."},
        {"role": "user", "content": "Question or preceding SciCode functions."},
    ]
    if history:
        prompt += [
            {"role": "assistant", "content": "Earlier derivation."},
            {"role": "user", "content": "Format the answer."},
        ]
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": "Final answer."}],
        "metadata": {"benchmark": benchmark, "stage": "answer"},
    }


class Buffer:
    def __init__(self, samples):
        self.samples = samples

    def get_samples(self, count):
        return self.samples[:count]


class CharacterMask:
    """Small test double exposing exactly which message text is supervised."""

    def get_loss_mask(self, messages):
        tokens, masks = [], []
        for message in messages:
            tokens.extend(map(ord, message["content"]))
            masks.extend([message["step_loss_mask"]] * len(message["content"]))
        return tokens, masks


@pytest.mark.parametrize(
    "benchmark,history",
    [("critpt", False), ("critpt", True), ("scicode", False), ("cmphysbench", False)],
)
def test_completion_only_and_original_samples_unchanged(
    monkeypatch, benchmark, history
):
    row = example(benchmark, history)
    source = SimpleNamespace(
        prompt=row["prompt"], label=row["completion"], metadata=row["metadata"], index=7
    )
    original = deepcopy(source)
    monkeypatch.setattr(sft, "_mask_generator", lambda *args: CharacterMask())
    result = sft.generate_rollout(arguments(), 0, Buffer([[source]]))[0][0]
    assert source == original
    assert result is not source
    assert result.metadata == row["metadata"]
    assert result.prompt == row["prompt"]
    target = row["completion"][0]["content"]
    assert result.response_length == len(target)
    assert result.loss_mask == [1] * len(target)
    assert "".join(map(chr, result.tokens[-result.response_length :])) == target
    if history:
        assert "Earlier derivation." in "".join(map(chr, result.tokens))


def test_message_masks_override_input_flags():
    row = example(history=True)
    row["prompt"][2]["step_loss_mask"] = 1
    row["completion"][0]["step_loss_mask"] = 0
    converted = sft._messages(load_sample(row))
    assert [m["step_loss_mask"] for m in converted] == [0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    "override",
    [
        {"apply_chat_template": True},
        {"label_key": None},
        {"input_key": "messages"},
        {"metadata_key": "other_metadata"},
        {"apply_chat_template_kwargs": {"enable_thinking": False}},
        {"tool_key": "tools"},
        {"n_samples_per_prompt": 2},
        {"loss_type": "policy_loss"},
        {"rollout_global_dataset": False},
        {"compute_advantages_and_returns": True},
        {"loss_mask_type": "distill_qwen"},
        {"multimodal_keys": {"image": "image"}},
    ],
)
def test_invalid_configuration_fails_before_loading_tokenizer(override):
    with pytest.raises(ValueError):
        sft.generate_rollout(arguments(**override), 0, Buffer([]))


def test_evaluation_is_rejected():
    with pytest.raises(ValueError, match="evaluation"):
        sft.generate_rollout(arguments(), 0, Buffer([]), evaluation=True)


@pytest.mark.parametrize(
    "tokens,mask,limit,match",
    [
        ([1, 2], [1], 10, "different lengths"),
        ([1, 2], [0, 0], 10, "supervised"),
        ([1, 2], [0, 2], 10, "binary"),
        ([1, 2], [0, 1], 1, "seq_length"),
    ],
)
def test_bad_tokenization_has_sample_context(monkeypatch, tokens, mask, limit, match):
    row = example()
    source = SimpleNamespace(
        prompt=row["prompt"], label=row["completion"], metadata={}, index=42
    )
    generator = SimpleNamespace(get_loss_mask=lambda messages: (tokens, mask))
    monkeypatch.setattr(sft, "_mask_generator", lambda *args: generator)
    with pytest.raises(ValueError, match=f"SFT sample 42:.*{match}"):
        sft.generate_rollout(arguments(seq_length=limit), 0, Buffer([[source]]))


def test_batch_order_and_group_shape(monkeypatch):
    row = example()
    samples = [
        SimpleNamespace(
            prompt=row["prompt"], label=row["completion"], metadata={}, index=i
        )
        for i in range(2)
    ]
    monkeypatch.setattr(sft, "_mask_generator", lambda *args: CharacterMask())
    result = sft.generate_rollout(arguments(), 0, Buffer([[s] for s in samples]))
    assert [group[0].index for group in result] == [0, 1]
    with pytest.raises(ValueError, match="one sample"):
        sft.generate_rollout(arguments(), 0, Buffer([samples]))
