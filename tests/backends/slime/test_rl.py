import asyncio
import json

import pytest

pytest.importorskip("slime.rollout.data_source")

from ddpr.backends.slime.plugin import RolloutDataSource


def test_prompt_reference_and_isolation(args, record):
    from slime.utils.types import Sample

    source = RolloutDataSource(args)
    groups = source.get_samples(5)
    assert len(groups) == 5  # Several epochs of a one-row export.
    assert [g[0].group_index for g in groups] == list(range(5))
    assert [s.index for g in groups for s in g] == list(range(10))
    sample = groups[0][0]
    assert "Earlier derivation." in sample.prompt
    assert "TEACHER" not in sample.prompt
    assert sample.prompt.count("assistant:") == 2
    assert sample.label is None
    assert sample.sample_id == record["id"]
    assert sample.reference_completion == record["completion"][0]["content"]
    assert sample.metadata == record["metadata"]
    serialized = Sample.from_dict(sample.to_dict())
    assert serialized.sample_id == sample.sample_id
    assert serialized.reference_completion == sample.reference_completion
    sample.metadata["quality"]["verified"] = True
    assert not groups[0][1].metadata["quality"]["verified"]
    assert not source.dataset.samples[0].metadata["quality"]["verified"]


def test_retry_and_resume(args, record):
    # Multiple records exercise deterministic shuffling after restoring the cursor.
    with open(args.prompt_data, "w") as data:
        data.writelines(
            json.dumps({**record, "id": str(index)}) + "\n" for index in range(3)
        )
    source = RolloutDataSource(args)
    group = source.get_samples(1)[0]
    group[0].status = group[0].Status.ABORTED
    group[0].response = "Partial generation"
    source.add_samples([group])
    retried = source.get_samples(1)[0]
    assert retried[0] is group[0]
    assert retried[0].response == "Partial generation"
    source.get_samples(4)
    source.save(0)
    restored = RolloutDataSource(args)
    restored.load(0)
    expected = source.get_samples(5)
    actual = restored.get_samples(5)
    assert [[s.to_dict() for s in g] for g in actual] == [
        [s.to_dict() for s in g] for g in expected
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("apply_chat_template", False),
        ("loss_type", "sft_loss"),
        ("compute_advantages_and_returns", False),
        ("n_samples_per_prompt", 1),
        ("custom_rm_path", None),
        ("advantage_estimator", "ppo"),
        ("debug_train_only", True),
        ("multimodal_keys", {"image": "images"}),
    ],
)
def test_reject_sft_or_unsupported_settings(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        RolloutDataSource(args)


def test_native_reward_receives_ddpr_fields(args, tmp_path, monkeypatch):
    from slime.rollout.rm_hub import async_rm, batched_async_rm

    plugin = tmp_path / "ddpr_rl_reward.py"
    plugin.write_text(
        "async def score(args, sample, **kwargs):\n"
        "    assert sample.sample_id == 'physics:1'\n"
        "    assert sample.reference_completion == 'TEACHER: E = mv²/2.'\n"
        "    assert sample.label is None\n"
        "    assert sample.response == 'Fresh response'\n"
        "    assert not sample.metadata['quality']['verified']\n"
        "    return 0.25\n"
        "async def batch(args, samples, **kwargs):\n"
        "    return [await score(args, sample) for sample in samples]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args.custom_rm_path = "ddpr_rl_reward.score"
    group = RolloutDataSource(args).get_samples(1)[0]
    for sample in group:
        sample.response = "Fresh response"
    assert asyncio.run(async_rm(args, group[0])) == 0.25
    args.custom_rm_path = "ddpr_rl_reward.batch"
    assert asyncio.run(batched_async_rm(args, group)) == [0.25, 0.25]
