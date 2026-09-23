import json

import pytest

pytest.importorskip("slime.rollout.data_source")

from ddpr.backends.slime import data
from ddpr.backends.slime.plugin import RolloutDataSource


def test_read_records_preserves_fields_and_line_numbers(tmp_path):
    path = tmp_path / "sft.jsonl"
    record = {"id": "physics:1", "extra": {"notation": "ε"}}
    path.write_text("\n" + json.dumps(record) + "\n\n{}\n", encoding="utf-8")
    assert list(data.read_records(path)) == [(2, record), (4, {})]


@pytest.mark.parametrize("invalid", ["{broken", "[]", "null"])
def test_read_records_reports_invalid_line(tmp_path, invalid):
    path = tmp_path / "sft.jsonl"
    path.write_text("\n{}\n\n" + invalid + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sft.jsonl:4:"):
        list(data.read_records(path))


@pytest.mark.parametrize("task", ["sft", "rl"])
def test_conversion_error_keeps_original_line(args, record, task):
    if task == "sft":
        args.loss_type = "sft_loss"
        args.n_samples_per_prompt = 1
        args.compute_advantages_and_returns = False
        args.apply_chat_template = False
        args.loss_mask_type = "qwen3_5"
    valid = json.dumps(record)
    record["completion"] = []
    with open(args.prompt_data, "w", encoding="utf-8") as source:
        source.write("\n" + valid + "\n\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="sft.jsonl:4: completion must not be empty"):
        RolloutDataSource(args)


@pytest.mark.parametrize("change", ["empty", "bad_json", "long", "tools", "assistant"])
def test_invalid_export_reports_location(args, record, change):
    if change == "empty":
        content = ""
    elif change == "bad_json":
        content = "{broken"
    else:
        if change == "long":
            args.rollout_max_prompt_len = 1
        elif change == "tools":
            record["tools"] = [{"name": "tool"}]
        else:
            record["prompt"].pop()
        content = json.dumps(record)
    with open(args.prompt_data, "w") as data:
        data.write(content)
    with pytest.raises(ValueError, match="sft.jsonl:"):
        RolloutDataSource(args)
