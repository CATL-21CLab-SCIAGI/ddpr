"""Optional completion-mask checks against Swift's real Qwen3.8 template."""

import os

import pytest

from ddpr.backends.swift import sft

pytestmark = pytest.mark.skipif(
    os.environ.get("DDPR_INSTALLED_SWIFT") != "1",
    reason="set DDPR_INSTALLED_SWIFT=1 in an MS-Swift environment",
)


@pytest.fixture
def template():
    checkpoint = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not checkpoint:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to local tokenizer files")
    import torch
    from swift.model import ModelInfo, ModelMeta
    from swift.template import get_template
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.model_info = ModelInfo(
        model_type="qwen3_5",
        model_dir=checkpoint,
        torch_dtype=torch.bfloat16,
        max_model_len=8192,
        quant_method=None,
        quant_bits=None,
        config=AutoConfig.from_pretrained(checkpoint, local_files_only=True),
        task_type="causal_lm",
    )
    tokenizer.model_meta = ModelMeta(model_type="qwen3_5", model_groups=[])
    template = get_template(tokenizer, template_type="qwen3_8", max_length=8192)
    template.set_mode("train")
    return template


@pytest.mark.parametrize(
    "completion",
    ["ψ(x) = 0", "<think>\nApply the boundary condition.\n</think>\n\nψ(x) = 0"],
)
def test_qwen38_completion_labels(record, template, completion):
    from swift.template import MaxLengthError

    tokenizer = template.tokenizer
    record["completion"][0]["content"] = completion
    row = sft.preprocess(record)
    encoded = template.encode(row)
    text = tokenizer.decode(encoded["input_ids"])
    supervised = tokenizer.decode([v for v in encoded["labels"] if v != -100])
    assert record["prompt"][2]["content"] in text
    assert record["prompt"][2]["content"] not in supervised
    assert record["prompt"][-1]["content"] not in supervised
    assert record["completion"][0]["content"] in supervised
    assert "<|im_end|>" in supervised
    template.max_length = 1
    with pytest.raises(MaxLengthError):
        template.encode(row)


def test_exported_sft_labels(exported_sft, template):
    _, records = exported_sft
    for record in records:
        row = sft.preprocess(record)
        if row is None:
            continue
        encoded = template.encode(row)
        rendered = template.tokenizer.decode(encoded["input_ids"])
        supervised = template.tokenizer.decode(
            [v for v in encoded["labels"] if v != -100]
        )
        completion = record["completion"][0]["content"].strip()
        # Swift recognizes inline thinking instead of prepending an empty block.
        prefix = (
            ""
            if completion.startswith("<think>") and "</think>" in completion
            else "<think>\n\n</think>\n\n"
        )
        assert supervised == prefix + completion + "<|im_end|>\n"
        for message in record["prompt"]:
            # Swift's official Qwen template strips surrounding whitespace.
            assert message["content"].strip() in rendered


@pytest.mark.parametrize("lazy", [False, True])
def test_strict_overlength_fails_without_dropping_rows(record, template, lazy):
    from datasets import Dataset
    from swift.dataset.utils import AddLengthPreprocessor, LazyLLMDataset
    from swift.template import MaxLengthError

    template.max_length = 1
    # Swift maps the launcher's `delete` strategy to the template's `raise`.
    template.truncation_strategy = "raise"
    rows = Dataset.from_list([sft.preprocess(record)])
    with pytest.raises(MaxLengthError):
        if lazy:
            LazyLLMDataset(rows, template.encode, strict=True)[0]
        else:
            AddLengthPreprocessor(template=template)(rows, strict=True, num_proc=None)
    assert len(rows) == 1
