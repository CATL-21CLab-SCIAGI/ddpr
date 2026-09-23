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


def test_qwen38_completion_labels(record, template):
    from swift.template import MaxLengthError

    tokenizer = template.tokenizer
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
