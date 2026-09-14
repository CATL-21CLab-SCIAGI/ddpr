"""Optional upstream contract checks with an in-memory synthetic tokenizer.

These verify Slime's actual masking code, not a particular student checkpoint.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from ddpr.backends.slime.sft import _messages
from ddpr.data.adapters.ddsr_bench import load_sample


@pytest.mark.parametrize("mask_type", ["qwen", "qwen3", "qwen3_5"])
def test_upstream_masks_preserve_history_and_supervise_only_completion(mask_type):
    root = os.environ.get("DDPR_SLIME_ROOT")
    if not root:
        pytest.skip("set DDPR_SLIME_ROOT to a Slime checkout")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    path = Path(root) / "slime/utils/mask_utils.py"
    spec = importlib.util.spec_from_file_location("ddpr_upstream_masks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vocabulary = [
        "[UNK]",
        "<|im_start|>",
        "<|im_end|>",
        "system",
        "user",
        "assistant",
        "FOR",
        "TESTING",
        "ONLY",
        "CALCULATING",
        "LOSS",
        "MASK",
        "instruction",
        "question",
        "derivation",
        "format",
        "answer",
    ]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    example = load_sample(
        {
            "prompt": [
                {"role": "system", "content": "instruction"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "derivation"},
                {"role": "user", "content": "format"},
            ],
            "completion": [{"role": "assistant", "content": "answer"}],
        }
    )
    conversation = _messages(example)
    generator = module.MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=mask_type)
    tokens, mask = generator.get_loss_mask(conversation)
    assert tokens == tokenizer.apply_chat_template(
        conversation, tokenize=True, return_dict=False
    )
    assert len(tokens) == len(mask)
    assert "derivation" in tokenizer.decode(tokens)
    supervised = tokenizer.decode([token for token, keep in zip(tokens, mask) if keep])
    assert supervised == "answer <|im_end|>"
