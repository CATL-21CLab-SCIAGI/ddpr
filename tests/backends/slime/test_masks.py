"""Installed Slime mask checks with synthetic and real tokenizers."""

import os

import pytest

pytest.importorskip("slime")

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

from ddpr.backends.slime.sft import _messages
from ddpr.data.adapters.ddsr_bench import load_sample


@pytest.mark.parametrize("mask_type", ["qwen", "qwen3", "qwen3_5"])
def test_upstream_masks_preserve_history_and_supervise_only_completion(mask_type):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
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
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=mask_type)
    tokens, mask = generator.get_loss_mask(conversation)
    assert tokens == tokenizer.apply_chat_template(
        conversation, tokenize=True, return_dict=False
    )
    assert len(tokens) == len(mask)
    assert "derivation" in tokenizer.decode(tokens)
    supervised = tokenizer.decode([token for token, keep in zip(tokens, mask) if keep])
    assert supervised == "answer <|im_end|>"


@pytest.mark.parametrize(
    "history,system", [(False, False), (False, True), (True, False), (True, True)]
)
def test_qwen38_completion_mask(history, system):
    checkpoint = os.environ.get("DDPR_QWEN38_TOKENIZER")
    if not checkpoint:
        pytest.skip("set DDPR_QWEN38_TOKENIZER to the local Qwen3.8-27B tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = [{"role": "user", "content": "QUESTION: Find the energy of a particle."}]
    if system:
        prompt.insert(
            0, {"role": "system", "content": "SYSTEM: Solve physics carefully."}
        )
    if history:
        prompt.extend(
            [
                {
                    "role": "assistant",
                    "content": "HISTORY: Integrate the force to obtain the energy.",
                },
                {
                    "role": "user",
                    "content": "FORMAT: Include an equation and Python code.",
                },
            ]
        )
    answer = "The energy is \u03b5 = mv²/2.\n```python\ndef answer(m, v):\n    return m*v*v/2\n```"
    conversation = _messages(
        load_sample(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": answer}],
            }
        )
    )
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    tokens, mask = generator.get_loss_mask(conversation)
    assert tokens == tokenizer.apply_chat_template(
        conversation, tokenize=True, return_dict=False
    )
    assert len(tokens) == len(mask)
    rendered = tokenizer.decode(tokens)
    for message in prompt:
        assert message["content"] in rendered
    supervised = tokenizer.decode([token for token, keep in zip(tokens, mask) if keep])
    # No hidden reasoning is exported. Qwen's official template supplies an
    # empty thinking block; Slime masks its opener but supervises its closure.
    assert supervised == "\n\n</think>\n\n" + answer + "<|im_end|>\n"
