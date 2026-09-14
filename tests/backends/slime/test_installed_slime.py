"""Exercise the installed Slime loader and ddpr adapter without model weights."""

import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ddpr.backends.slime.sft import generate_rollout


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
