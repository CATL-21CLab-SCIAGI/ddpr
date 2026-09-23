from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample
from transformers import PreTrainedTokenizerBase

from ddpr.backends.slime import data
from ddpr.backends.slime.data import load_sample


def _sample(
    record: Mapping[str, Any], tokenizer: PreTrainedTokenizerBase, args: Namespace
) -> Sample:
    example = load_sample(record)
    if example.prompt[-1]["role"] != "user":
        raise ValueError("RL prompts must end with a user message")
    prompt = tokenizer.apply_chat_template(
        list(example.prompt),
        tokenize=False,
        add_generation_prompt=True,
        **(args.apply_chat_template_kwargs or {}),
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    limit = args.rollout_max_prompt_len
    if limit is not None and len(tokens) > limit:
        raise ValueError(
            f"{len(tokens)} prompt tokens exceed rollout_max_prompt_len={limit}"
        )
    sample = Sample(prompt=prompt, metadata=deepcopy(example.metadata))
    # Slime preserves extra attributes in Sample.to_dict/from_dict and deep copies.
    sample.sample_id = example.id
    sample.reference_completion = example.completion[0]["content"]
    return sample


def _validate_args(args: Namespace) -> None:
    required = {
        "rollout_global_dataset": True,
        "input_key": "prompt",
        "label_key": "completion",
        "metadata_key": "metadata",
        "apply_chat_template": True,
        "loss_type": "policy_loss",
        "compute_advantages_and_returns": True,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"ddpr RL requires {name}={expected!r}")
    if args.n_samples_per_prompt < 2:
        raise ValueError("ddpr GRPO requires at least two samples per prompt")
    if args.advantage_estimator != "grpo":
        raise ValueError("ddpr Slime RL currently supports GRPO")
    if not args.custom_rm_path:
        raise ValueError("ddpr RL requires an explicit custom reward function")
    if getattr(args, "debug_train_only", False):
        raise ValueError("ddpr RL requires generation workers")
    if getattr(args, "multimodal_keys", None) or getattr(args, "tool_key", None):
        raise ValueError("ddpr RL supports text conversations without tools")


def load_samples(args: Namespace) -> list[Sample]:
    """Prepare prompt-only RL samples with IDs and teacher-reference fields."""
    _validate_args(args)
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    source_path = Path(args.prompt_data).expanduser()
    samples = []
    for line_number, record in data.read_records(source_path):
        try:
            samples.append(_sample(record, tokenizer=tokenizer, args=args))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source_path}:{line_number}: {error}") from error
    if getattr(args, "dump_details", None) is not None:
        tokenizer.save_pretrained(Path(args.dump_details) / "tokenizer")

    return samples
