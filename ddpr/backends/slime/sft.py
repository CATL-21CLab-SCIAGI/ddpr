from __future__ import annotations

from argparse import Namespace
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slime.rollout.data_source import DataSource
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.types import Sample

from ddpr.data.adapters.ddsr_bench import load_sample
from ddpr.data.schemas import SftSample


def _messages(example: SftSample) -> list[dict[str, str | int]]:
    """Preserve context while supervising only the exported completion."""
    return [dict(message, step_loss_mask=0) for message in example.prompt] + [
        dict(message, step_loss_mask=1) for message in example.completion
    ]


@lru_cache(maxsize=1)
def _mask_generator(checkpoint: str, mask_type: str) -> MultiTurnLossMaskGenerator:
    # Slime is installed separately in the training worker environment.
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(checkpoint, trust_remote_code=True)
    if mask_type == "qwen" and "<｜Assistant｜>" in tokenizer.get_added_vocab():
        raise ValueError("Slime's distill_qwen mask drops multi-message context")
    return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=mask_type)


def _validate_args(args: Namespace, evaluation: bool) -> None:
    if evaluation:
        raise ValueError("ddpr SFT does not perform generation-based evaluation")
    required = {
        "rollout_global_dataset": True,
        "input_key": "prompt",
        "label_key": "completion",
        "loss_type": "sft_loss",
        "n_samples_per_prompt": 1,
        "disable_compute_advantages_and_returns": True,
    }
    for name, expected in required.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"ddpr SFT requires {name}={expected!r}")
    if getattr(args, "apply_chat_template", False):
        raise ValueError("leave --apply-chat-template unset for ddpr SFT")
    if getattr(args, "multimodal_keys", None) or getattr(args, "tool_key", None):
        raise ValueError("ddpr SFT supports exported text conversations only")
    if args.loss_mask_type not in ("qwen", "qwen3", "qwen3_5"):
        raise ValueError("use a supported multi-turn loss mask: qwen, qwen3, qwen3_5")


def _sample(
    source: Sample,
    generator: MultiTurnLossMaskGenerator,
    limit: int | None,
) -> Sample:
    example = load_sample(
        {
            "prompt": source.prompt,
            "completion": source.label,
            "metadata": source.metadata,
        }
    )
    if example.metadata.get("tools"):
        raise ValueError("tool definitions are unsupported for text SFT")
    tokens, mask = generator.get_loss_mask(_messages(example))
    if len(tokens) != len(mask):
        raise ValueError("token IDs and loss mask have different lengths")
    if not mask or any(value not in (0, 1) for value in mask) or 1 not in mask:
        raise ValueError("loss mask must be binary with supervised tokens")
    if limit is not None and len(tokens) > limit:
        raise ValueError(f"{len(tokens)} tokens exceed seq_length={limit}")

    sample = copy(source)
    sample.tokens = tokens
    sample.response_length = len(mask) - mask.index(1)
    sample.loss_mask = mask[-sample.response_length :]
    sample.reward = 0
    return sample


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_buffer: DataSource,
    evaluation: bool = False,
) -> list[list[Sample]]:
    """Prepare offline SFT batches through Slime's rollout interface."""
    _validate_args(args, evaluation)
    generator = _mask_generator(args.hf_checkpoint, args.loss_mask_type)
    limit = getattr(args, "seq_length", None)
    batches = []
    for position, group in enumerate(data_buffer.get_samples(args.rollout_batch_size)):
        if len(group) != 1:
            raise ValueError("ddpr SFT requires one sample per prompt group")
        source = group[0]
        try:
            sample = _sample(source, generator, limit)
        except (TypeError, ValueError) as error:
            index = getattr(source, "index", position)
            raise ValueError(f"SFT sample {index}: {error}") from error
        batches.append([sample])
    return batches
