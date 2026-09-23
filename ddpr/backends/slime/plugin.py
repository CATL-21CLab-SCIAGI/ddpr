"""Connect ddpr's training samples to Slime's data-source interface."""

from __future__ import annotations

from argparse import Namespace
from copy import copy

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample

from ddpr.backends.slime import rl, sft
from ddpr.backends.slime.data import Dataset


class RolloutDataSource(RolloutDataSourceWithBuffer):
    """Adapt exports before Slime groups prompts and buffers unfinished rollouts."""

    def __init__(self, args: Namespace) -> None:
        loaders = {"sft_loss": sft.load_samples, "policy_loss": rl.load_samples}
        if args.loss_type not in loaders:
            raise ValueError(f"unsupported ddpr loss type: {args.loss_type!r}")
        samples = loaders[args.loss_type](args)
        # Initialize Slime's counters and retry buffer without its lossy loader.
        source_args = copy(args)
        source_args.prompt_data = None
        super().__init__(source_args)
        self.args = args
        self.dataset = Dataset(samples, args.rollout_seed)
        if args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if num_samples < 0:
            raise ValueError("num_samples must not be negative")
        # Native sampling crosses at most one epoch per call. Small exports may
        # need several epochs to fill an oversampled rollout batch.
        groups = []
        while len(groups) < num_samples:
            count = min(num_samples - len(groups), len(self.dataset))
            groups.extend(super().get_samples(count))
        return groups
