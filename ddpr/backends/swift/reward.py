"""Register the shared Reward contract through Swift's external plugin API."""

import os

from swift.rewards import ORM, orms

from ddpr.rewards import load_reward


class RewardAdapter(ORM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward = load_reward(os.environ["DDPR_REWARD"])

    def __call__(
        self, completions, sample_id, reference_completion, metadata, **kwargs
    ):
        return [
            self.reward(
                response=response,
                sample_id=identifier,
                reference_completion=reference,
                metadata=meta,
            )
            for response, identifier, reference, meta in zip(
                completions, sample_id, reference_completion, metadata, strict=True
            )
        ]


orms["ddpr_reward"] = RewardAdapter
