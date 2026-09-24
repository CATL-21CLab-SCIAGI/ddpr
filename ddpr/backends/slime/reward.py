"""Adapt the shared Reward contract to Slime's single/group async hook."""

import asyncio
from threading import Lock

from ddpr.rewards import load_reward

_lock = Lock()


async def reward(args, sample, **kwargs):

    def score(scorer, item):
        return scorer(
            response=item.response,
            sample_id=item.sample_id,
            reference_completion=item.reference_completion,
            metadata=item.metadata,
        )

    # Hold the lock inside the worker: cancellation of its awaiting coroutine
    # must not allow another call to enter a still-running verifier.
    def evaluate():
        with _lock:
            scorer = load_reward(args.ddpr_reward)
            if isinstance(sample, list):
                return [score(scorer, item) for item in sample]
            return score(scorer, sample)

    return await asyncio.to_thread(evaluate)
