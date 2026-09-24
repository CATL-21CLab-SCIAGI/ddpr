"""Load user-selected reward implementations."""

import importlib
from functools import cache

from ddpr.rewards.base import Reward


@cache
def load_reward(path: str) -> Reward:
    """Instantiate an importable, zero-argument Reward subclass once per process."""
    module, _, name = path.rpartition(".")
    if not module or not name:
        raise ValueError("reward must be specified as module.Class")
    reward_type = getattr(importlib.import_module(module), name)
    if not isinstance(reward_type, type) or not issubclass(reward_type, Reward):
        raise TypeError(f"{path} must be a Reward subclass")
    return reward_type()
