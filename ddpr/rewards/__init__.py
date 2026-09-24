"""Shared reward contract, loading and synthetic example."""

from ddpr.rewards.base import Reward
from ddpr.rewards.loader import load_reward
from ddpr.rewards.smoke import SmokeReward

__all__ = ["Reward", "SmokeReward", "load_reward"]
