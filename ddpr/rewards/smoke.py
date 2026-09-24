"""Synthetic reward example for training integration checks."""

import hashlib
from collections.abc import Mapping
from typing import Any

from ddpr.rewards.base import Reward


class SmokeReward(Reward):
    """Deterministic text-hash score for integration tests, not answer quality."""

    def __call__(
        self,
        *,
        response: str,
        sample_id: str,
        reference_completion: str,
        metadata: Mapping[str, Any],
    ) -> float:
        # Different texts usually produce different advantages; identical texts
        # deliberately receive identical scores, including across worker ranks.
        digest = hashlib.sha256(response.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / (2**64 - 1)
