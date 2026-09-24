from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ddpr.backends.swift.data import load_sample


def preprocess(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prepare an RL prompt and reference, skipping empty reference text."""
    example = load_sample(record)
    if not example.completion[0]["content"].strip():
        return None
    return {
        "messages": [dict(message) for message in example.prompt],
        "sample_id": example.id,
        "metadata": deepcopy(example.metadata),
        # Available to reward plugins, never appended to the policy prompt.
        "reference_completion": example.completion[0]["content"],
    }
