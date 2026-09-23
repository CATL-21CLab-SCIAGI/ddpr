from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ddpr.backends.swift.data import load_sample
from ddpr.data.schemas import SftSample


def _messages(example: SftSample) -> list[dict[str, str | bool]]:
    """Preserve context while supervising only the exported completion."""
    messages = [dict(message) for message in example.prompt]
    for message in messages:
        if message["role"] == "assistant":
            message["loss"] = False
    return messages + [dict(message, loss=True) for message in example.completion]


def preprocess(record: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare one offline SFT example through Swift's dataset interface."""
    example = load_sample(record)
    return {
        "messages": _messages(example),
        "sample_id": example.id,
        "metadata": deepcopy(example.metadata),
    }
