from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ddpr.data.adapters.ddsr_bench import load_sample as load_export
from ddpr.data.schemas import SftSample


def load_sample(record: Mapping[str, Any]) -> SftSample:
    """Load an exported text conversation supported by Swift."""
    sample = load_export(record)
    if sample.metadata.get("tools") or any(
        record.get(key) for key in ("tools", "images", "videos", "audios")
    ):
        raise ValueError("ddpr Swift supports text conversations without tools")
    if sample.prompt[-1]["role"] != "user":
        raise ValueError("Swift prompts must end with a user message")

    return sample
