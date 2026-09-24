from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ddpr.data.schemas import SftSample


def _messages(value: Any, field: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a message list")
    if not value:
        raise ValueError(f"{field} must not be empty")
    messages = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"{field}[{index}] must be a message object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"{field}[{index}] has an unsupported role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"{field}[{index}].content must be text")
        # Exported SFT messages contain visible content, not provider reasoning.
        messages.append({"role": role, "content": content})
    return tuple(messages)


def load_sample(record: Mapping[str, Any]) -> SftSample:
    """Normalize one exported ddsr-bench SFT record."""
    prompt = _messages(record.get("prompt"), "prompt")
    completion = _messages(record.get("completion"), "completion")
    if len(completion) != 1 or completion[0]["role"] != "assistant":
        raise ValueError("completion must contain exactly one assistant message")
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be an object")
    sample_id = record.get("id")
    if sample_id is not None and not isinstance(sample_id, str):
        raise TypeError("sample ID must be a string or None")
    return SftSample(
        id=sample_id,
        prompt=prompt,
        completion=completion,
        metadata=dict(metadata or {}),
    )
