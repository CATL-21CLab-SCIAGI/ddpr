from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SftSample:
    """Mirror ddsr-bench's SFT contract without importing its package.

    The ID is optional because some framework loaders discard that field.
    """

    prompt: tuple[dict[str, str], ...]
    completion: tuple[dict[str, str], ...]
    metadata: dict[str, Any]
    id: str | None = None
