from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from ddpr.data.adapters.ddsr_bench import load_sample as load_export
from ddpr.data.schemas import SftSample


def load_sample(record: Mapping[str, Any]) -> SftSample:
    """Load an exported text conversation supported by Slime."""
    example = load_export(record)
    if example.metadata.get("tools") or any(
        record.get(key) for key in ("tools", "images", "videos", "audios")
    ):
        raise ValueError("ddpr Slime supports text conversations without tools")
    return example


def read_records(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read JSON objects with their original line numbers for error reporting."""
    source_path = Path(path).expanduser()
    if source_path.suffix != ".jsonl":
        raise ValueError("ddpr requires an exported sft.jsonl file")
    found = False
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be an object")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source_path}:{line_number}: {error}") from error
            found = True
            yield line_number, record
    if not found:
        raise ValueError(f"{source_path}: no records")


class Dataset:
    def __init__(self, samples: list[Sample], seed: int) -> None:
        self.origin_samples = samples
        self.samples = samples
        self.seed = seed

    def shuffle(self, epoch_id: int) -> None:
        self.samples = self.origin_samples.copy()
        random.Random(self.seed + epoch_id).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
