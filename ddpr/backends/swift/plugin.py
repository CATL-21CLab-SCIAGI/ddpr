"""Register ddsr-bench exports through Swift's --external_plugins interface."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset
from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

from ddpr.backends.swift import rl, sft


class DdsrBenchPreprocessor(RowPreprocessor):
    def __init__(
        self, preprocess: Callable[[dict[str, Any]], dict[str, Any] | None]
    ) -> None:
        super().__init__()
        self._preprocess = preprocess

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        return self._preprocess(row)

    def __call__(
        self, dataset: Dataset | IterableDataset, **kwargs: Any
    ) -> Dataset | IterableDataset:
        # Swift reports the original and retained counts after preprocessing.
        result = super().__call__(dataset, **kwargs)
        if isinstance(result, Dataset) and not len(result):
            raise ValueError("no usable samples remain after preprocessing")
        return result


# Named entries let SFT and RL share a source file with different views.
path = Path(os.environ["DDPR_SFT_DATA"]).expanduser().resolve(strict=True)
if not path.is_file() or path.suffix != ".jsonl":
    raise ValueError("DDPR_SFT_DATA must point to an exported sft.jsonl file")
for task, preprocess in (("sft", sft.preprocess), ("rl", rl.preprocess)):
    register_dataset(
        DatasetMeta(
            dataset_name=f"ddpr_{task}",
            dataset_path=str(path),
            preprocess_func=DdsrBenchPreprocessor(preprocess),
        )
    )
