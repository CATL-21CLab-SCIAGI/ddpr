import json
import os
from itertools import islice
from pathlib import Path

import pytest


@pytest.fixture
def exported_sft(tmp_path):
    source = os.environ.get("DDPR_TEST_SFT_DATA")
    if not source:
        pytest.skip("set DDPR_TEST_SFT_DATA to a real ddsr-bench export")
    limit = int(os.environ.get("DDPR_TEST_SFT_LIMIT", "3"))
    if limit < 1:
        raise ValueError("DDPR_TEST_SFT_LIMIT must be positive")
    with Path(source).expanduser().open(encoding="utf-8") as stream:
        lines = list(islice((line for line in stream if line.strip()), limit))
    assert lines, "the exported dataset is empty"
    records = [json.loads(line) for line in lines]
    # Retain complete original records; no filtering, adaptation or truncation.
    path = tmp_path / "exported-sft.jsonl"
    path.write_text("".join(lines), encoding="utf-8")
    return path, records
