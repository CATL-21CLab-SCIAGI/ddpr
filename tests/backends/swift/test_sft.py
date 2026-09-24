from copy import deepcopy

import pytest

from ddpr.backends.swift import sft


def test_sft_masks_only_exported_completion(record):
    original = deepcopy(record)
    row = sft.preprocess(record)
    assert [m.get("loss") for m in row["messages"]] == [None, None, False, None, True]
    assert row["sample_id"] == record["id"]
    assert row["messages"][-1]["content"] == "ψ(x) = 0"
    row["messages"][0]["content"] = "changed"
    row["metadata"]["quality"]["verified"] = True
    assert record == original


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_completion_is_skipped_without_mutation(record, content):
    record["completion"][0]["content"] = content
    original = deepcopy(record)
    assert sft.preprocess(record) is None
    assert record == original


@pytest.mark.parametrize("field,value", [("metadata", []), ("tools", ["tool"])])
def test_empty_completion_does_not_hide_invalid_input(record, field, value):
    record["completion"][0]["content"] = ""
    record[field] = value
    with pytest.raises((TypeError, ValueError)):
        sft.preprocess(record)


@pytest.mark.parametrize(
    "completion", [None, [], [{"role": "assistant", "content": 0}]]
)
def test_malformed_completion_is_not_skipped(record, completion):
    record["completion"] = completion
    with pytest.raises((TypeError, ValueError)):
        sft.preprocess(record)
