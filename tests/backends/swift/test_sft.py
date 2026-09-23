from copy import deepcopy

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
