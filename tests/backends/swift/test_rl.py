import pytest

from ddpr.backends.swift import rl


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_reference_is_skipped(record, content):
    record["completion"][0]["content"] = content
    assert rl.preprocess(record) is None


@pytest.mark.parametrize("field,value", [("metadata", []), ("tools", ["tool"])])
def test_empty_reference_does_not_hide_invalid_input(record, field, value):
    record["completion"][0]["content"] = ""
    record[field] = value
    with pytest.raises((TypeError, ValueError)):
        rl.preprocess(record)


def test_rl_keeps_teacher_out_of_policy_prompt(record):
    row = rl.preprocess(record)
    assert len(row["messages"]) == len(record["prompt"])
    assert row["messages"][-1]["role"] == "user"
    assert all("loss" not in m for m in row["messages"])
    assert row["reference_completion"] == "ψ(x) = 0"
    assert row["metadata"] == record["metadata"]
    assert "solution" not in row
    assert "reward" not in row
