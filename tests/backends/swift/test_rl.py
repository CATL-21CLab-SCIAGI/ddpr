from ddpr.backends.swift import rl


def test_rl_keeps_teacher_out_of_policy_prompt(record):
    row = rl.preprocess(record)
    assert len(row["messages"]) == len(record["prompt"])
    assert row["messages"][-1]["role"] == "user"
    assert all("loss" not in m for m in row["messages"])
    assert row["reference_completion"] == "ψ(x) = 0"
    assert row["metadata"] == record["metadata"]
    assert "solution" not in row
    assert "reward" not in row
