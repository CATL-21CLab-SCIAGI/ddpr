import pytest

from ddpr.backends.swift import rl, sft


@pytest.mark.parametrize("preprocess", [sft.preprocess, rl.preprocess])
def test_empty_completion_is_skipped(record, preprocess):
    record["completion"][0]["content"] = " "
    assert preprocess(record) is None


@pytest.mark.parametrize("preprocess", [sft.preprocess, rl.preprocess])
def test_prompt_must_end_with_user(record, preprocess):
    record["prompt"].pop()
    with pytest.raises(ValueError, match="end with a user"):
        preprocess(record)


@pytest.mark.parametrize("field", ["tools", "images", "videos", "audios"])
def test_unsupported_inputs(record, field):
    record[field] = ["unsupported"]
    with pytest.raises(ValueError, match="text conversations"):
        sft.preprocess(record)
