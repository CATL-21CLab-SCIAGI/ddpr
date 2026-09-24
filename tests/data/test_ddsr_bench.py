from copy import deepcopy

import pytest

from ddpr.data.adapters.ddsr_bench import load_sample


def test_record_preserves_text_and_metadata_without_mutation():
    record = {
        "id": "trial:formatting",
        "prompt": [{"role": "user", "content": "Find ψ(x)."}],
        "completion": [{"role": "assistant", "content": "  ψ(x) = 0\n"}],
        "metadata": {"benchmark": "critpt", "quality": {"verified": False}},
    }
    original = deepcopy(record)
    example = load_sample(record)
    assert example.completion[0]["content"] == "  ψ(x) = 0\n"
    assert example.id == record["id"]
    assert example.metadata == record["metadata"]
    example.prompt[0]["content"] = "changed"
    assert record == original


def test_loader_fields_have_no_id():
    sample = load_sample(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "completion": [{"role": "assistant", "content": "answer"}],
        }
    )
    assert sample.id is None
    assert sample.metadata == {}


def test_invalid_record_id():
    with pytest.raises(TypeError, match="sample ID"):
        load_sample(
            {
                "id": 123,
                "prompt": [{"role": "user", "content": "question"}],
                "completion": [{"role": "assistant", "content": "answer"}],
            }
        )


@pytest.mark.parametrize(
    "prompt,completion,metadata,match",
    [
        ("rendered", [], {}, "prompt"),
        ([], [], {}, "prompt"),
        ([{}], [], {}, "role"),
        ([{"role": "user", "content": []}], [], {}, "text"),
        ([{"role": "tool", "content": "result"}], [], {}, "role"),
        ([{"role": "user", "content": "q"}], None, {}, "completion"),
        (
            [{"role": "user", "content": "q"}],
            [{"role": "user", "content": "a"}],
            {},
            "assistant",
        ),
        (
            [{"role": "user", "content": "q"}],
            [{"role": "assistant", "content": "a"}],
            [],
            "metadata",
        ),
    ],
)
def test_invalid_fields(prompt, completion, metadata, match):
    with pytest.raises((TypeError, ValueError), match=match):
        load_sample({"prompt": prompt, "completion": completion, "metadata": metadata})


def test_empty_text_is_preserved_for_task_specific_selection():
    sample = load_sample(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "completion": [{"role": "assistant", "content": " \n"}],
        }
    )
    assert sample.completion[0]["content"] == " \n"
