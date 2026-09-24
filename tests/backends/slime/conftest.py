import json
from types import SimpleNamespace

import pytest


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return (
            "\n".join(m["role"] + ": " + m["content"] for m in messages)
            + "\nassistant:"
        )

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return list(prompt.encode())


@pytest.fixture
def record():
    return {
        "id": "physics:1",
        "prompt": [
            {"role": "user", "content": "Find energy."},
            {"role": "assistant", "content": "Earlier derivation."},
            {"role": "user", "content": "Give the answer."},
        ],
        "completion": [{"role": "assistant", "content": "TEACHER: E = mv²/2."}],
        "metadata": {"benchmark": "cmphysbench", "quality": {"verified": False}},
    }


@pytest.fixture
def args(tmp_path, monkeypatch, record):
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps(record) + "\n")
    pytest.importorskip("slime.rollout.data_source")
    from ddpr.backends.slime import rl

    monkeypatch.setattr(rl, "load_tokenizer", lambda *a, **kw: Tokenizer())
    return SimpleNamespace(
        prompt_data=str(path),
        hf_checkpoint="student",
        rollout_global_dataset=True,
        input_key="prompt",
        label_key="completion",
        metadata_key="metadata",
        apply_chat_template=True,
        apply_chat_template_kwargs={},
        rollout_max_prompt_len=1024,
        loss_type="policy_loss",
        compute_advantages_and_returns=True,
        n_samples_per_prompt=2,
        advantage_estimator="grpo",
        custom_rm_path="reward.score",
        rollout_seed=42,
        rollout_shuffle=True,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
        dump_details=None,
    )
