import pytest

from ddpr.rewards import Reward


def test_reward_requires_a_concrete_verifier():
    with pytest.raises(TypeError):
        Reward()


def test_reward_receives_response_and_reference_separately():
    class TestReward(Reward):
        def __call__(self, *, response, sample_id, reference_completion, metadata):
            assert response == "Generated answer"
            assert sample_id == "physics:1"
            assert reference_completion == "Unverified teacher answer"
            assert metadata == {"benchmark": "test"}
            return 0.5

    assert (
        TestReward()(
            response="Generated answer",
            sample_id="physics:1",
            reference_completion="Unverified teacher answer",
            metadata={"benchmark": "test"},
        )
        == 0.5
    )


def test_smoke_reward_is_deterministic_and_not_reference_matching():
    from ddpr.rewards import SmokeReward, load_reward

    reward = load_reward("ddpr.rewards.SmokeReward")
    assert isinstance(reward, SmokeReward)
    fields = {"sample_id": "test", "reference_completion": "teacher", "metadata": {}}
    score = reward(response="answer", **fields)
    assert 0 <= score <= 1
    assert score == reward(response="answer", **fields)
    assert score != reward(response="another answer", **fields)
    assert score == reward(
        response="answer", sample_id="other", reference_completion="answer", metadata={}
    )


def test_reward_loader_rejects_non_reward():
    from ddpr.rewards import load_reward

    with pytest.raises(TypeError):
        load_reward("builtins.str")
    with pytest.raises(ValueError):
        load_reward("MissingModule")


def test_backend_adapters_preserve_fields_and_order(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from ddpr.backends.slime import reward as slime

    swift = pytest.importorskip("ddpr.backends.swift.reward")
    calls = []

    class Verifier(Reward):
        def __call__(self, **fields):
            calls.append(fields)
            return float(fields["metadata"]["score"])

    monkeypatch.setenv("DDPR_REWARD", "test.Verifier")
    monkeypatch.setattr(slime, "load_reward", lambda path: Verifier())
    monkeypatch.setattr(swift, "load_reward", lambda path: Verifier())
    fields = [
        {
            "response": "generated",
            "sample_id": str(i),
            "reference_completion": "teacher",
            "metadata": {"score": i},
        }
        for i in (1, 0)
    ]
    samples = [SimpleNamespace(**row) for row in fields]
    assert asyncio.run(
        slime.reward(SimpleNamespace(ddpr_reward="test.Verifier"), samples)
    ) == [1.0, 0.0]
    assert calls == fields
    calls.clear()
    assert (
        asyncio.run(
            slime.reward(SimpleNamespace(ddpr_reward="test.Verifier"), samples[0])
        )
        == 1.0
    )
    assert calls == fields[:1]
    calls.clear()
    adapter = swift.RewardAdapter()
    assert adapter(
        completions=[row["response"] for row in fields],
        sample_id=[row["sample_id"] for row in fields],
        reference_completion=[row["reference_completion"] for row in fields],
        metadata=[row["metadata"] for row in fields],
    ) == [1.0, 0.0]
    assert calls == fields
