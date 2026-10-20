import asyncio
import pickle
import threading
from types import SimpleNamespace

from ddpr.backends.slime import reward, train


def test_training_passes_reward_in_serializable_arguments(monkeypatch, tmp_path):
    (tmp_path / "train.py").write_text(
        "from argparse import ArgumentParser\n"
        "def parse_args(add_custom_arguments):\n"
        "    return add_custom_arguments(ArgumentParser()).parse_args()\n"
        "def train(args):\n"
        '    assert args.ddpr_reward == "ddpr.rewards.SmokeReward"\n'
    )
    monkeypatch.setenv("DDPR_SLIME_ROOT", str(tmp_path))
    monkeypatch.setenv("DDPR_REWARD", "stale.Reward")
    monkeypatch.setattr(
        "sys.argv", ["train", "--ddpr-reward", "ddpr.rewards.SmokeReward"]
    )
    import sys

    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=lambda **kwargs: captured.update(kwargs)),
    )
    monkeypatch.setenv("AUTO_CRITPT_GRADER_JUDGE_API_KEY_ENV", "TEST_JUDGE_KEY")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-only-secret")
    monkeypatch.setenv("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE", "false")
    monkeypatch.setenv("UNRELATED_SECRET", "not-forwarded")
    train.main()
    env = captured["runtime_env"]["env_vars"]
    assert env["TEST_JUDGE_KEY"] == "test-only-secret"
    assert env["DDPR_CRITPT_AI_EQUIVALENCE_JUDGE"] == "false"
    assert "UNRELATED_SECRET" not in env
    args = pickle.loads(
        pickle.dumps(SimpleNamespace(ddpr_reward="ddpr.rewards.SmokeReward"))
    )
    monkeypatch.delenv("DDPR_REWARD")
    sample = SimpleNamespace(
        response="text", sample_id="id", reference_completion="ref", metadata={}
    )
    assert isinstance(asyncio.run(reward.reward(args, sample)), float)


def test_concurrent_and_cancelled_calls_do_not_overlap(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def scorer(**fields):
        calls.append(fields["response"])
        if fields["response"] == "first":
            entered.set()
            assert release.wait(3)
        return 1.0

    monkeypatch.setattr(reward, "load_reward", lambda path: scorer)
    args = SimpleNamespace(ddpr_reward="test.Reward")

    def sample(response):
        return SimpleNamespace(
            response=response, sample_id="id", reference_completion="ref", metadata={}
        )

    async def run():
        first = asyncio.create_task(reward.reward(args, sample("first")))
        assert await asyncio.to_thread(entered.wait, 3)
        first.cancel()
        second = asyncio.create_task(reward.reward(args, sample("second")))
        try:
            await asyncio.sleep(0.05)
            assert calls == ["first"]
        finally:
            release.set()
        await asyncio.gather(first, return_exceptions=True)
        assert await second == 1.0
        assert calls == ["first", "second"]

    asyncio.run(run())


def test_live_ray_reward_environment(monkeypatch):
    import os

    import pytest

    if os.getenv("DDPR_TEST_RAY") != "1":
        pytest.skip("set DDPR_TEST_RAY=1 for a local Ray worker check")
    import ray

    # Forward only test values; do not send ambient credentials to this test job.
    for key in train._reward_environment():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DDPR_CRITPT_AI_EQUIVALENCE_JUDGE", "false")
    monkeypatch.setenv("AUTO_CRITPT_GRADER_JUDGE_API_KEY_ENV", "TEST_JUDGE_KEY")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-only")
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        runtime_env={"env_vars": train._reward_environment()},
    )
    try:

        @ray.remote
        def inspect_reward():
            from ddpr.rewards.critpt_ai import CritptAiReward

            return CritptAiReward().llm_fallback, os.environ["TEST_JUDGE_KEY"]

        # Slime actors supply their own environment, which must retain job values.
        assert ray.get(
            inspect_reward.options(
                runtime_env={"env_vars": {"LOCAL_FLAG": "1"}}
            ).remote()
        ) == (False, "test-only")
    finally:
        ray.shutdown()
