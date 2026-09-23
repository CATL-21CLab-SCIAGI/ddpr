import pytest


@pytest.fixture
def record():
    return {
        "id": "physics:format",
        "prompt": [
            {"role": "system", "content": "Solve physics problems."},
            {"role": "user", "content": "Find ψ(x)."},
            {"role": "assistant", "content": "Earlier derivation", "loss": True},
            {"role": "user", "content": "Give the final expression."},
        ],
        "completion": [{"role": "assistant", "content": "ψ(x) = 0", "loss": False}],
        "metadata": {"benchmark": "cmphysbench", "quality": {"verified": False}},
    }
