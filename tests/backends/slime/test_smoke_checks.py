import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import smoke_checks

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return smoke_checks


@pytest.mark.parametrize(
    "gradient,update", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("size", [1, 4097])
def test_rank_must_have_gradients_and_update(smoke, tmp_path, gradient, update, size):
    model = torch.nn.Linear(size, 1, bias=False)
    model.weight.main_grad = torch.ones_like(model.weight) if gradient else None

    def step():
        if update:
            with torch.no_grad():
                model.weight.add_(1)
        return True, 1.0, 0

    optimizer = SimpleNamespace(step=step)
    smoke.before_step(SimpleNamespace(save=tmp_path), 0, 0, [model], optimizer, None)
    if gradient and update:
        assert optimizer.step()[0]
    else:
        with pytest.raises(AssertionError):
            optimizer.step()
    assert optimizer.step is step
    record = json.loads((tmp_path / "checks/rollout-0-step-0-rank-0.json").read_text())
    assert record["gradient_nonzero"]["weight"] == size * int(gradient)
    assert record["changed_values"]["weight"] == min(size, 4096) * int(update)
    assert record["checked_values"]["weight"] == min(size, 4096)
