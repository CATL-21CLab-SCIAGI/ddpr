"""Test-only optimizer checks for Slime smoke jobs."""

import json
from pathlib import Path

import torch


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    parameters = {
        name: parameter
        for chunk in model
        for name, parameter in chunk.named_parameters()
        if parameter.requires_grad
    }
    # Bound CPU memory when the regular launcher trains the full model.
    before = {
        name: p.detach().reshape(-1)[:4096].cpu().clone()
        for name, p in parameters.items()
    }
    original_step = optimizer.step

    def checked_step(*positional, **keywords):
        try:
            gradients = {}
            for name, parameter in parameters.items():
                gradient = getattr(parameter, "main_grad", parameter.grad)
                gradients[name] = (
                    0 if gradient is None else gradient.detach().count_nonzero().item()
                )
            result = original_step(*positional, **keywords)
            torch.cuda.synchronize()
            changes = {
                name: (p.detach().reshape(-1)[:4096].cpu() != before[name])
                .count_nonzero()
                .item()
                for name, p in parameters.items()
            }
            rank = torch.distributed.get_rank()
            record = {
                "rank": rank,
                "rollout": rollout_id,
                "step": step_id,
                "optimizer_success": bool(result[0]),
                "gradient_nonzero": gradients,
                "changed_values": changes,
                "checked_values": {name: p.numel() for name, p in before.items()},
            }
            output = Path(args.save) / "checks"
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"rollout-{rollout_id}-step-{step_id}-rank-{rank}.json"
            path.write_text(json.dumps(record, indent=2))
            assert record["optimizer_success"], record
            assert any(gradients.values()), record
            assert any(changes.values()), record
            return result
        finally:
            optimizer.step = original_step

    optimizer.step = checked_step
