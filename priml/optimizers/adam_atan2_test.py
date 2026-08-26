"""Tests for AdamATan2 parity with the reference package."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

from torch import Tensor

import torch

from priml.optimizers.adam_atan2 import AdamATan2


_GOLDEN = Path(__file__).parent / "testdata" / "adam_atan2_0_0_3.pt"


class _AdamATan2Golden(TypedDict):
    initial_param: Tensor
    grads: Tensor
    expected_param: Tensor


def test_adam_atan2_matches_reference_bias_corrections() -> None:
    param = torch.tensor([0.5, -0.25, 0.125], dtype=torch.float64)
    expected = param.clone()
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    opt = AdamATan2([param], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    grads = [
        torch.tensor([0.01, -0.02, 0.04], dtype=torch.float64),
        torch.tensor([0.03, -0.01, -0.02], dtype=torch.float64),
        torch.tensor([-0.02, 0.05, 0.01], dtype=torch.float64),
    ]
    for step, grad in enumerate(grads, start=1):
        param.grad = grad.clone()
        expected, exp_avg, exp_avg_sq = _reference_step(
            expected,
            grad,
            exp_avg,
            exp_avg_sq,
            step=step,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
        )
        opt.step()
        torch.testing.assert_close(param, expected, rtol=0, atol=1e-15)

    state = opt.state[param]
    # ``step`` is intentionally a Python int (not a 0-dim Tensor) to avoid
    # per-step GPU<->CPU syncs; see adam_atan2.py for the rationale.
    assert isinstance(state["step"], int)
    assert state["step"] == len(grads)
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


def test_adam_atan2_matches_external_package_golden() -> None:
    """Replay the ``adam-atan2==0.0.3`` package's reference oracle exactly."""
    golden = cast(
        _AdamATan2Golden,
        torch.load(_GOLDEN, weights_only=True, map_location="cpu"),
    )
    param = golden["initial_param"].clone()
    opt = AdamATan2(
        [param],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    for grad in golden["grads"]:
        param.grad = grad.clone()
        opt.step()
    assert torch.equal(param, golden["expected_param"])


def _reference_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    param = param.clone()
    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()

    param.mul_(1.0 - lr * weight_decay)
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    step_size = lr / (1.0 - beta1**step)
    bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
    denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
    param.add_(torch.atan2(exp_avg, denom), alpha=-step_size)
    return param, exp_avg, exp_avg_sq
