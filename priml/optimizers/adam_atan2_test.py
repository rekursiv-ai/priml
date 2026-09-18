"""Tests for AdamATan2 parity with the reference package."""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict, cast

from torch import Tensor

import pytest
import torch

from priml.optimizers.adam_atan2 import AdamATan2


_CWD: Final = Path(__file__).resolve().parent


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

    state = cast(dict[str, object], opt.state[param])
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
        torch.load(
            _CWD / "testdata" / "adam_atan2_0_0_3.pt",
            weights_only=True,
            map_location="cpu",
        ),
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


def test_config_builds_a_constructor_awaiting_parameters() -> None:
    build = AdamATan2.Config(lr=0.5, betas=(0.8, 0.9), weight_decay=0.0).make()
    optimizer = build([torch.zeros(2)])
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.5
    assert group["betas"] == (0.8, 0.9)
    assert group["weight_decay"] == 0.0


def test_finalized_config_is_not_finalized_twice() -> None:
    optimizer = AdamATan2.Config(lr=0.5).finalize().make()([torch.zeros(2)])
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_invalid_hyperparameters_are_rejected() -> None:
    p = torch.zeros(2)
    with pytest.raises(ValueError, match="Invalid learning rate"):
        AdamATan2([p], lr=-1.0)
    with pytest.raises(ValueError, match="Invalid betas"):
        AdamATan2([p], betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        AdamATan2([p], weight_decay=-1.0)


def test_step_evaluates_and_returns_the_closure() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)
    calls: list[int] = []

    def closure() -> float:
        calls.append(1)
        return 3.0

    assert AdamATan2([p]).step(closure) == 3.0
    assert calls == [1]


def test_params_without_gradient_are_skipped() -> None:
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = AdamATan2([p], weight_decay=0.5)
    optimizer.step()
    torch.testing.assert_close(p, torch.ones(2))
    assert len(optimizer.state) == 0


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
