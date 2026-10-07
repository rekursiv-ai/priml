"""Tests for optimizer learning-rate utilities."""

from __future__ import annotations

from torch import nn

import pytest
import torch

from priml.lib.codec import ReadError
from priml.optimizers.lr import (
    apply_lr_scale,
    clip_grad_norm,
    learning_rate,
    lr_scale,
    remember_initial_lrs,
    step_optimizers,
    zero_optimizers,
)


def test_lr_scale_warms_up_then_cosine_decays_and_clamps() -> None:
    assert lr_scale(0, 10, warmup_steps=2, min_ratio=0.1) == 0.0
    assert lr_scale(1, 10, warmup_steps=2, min_ratio=0.1) == 0.5
    assert lr_scale(2, 10, warmup_steps=2, min_ratio=0.1) == 1.0
    assert lr_scale(6, 10, warmup_steps=2, min_ratio=0.1) == pytest.approx(0.55)
    assert lr_scale(10, 10, warmup_steps=2, min_ratio=0.1) == 0.1
    assert lr_scale(12, 10, warmup_steps=2, min_ratio=0.1) == 0.1
    assert lr_scale(0, 0) == 1.0
    assert lr_scale(0, 0, warmup_steps=1) == 0.0
    assert lr_scale(-1, 4, warmup_steps=1) == -1.0
    assert lr_scale(3, 3, warmup_steps=2) == 0.0


def test_learning_rate_helpers_preserve_initial_rate_and_scale_groups() -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter], "lr": 0.2},
            {"params": [nn.Parameter(torch.ones(3))], "lr": 0.4},
        ],
    )
    remember_initial_lrs([optimizer])
    optimizer.param_groups[0]["lr"] = 9.0
    optimizer.param_groups[1]["initial_lr"] = 0.8

    apply_lr_scale([optimizer], 0.25)

    assert [group["lr"] for group in optimizer.param_groups] == [0.05, 0.2]
    assert learning_rate(optimizer) == 0.05
    assert learning_rate(optimizer, group_index=1) == 0.2
    remember_initial_lrs([optimizer])
    assert [group["initial_lr"] for group in optimizer.param_groups] == [0.2, 0.8]


def test_learning_rate_requires_initial_rate_before_scaling() -> None:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.3)
    with pytest.raises(KeyError, match="initial_lr"):
        apply_lr_scale([optimizer], 0.5)


def test_learning_rate_rejects_a_non_numeric_group_value() -> None:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(2))], lr=0.3)
    optimizer.param_groups[0]["lr"] = "invalid"

    with pytest.raises(ReadError):
        learning_rate(optimizer)


def test_optimizer_helpers_step_zero_and_clip_gradients() -> None:
    parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([3.0, 4.0])

    assert clip_grad_norm([parameter], None) is None
    assert clip_grad_norm([parameter], 2.5) == pytest.approx(5.0)
    assert torch.linalg.vector_norm(parameter.grad) == pytest.approx(2.5)
    step_optimizers([optimizer])
    assert torch.allclose(parameter, torch.tensor([2.85, 3.8]))
    zero_optimizers([optimizer])
    assert parameter.grad is None
    parameter.grad = torch.ones_like(parameter)
    zero_optimizers([optimizer], set_to_none=False)
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
