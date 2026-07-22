"""Tests for training optimizer helpers."""

from __future__ import annotations

import torch

from priml.optimizers import (
    apply_lr_scale,
    clip_grad_norm,
    lr_scale,
    remember_initial_lrs,
    step_optimizers,
    zero_optimizers,
)


def test_lr_scale_clamps_progress_past_total() -> None:
    """Past ``total_steps`` the scale must stay at ``min_ratio``, not cycle.

    Without clamping progress to 1, ``cos(pi * progress)`` keeps cycling: at
    ``step == 2 * total_steps`` it returns to its maximum (full LR), and
    intermediate over-runs land between min_ratio and 1.0. A decayed schedule
    must be monotone non-increasing and flat at the floor once training ends.
    """
    min_ratio = 0.1
    assert lr_scale(100, 100, min_ratio=min_ratio) == min_ratio
    assert lr_scale(150, 100, min_ratio=min_ratio) == min_ratio
    assert lr_scale(200, 100, min_ratio=min_ratio) == min_ratio
    assert lr_scale(1000, 100, min_ratio=min_ratio) == min_ratio


def test_remember_initial_lrs_preserves_original_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.2)

    remember_initial_lrs([optimizer])
    optimizer.param_groups[0]["lr"] = 0.1
    remember_initial_lrs([optimizer])

    assert optimizer.param_groups[0]["initial_lr"] == 0.2


def test_apply_lr_scale_uses_initial_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.2)

    remember_initial_lrs([optimizer])
    apply_lr_scale([optimizer], 0.25)

    assert optimizer.param_groups[0]["lr"] == 0.05


def test_step_and_zero_multiple_optimizers() -> None:
    param_a = torch.nn.Parameter(torch.tensor([1.0]))
    param_b = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer_a = torch.optim.SGD([param_a], lr=0.1)
    optimizer_b = torch.optim.SGD([param_b], lr=0.1)
    param_a.grad = torch.tensor([1.0])
    param_b.grad = torch.tensor([2.0])

    step_optimizers([optimizer_a, optimizer_b])
    zero_optimizers([optimizer_a, optimizer_b], set_to_none=True)

    torch.testing.assert_close(param_a, torch.tensor([0.9]))
    torch.testing.assert_close(param_b, torch.tensor([1.8]))
    assert param_a.grad is None
    assert param_b.grad is None


def test_clip_grad_norm_can_be_disabled() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])

    grad_norm = clip_grad_norm([param], None)

    assert grad_norm is None
    torch.testing.assert_close(param.grad, torch.tensor([2.0]))


def test_clip_grad_norm_returns_norm() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])

    grad_norm = clip_grad_norm([param], 1.0)

    assert grad_norm is not None
    torch.testing.assert_close(grad_norm, torch.tensor(2.0))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
