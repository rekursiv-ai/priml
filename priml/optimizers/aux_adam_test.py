"""Tests for group-stepped AdamW used by sparse optimizer partitions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.optimizers.aux_adam import AuxAdam
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable


def test_aux_adam_updates_gradient_parameters_and_advances_group() -> None:
    """Update moments while the group counter follows every optimizer call."""
    parameter = nn.Parameter(torch.tensor([0.5, -1.25]))
    initial = parameter.detach().clone()
    optimizer = AuxAdam.Config(
        lr=0.0007,
        betas=(0.72, 0.91),
        eps=1e-7,
        weight_decay=0.13,
    ).make()([parameter])

    with host_agnostic_numerics():
        for gradient in (
            torch.tensor([0.375, -0.625]),
            None,
            torch.tensor([1.125, -1.875]),
        ):
            parameter.grad = gradient
            optimizer.step()

    assert optimizer.param_groups[0]["step"] == 3
    assert parameter not in optimizer.state or optimizer.state[parameter]
    assert not torch.equal(parameter, initial)


def test_aux_adam_increments_the_group_without_any_gradient() -> None:
    """Keep group schedule state aligned on steps with absent gradients."""
    parameter = nn.Parameter(torch.tensor([0.5, -1.25]))
    optimizer = AuxAdam.Config().make()([parameter])
    initial = parameter.detach().clone()

    optimizer.step()

    group = optimizer.param_groups[0]
    assert group["step"] == 1
    assert parameter not in optimizer.state
    assert torch.equal(parameter, initial)


def test_aux_adam_defaults_keep_the_released_auxiliary_recipe() -> None:
    """Keep the default rate, moments, epsilon, and decay explicit."""
    config = AuxAdam.Config()
    optimizer = AuxAdam([nn.Parameter(torch.ones(2))])
    group = optimizer.param_groups[0]

    assert config.lr == 1e-4
    assert config.betas == (0.9, 0.95)
    assert config.eps == 1e-8
    assert config.weight_decay == 0.1
    assert group["lr"] == config.lr
    assert group["betas"] == config.betas
    assert group["eps"] == config.eps
    assert group["weight_decay"] == config.weight_decay


def test_aux_adam_closure_runs_once_with_grad_enabled() -> None:
    """Run an optimizer closure once and preserve its loss value."""
    parameter = nn.Parameter(torch.tensor([1.5, -2.0]))
    optimizer = AuxAdam.Config().make()([parameter])
    calls = 0

    def closure() -> Tensor:
        nonlocal calls
        calls += 1
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert calls == 1
    assert isinstance(loss, Tensor)
    assert loss.item() == 6.25
    assert optimizer.param_groups[0]["step"] == 1


@pytest.mark.parametrize(
    "build",
    [
        partial(AuxAdam, lr=-1.0),
        partial(AuxAdam, betas=(1.0, 0.9)),
        partial(AuxAdam, eps=-1e-8),
        partial(AuxAdam, weight_decay=-0.1),
    ],
)
def test_aux_adam_rejects_invalid_hyperparameters(
    build: Callable[..., AuxAdam],
) -> None:
    """Reject invalid rates, moment coefficients, epsilon, and decay."""
    with pytest.raises(ValueError, match=r".+"):
        build([nn.Parameter(torch.ones(2))])


def test_aux_adam_rejects_sparse_gradients() -> None:
    """Reject sparse gradients before dense moment arithmetic."""
    parameter = nn.Parameter(torch.ones(3))
    optimizer = AuxAdam.Config().make()([parameter])
    parameter.grad = torch.ones(3).to_sparse()

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optimizer.step()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
