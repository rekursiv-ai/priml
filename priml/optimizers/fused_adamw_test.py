"""Tests for the fused AdamW step."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import math

from torch import Tensor

import pytest
import torch

from priml.optimizers.fused_adamw import FusedAdamW


def _reference(
    parameter: Tensor,
    gradient: Tensor,
    first: Tensor,
    second: Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> Tensor:
    """One AdamW step written out longhand, in the fused spelling.

    Independent of the implementation under test: it is the arithmetic the
    module claims to perform, so an error in the module cannot hide here.
    """
    decayed: Tensor = parameter.clone() * (1 - lr * weight_decay)
    first = first.lerp(gradient, 1 - beta1)
    second = second.lerp(gradient.square(), 1 - beta2)
    denominator = (second / (1 - math.pow(beta2, step))).sqrt() + eps
    update = first / denominator * (lr / (1 - math.pow(beta1, step)))
    return decayed - update


def test_one_step_matches_the_longhand_arithmetic() -> None:
    """The compiled kernel computes what the module says it computes.

    Compared with a tolerance rather than ``torch.equal``: the reference runs
    the operations one at a time while the kernel is fused, so the two round
    differently in the last bits by construction. Bit equality against the
    REFERENCE IMPLEMENTATION is a different claim, and lives in the parity test
    beside the baseline that makes it.
    """
    torch.manual_seed(0)
    initial = torch.randn(8, 4)
    gradient = torch.randn(8, 4)

    parameter = initial.clone().requires_grad_(True)
    parameter.grad = gradient.clone()
    # Eager: this asserts the arithmetic, which is the same either way, and
    # Dynamo traces the step on first use for 10s that is never cached. What
    # compiling changes is the last bits, which the goldens pin.
    FusedAdamW(
        [parameter],
        lr=0.6,
        betas=(0.8, 0.95),
        eps=1e-10,
        compile=False,
    ).step()

    expected = _reference(
        initial,
        gradient,
        torch.zeros_like(initial),
        torch.zeros_like(initial),
        step=1,
        lr=0.6,
        beta1=0.8,
        beta2=0.95,
        eps=1e-10,
        weight_decay=0.0,
    )
    torch.testing.assert_close(parameter.detach(), expected, rtol=0, atol=1e-6)


def test_the_bias_correction_divides_before_the_root() -> None:
    """The spelling is the point: it must NOT equal torch's split form.

    ``sqrt(v / bias2)`` and ``sqrt(v) / sqrt(bias2)`` agree in exact arithmetic
    and differ in the last bits, so a run is bit-comparable with the reference
    only under the first. If this ever passes, the kernel has been rewritten
    into torch's form and every parity claim resting on it is void.
    """
    torch.manual_seed(0)
    initial = torch.randn(64, 32)
    gradient = torch.randn(64, 32)

    ours = initial.clone().requires_grad_(True)
    ours.grad = gradient.clone()
    FusedAdamW([ours], lr=0.6, betas=(0.8, 0.95), eps=1e-10, compile=False).step()

    theirs = initial.clone().requires_grad_(True)
    theirs.grad = gradient.clone()
    torch.optim.AdamW(
        [theirs],
        lr=0.6,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
    ).step()

    assert not torch.equal(ours.detach(), theirs.detach())
    # Close, though: the two are the same algorithm, so a large gap would mean
    # a real error rather than the rounding this test is about.
    assert torch.allclose(ours.detach(), theirs.detach(), atol=1e-5)


def test_decoupled_decay_shrinks_a_parameter_with_no_gradient() -> None:
    """Decay multiplies the weight, so it applies even at zero gradient."""
    parameter = torch.full((4,), 2.0, requires_grad=True)
    parameter.grad = torch.zeros(4)
    FusedAdamW([parameter], lr=0.1, weight_decay=0.5, compile=False).step()
    # 2 * (1 - 0.1 * 0.5); the update itself is zero because the gradient is.
    assert torch.allclose(parameter.detach(), torch.full((4,), 1.9))


def test_the_schedule_can_move_the_rate_between_steps() -> None:
    """A rate written into the group must reach the next step.

    The scalars are 0-D tensors precisely so a scheduler can change them
    without recompiling; if they were ever baked in, the second step would
    silently reuse the first step's rate.
    """
    torch.manual_seed(0)
    parameter = torch.randn(4, requires_grad=True)
    optimizer = FusedAdamW([parameter], lr=0.1, compile=False)
    parameter.grad = torch.ones(4)
    optimizer.step()

    before = parameter.detach().clone()
    optimizer.param_groups[0]["lr"] = 0.0
    parameter.grad = torch.ones(4)
    optimizer.step()
    assert torch.equal(parameter.detach(), before)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        # A partially-applied CONSTRUCTOR per case, not a kwargs dict: a dict
        # holding values of four different types widens to
        # ``dict[str, object]``, and splatting that loses every argument's type
        # at the call. ``partial`` keeps each keyword checked against the
        # signature, which a lambda would not.
        (partial(FusedAdamW, lr=-1.0), "learning rate"),
        (partial(FusedAdamW, betas=(1.0, 0.9)), "betas"),
        (partial(FusedAdamW, eps=-1e-8), "eps"),
        (partial(FusedAdamW, weight_decay=-0.1), "weight_decay"),
    ],
)
def test_invalid_hyperparameters_are_refused(
    build: Callable[[list[Tensor]], FusedAdamW],
    message: str,
) -> None:
    """Every hyperparameter is bounded at construction, not at the first step."""
    with pytest.raises(ValueError, match=message):
        build([torch.zeros(2, requires_grad=True)])


def test_the_config_builds_a_constructor_awaiting_parameters() -> None:
    """``make()`` yields a constructor: a config tree has no parameters."""
    config = FusedAdamW.Config()
    config.compile = False
    config.lr = 0.02
    config.betas = (0.8, 0.95)
    build = config.make()
    optimizer = build([torch.zeros(2, requires_grad=True)])
    assert isinstance(optimizer, FusedAdamW)
    assert optimizer.param_groups[0]["lr"] == 0.02
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
