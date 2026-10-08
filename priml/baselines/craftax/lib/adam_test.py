"""Unit tests for Adam behind a global gradient-norm clip, on the CPU."""

from __future__ import annotations

import math

import pytest
import torch

from priml.baselines.craftax.lib.adam import ClippedAdam


@pytest.mark.parametrize(("scale", "norm"), [(10.0, 1.0), (0.01, None)])
def test_a_step_is_the_global_clip_then_adams_step(
    scale: float,
    norm: float | None,
) -> None:
    """Above the limit the gradients shrink to it together; below it they stay."""
    generator = torch.Generator().manual_seed(0)
    weights = [
        torch.randn(4, 3, generator=generator),
        torch.randn(5, generator=generator),
    ]
    gradients = [scale * torch.randn(w.shape, generator=generator) for w in weights]
    clipped = [torch.nn.Parameter(w.clone()) for w in weights]
    plain = [torch.nn.Parameter(w.clone()) for w in weights]
    for parameters in (clipped, plain):
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient.clone()
    ClippedAdam(clipped, lr=0.1, max_grad_norm=1.0, eps=1e-5).step()
    torch.nn.utils.clip_grad_norm_(plain, 1.0)
    torch.optim.Adam(plain, lr=0.1, eps=1e-5).step()
    for a, b in zip(clipped, plain, strict=True):
        assert torch.equal(a, b)
    after = float(torch.nn.utils.get_total_norm(_gradients(clipped)))
    before = float(torch.nn.utils.get_total_norm(gradients))
    assert after == pytest.approx(before if norm is None else norm, rel=1e-5)


def test_a_rate_set_as_a_tensor_is_read_at_each_step() -> None:
    """A train step replaces each group's rate with a tensor it refills per epoch."""
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = ClippedAdam([parameter], lr=0.1, max_grad_norm=1.0)
    rate = torch.tensor(0.0)
    optimizer.param_groups[0]["lr"] = rate
    parameter.grad = torch.ones(3)
    optimizer.step()
    assert not parameter.any()
    rate.fill_(0.5)
    optimizer.step()
    assert parameter.ne(0).all()


def test_on_the_cpu_the_step_is_not_capturable() -> None:
    """Torch's Adam refuses capturable parameters off an accelerator."""
    optimizer = ClippedAdam(
        [torch.nn.Parameter(torch.zeros(2))],
        lr=0.1,
        max_grad_norm=1.0,
    )
    assert not optimizer.param_groups[0]["capturable"]


def test_a_closure_is_evaluated_before_the_clip() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = ClippedAdam([parameter], lr=0.1, max_grad_norm=1.0)

    def closure() -> float:
        optimizer.zero_grad()
        loss = (100.0 * parameter - 1.0).square().sum()
        loss.backward()
        return float(loss.detach())

    assert optimizer.step(closure) == 2.0
    assert parameter.grad is not None
    assert float(parameter.grad.norm()) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_clip_that_is_not_positive_and_finite_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="max_grad_norm"):
        ClippedAdam([torch.nn.Parameter(torch.zeros(2))], lr=0.1, max_grad_norm=value)


def _gradients(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    gradients: list[torch.Tensor] = []
    for parameter in parameters:
        assert parameter.grad is not None
        gradients.append(parameter.grad)
    return gradients


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
