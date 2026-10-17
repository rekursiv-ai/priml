"""Tests for the ``dl-activation-function`` activations, against their closed forms."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.mlsbench.dl.activation.activations import gelu, mish
from priml.cost import cost


def test_gelu_is_x_times_the_exact_normal_cdf() -> None:
    x = torch.tensor([-2.0, 0.0, 0.5, 3.0], dtype=torch.float64)
    expected = x * 0.5 * (1 + torch.erf(x / 2**0.5))
    torch.testing.assert_close(gelu(x), expected, rtol=1e-12, atol=0.0)


def test_mish_is_x_times_tanh_of_softplus() -> None:
    x = torch.tensor([-3.0, 0.0, 1.0, 30.0], dtype=torch.float64)
    expected = x * torch.tanh(torch.log1p(torch.exp(x)))
    torch.testing.assert_close(mish(x), expected, rtol=1e-12, atol=0.0)


def test_mish_gradient_matches_its_closed_form() -> None:
    x = torch.tensor([-1.5, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    mish(x).sum().backward()
    assert x.grad is not None
    sp = torch.nn.functional.softplus(x.detach())
    expected = torch.tanh(sp) + x.detach() * (1 - torch.tanh(sp) ** 2) * torch.sigmoid(
        x.detach(),
    )
    torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("activation", [gelu, mish], ids=["gelu", "mish"])
def test_each_activation_is_costed_per_element(activation: object) -> None:
    two, three = (
        cost(activation, channels=n, dtype=None)["flops"].cells for n in (2, 3)
    )
    assert {k: 3 * v for k, v in two.items()} == {k: 2 * v for k, v in three.items()}
    assert sum(two.values()) > 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
