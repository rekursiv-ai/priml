"""Tests for the activations that torch does not ship."""

from __future__ import annotations

import torch

from priml.math.activations import relu_squared


def test_the_negative_half_is_exactly_zero() -> None:
    """Rectified, not merely squared: ``x ** 2`` is even, so it would map -3
    and 3 to the same value and discard the sign the layer needs.
    """
    x = torch.tensor([-3.0, -1e-9, 0.0])
    assert torch.equal(relu_squared(x), torch.zeros(3))


def test_the_positive_half_is_the_square() -> None:
    x = torch.tensor([1e-9, 1.0, 3.0])
    torch.testing.assert_close(relu_squared(x), x.square(), rtol=0, atol=0)


def test_the_gradient_is_a_rectifier() -> None:
    """``d/dx relu(x)**2 == 2 * relu(x)`` -- continuous at the origin, where
    ``relu``'s own gradient jumps from 0 to 1. That continuity is the whole
    reason to prefer this over a plain ReLU.
    """
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    relu_squared(x).sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, 2 * torch.relu(x.detach()), rtol=0, atol=0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
