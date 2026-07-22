"""Regression tests for Newton optimizer dtype handling (LOSSOPT-005)."""

from __future__ import annotations

import torch

from priml.optimizers.newton import Newton


def test_newton_fp64_quadratic_converges() -> None:
    """fp64 params: Newton must converge to the minimum, not fall back to GD.

    For f(x) = (x - 3)^2 the Newton step reaches the exact minimum in one
    iteration. A float32/float64 dtype mismatch in the Hessian made
    ``torch.linalg.solve`` raise, silently falling back to gradient descent.
    """
    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = Newton(lr=1.0, damping=0.0)([param])

    def closure() -> torch.Tensor:
        return ((param - 3.0) ** 2).sum()

    opt.step(closure)

    torch.testing.assert_close(param.detach(), torch.tensor([3.0], dtype=torch.float64))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
