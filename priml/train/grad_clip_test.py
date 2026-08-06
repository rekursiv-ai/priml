"""Tests for global gradient-norm clipping.

The distributed (DTensor/mesh) reduction needs a multi-rank process group and
is exercised by the training integration tests; here we verify the
single-process math: the global norm spans all parameters, clipping rescales
in-place to the cap, a norm under the cap is left untouched, and the
measure-only path reports the same norm without mutating grads.
"""

from __future__ import annotations

import torch

from priml.train.grad_clip import (
    clip_grad_norm_,
    total_grad_norm,
    total_param_norm,
)


def _test_device() -> torch.device:
    """The device these single-process tests run on (MPS on Apple silicon)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# MPS has no fused foreach kernels (``foreach=True`` raises there), so let
# PyTorch pick the per-device path when the tests run on MPS; keep the fast
# ``True`` path on CPU/CUDA. This is exactly the knob callers set per backend
# now that grad_clip no longer auto-sniffs the device.
_FOREACH: bool | None = None if torch.backends.mps.is_available() else True


def _params_with_grads(grads: list[torch.Tensor]) -> list[torch.nn.Parameter]:
    device = _test_device()
    params: list[torch.nn.Parameter] = []
    for g in grads:
        g = g.to(device)  # noqa: PLW2901 -- move the fixture grad onto the test device
        p = torch.nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        params.append(p)
    return params


def _close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare on CPU so a result on the MPS test device matches a CPU literal."""
    torch.testing.assert_close(actual.cpu(), expected)


def test_total_norm_spans_all_parameters() -> None:
    """The norm equals the 2-norm of all grads concatenated."""
    a = torch.tensor([3.0, 0.0])
    b = torch.tensor([0.0, 4.0])
    params = _params_with_grads([a, b])
    # sqrt(3^2 + 4^2) = 5
    norm = total_grad_norm(params, foreach=_FOREACH)
    _close(norm, torch.tensor(5.0))


def test_clip_rescales_to_cap_when_norm_exceeds() -> None:
    """A norm above max_norm is rescaled so the post-clip norm equals the cap."""
    params = _params_with_grads([torch.tensor([3.0, 4.0])])  # norm 5
    returned = clip_grad_norm_(params, max_norm=1.0, foreach=_FOREACH)
    # Returned norm is the PRE-clip norm.
    _close(returned, torch.tensor(5.0))
    # Grads rescaled by 1/5 -> [0.6, 0.8], norm 1.0.
    clipped = params[0].grad
    assert clipped is not None
    _close(clipped, torch.tensor([0.6, 0.8]))
    _close(total_grad_norm(params, foreach=_FOREACH), torch.tensor(1.0))


def test_clip_leaves_grads_untouched_when_norm_within_cap() -> None:
    """A norm at or below max_norm does not rescale grads."""
    params = _params_with_grads([torch.tensor([0.3, 0.4])])  # norm 0.5
    grad = params[0].grad
    assert grad is not None
    before = grad.clone()
    clip_grad_norm_(params, max_norm=1.0, foreach=_FOREACH)
    _close(grad, before.cpu())  # clip mutates grad in place; grad is params[0].grad


def test_total_grad_norm_does_not_mutate_grads() -> None:
    """Measure-only path reports the norm and leaves grads unchanged."""
    params = _params_with_grads([torch.tensor([3.0, 4.0])])
    grad = params[0].grad
    assert grad is not None
    before = grad.clone()
    norm = total_grad_norm(params, foreach=_FOREACH)
    _close(norm, torch.tensor(5.0))
    _close(grad, before.cpu())  # measure-only path leaves grad unmutated


def test_single_tensor_input_is_accepted() -> None:
    """A bare parameter (not an iterable) is handled."""
    p = torch.nn.Parameter(torch.zeros(2, device=_test_device()))
    p.grad = torch.tensor([3.0, 4.0], device=_test_device())
    norm = clip_grad_norm_(p, max_norm=10.0, foreach=_FOREACH)
    _close(norm, torch.tensor(5.0))


def test_params_without_grad_are_skipped_in_norm() -> None:
    """Parameters with no gradient do not contribute to the norm."""
    device = _test_device()
    p_with = torch.nn.Parameter(torch.zeros(2, device=device))
    p_with.grad = torch.tensor([3.0, 4.0], device=device)
    p_without = torch.nn.Parameter(torch.zeros(2, device=device))  # grad stays None
    norm = total_grad_norm([p_with, p_without], foreach=_FOREACH)
    _close(norm, torch.tensor(5.0))


def test_inf_norm_uses_max_abs_gradient() -> None:
    """norm_type=inf clips on the max absolute gradient component."""
    params = _params_with_grads([torch.tensor([-7.0, 4.0])])
    norm = total_grad_norm(params, norm_type=float("inf"), foreach=_FOREACH)
    _close(norm, torch.tensor(7.0))


def test_total_param_norm_spans_all_parameters() -> None:
    """The parameter norm is the global norm over values across all tensors."""
    device = _test_device()
    params = [
        torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device)),
        torch.nn.Parameter(torch.tensor([12.0], device=device)),
    ]
    _close(total_param_norm(params), torch.tensor(13.0))


def test_total_param_norm_ignores_gradients() -> None:
    """Values drive the norm; a grad on the same parameter must not contribute."""
    device = _test_device()
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device))
    param.grad = torch.tensor([100.0, 100.0], device=device)
    _close(total_param_norm(param), torch.tensor(5.0))


def test_total_param_norm_inf_uses_max_abs_value() -> None:
    """norm_type=inf reports the max absolute parameter component."""
    device = _test_device()
    param = torch.nn.Parameter(torch.tensor([-7.0, 4.0], device=device))
    _close(total_param_norm(param, norm_type=float("inf")), torch.tensor(7.0))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
