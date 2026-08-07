from __future__ import annotations

from torch import Tensor, exp

import pytest
import torch

from priml.math.custom_types import TensorFn
from priml.math.diffusion.schedule import (
    compute_log_alpha,
    log_sigma_from_log_snr_per_rectified_flow,
    log_sigma_from_log_snr_per_variance_preserving,
)
from priml.math.diffusion.target import (
    TargetFn,
    target_eps,
    target_rectified_flow,
    target_v,
    target_v_eps,
    target_v_x,
    target_x,
)


@pytest.mark.parametrize(
    ("target_fn", "corrupt_fn"),
    [
        (target_x, log_sigma_from_log_snr_per_rectified_flow),
        (target_eps, log_sigma_from_log_snr_per_rectified_flow),
        (target_v, log_sigma_from_log_snr_per_rectified_flow),
        (target_v_x, log_sigma_from_log_snr_per_rectified_flow),
        (target_v_eps, log_sigma_from_log_snr_per_rectified_flow),
        (target_rectified_flow, log_sigma_from_log_snr_per_rectified_flow),
    ],
)
def test_target_x(
    target_fn: TargetFn,
    corrupt_fn: TensorFn,
) -> None:
    torch.manual_seed(42)
    x_shape = 2, 4, 3, 6, 5
    _xem = torch.randn(size=(3, *x_shape))
    x_original: Tensor = _xem[0]
    eps_original: Tensor = _xem[1]
    model_noise: Tensor = _xem[2]
    log_snr = torch.tensor([-5.0, 0.0, 2.0]).view(-1, 1, 1, 1, 1, 1)

    log_sigma = corrupt_fn(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    x_noisy = exp(log_alpha) * x_original + exp(log_sigma) * eps_original
    target, predict, x_clean, eps_clean = target_fn(
        model=0.6 * x_noisy + 0.4 * model_noise,
        x_noisy=x_noisy,
        log_snr=log_snr,
        log_sigma=log_sigma,
        x_original=x_original,
        eps_original=eps_original,
    )
    broadcast_shape = torch.broadcast_shapes(log_snr.shape, x_original.shape)
    # All outputs must be broadcastable to the expected shape.
    if target is not None:
        assert torch.broadcast_shapes(broadcast_shape, target.shape) == broadcast_shape
    if predict is not None:
        assert torch.broadcast_shapes(broadcast_shape, predict.shape) == broadcast_shape
    assert torch.broadcast_shapes(broadcast_shape, x_clean.shape) == broadcast_shape
    assert torch.broadcast_shapes(broadcast_shape, eps_clean.shape) == broadcast_shape


@pytest.mark.parametrize(
    ("target_fn", "corrupt_fn"),
    [
        (target_x, log_sigma_from_log_snr_per_rectified_flow),
        (target_eps, log_sigma_from_log_snr_per_rectified_flow),
        (target_v, log_sigma_from_log_snr_per_rectified_flow),
        (target_v_x, log_sigma_from_log_snr_per_variance_preserving),
        (target_v_x, log_sigma_from_log_snr_per_rectified_flow),
        (target_v_eps, log_sigma_from_log_snr_per_variance_preserving),
        (target_v_eps, log_sigma_from_log_snr_per_rectified_flow),
        (target_rectified_flow, log_sigma_from_log_snr_per_rectified_flow),
    ],
)
def test_target_reconstruction(
    target_fn: TargetFn,
    corrupt_fn: TensorFn,
) -> None:
    """With a perfect model, x_clean == x_original and eps_clean == eps_original."""
    torch.manual_seed(0)
    x_original = torch.randn(2, 3)
    eps_original = torch.randn(2, 3)
    log_snr = torch.tensor([3.0, -1.0]).unsqueeze(-1)

    log_sigma = corrupt_fn(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    a, s = exp(log_alpha), exp(log_sigma)
    x_noisy = a * x_original + s * eps_original

    # Perfect model output depends on what the function expects:
    #   target_x: model predicts x
    #   target_eps: model predicts eps
    #   target_v/v_x/v_eps: model predicts v = a*eps - s*x
    #   target_rectified_flow: model predicts eps - x
    if target_fn in (target_v, target_v_x, target_v_eps):
        perfect_model = a * eps_original - s * x_original
    elif target_fn is target_x:
        perfect_model = x_original
    elif target_fn is target_eps:
        perfect_model = eps_original
    elif target_fn is target_rectified_flow:
        perfect_model = eps_original - x_original
    else:
        raise ValueError(f"Unknown target_fn: {target_fn}")

    _, _, x_clean, eps_clean = target_fn(
        model=perfect_model,
        x_noisy=x_noisy,
        log_snr=log_snr,
        log_sigma=log_sigma,
    )
    torch.testing.assert_close(
        x_clean,
        x_original.expand_as(x_clean),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        eps_clean,
        eps_original.expand_as(eps_clean),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
