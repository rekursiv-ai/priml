from __future__ import annotations

from collections.abc import Callable

import functools

from torch import Tensor

import pytest
import torch

from priml.math.diffusion.schedule import (
    compute_log_alpha,
    input_conditioning_identity,
    input_conditioning_rectified_flow,
    log_sigma_from_log_snr_per_rectified_flow,
    log_sigma_from_log_snr_per_variance_preserving,
    log_snr_from_log_sigma_per_rectified_flow,
    log_snr_from_log_sigma_per_variance_preserving,
    log_snr_from_log_time_per_logit,
    log_snr_from_log_time_per_logtan,
    log_snr_from_log_time_per_truncnormicdf,
    log_time_from_log_snr_per_logit,
    log_time_from_log_snr_per_logtan,
    log_time_from_log_snr_per_truncnormicdf,
)


@pytest.mark.parametrize(
    ("f", "midpoint_rtol"),
    [
        (log_snr_from_log_time_per_logtan, 1e-1),
        (log_snr_from_log_time_per_truncnormicdf, 1e-4),
    ],
)
def test_schedule_respects_bounds(
    f: Callable[..., Tensor],
    midpoint_rtol: float,
) -> None:
    shift, low, high = -2.0, -9.0, +7.0
    f = functools.partial(f, shift=shift, low=low, high=high)
    log_t = torch.tensor([1.0, 0.5, 0.0]).log()
    snrs = f(log_t)
    torch.testing.assert_close(snrs[0], torch.tensor(low), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        snrs[1],
        torch.tensor(shift),
        rtol=max(midpoint_rtol, 1e-3),
        atol=1e-3,
    )
    torch.testing.assert_close(snrs[2], torch.tensor(high), rtol=1e-4, atol=1e-4)


def test_variance_preserving_scaler() -> None:
    log_snr = torch.linspace(-3.0, 3.0, 12)
    log_sigma = log_sigma_from_log_snr_per_variance_preserving(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    alpha, sigma = torch.exp(log_alpha), torch.exp(log_sigma)
    torch.testing.assert_close(alpha**2, 1 - sigma**2, rtol=1e-5, atol=1e-6)


def test_variance_preserving_inverse_with_sigma() -> None:
    sigma = torch.tensor([0.1, 0.5, 0.9])
    log_snr = log_snr_from_log_sigma_per_variance_preserving(sigma=sigma)
    log_sigma = log_sigma_from_log_snr_per_variance_preserving(log_snr)
    sigma_reconstructed = torch.exp(log_sigma)
    torch.testing.assert_close(sigma, sigma_reconstructed, rtol=1e-5, atol=1e-5)


def test_variance_preserving_inverse_with_log_sigma() -> None:
    log_sigma_in = torch.tensor([-2.0, -1.0, 0.0])
    log_snr = log_snr_from_log_sigma_per_variance_preserving(log_sigma=log_sigma_in)
    log_sigma = log_sigma_from_log_snr_per_variance_preserving(log_snr)
    torch.testing.assert_close(log_sigma_in, log_sigma, rtol=1e-5, atol=1e-5)


def test_variance_preserving_inverse_error() -> None:
    with pytest.raises(ValueError, match=r".*"):
        log_snr_from_log_sigma_per_variance_preserving(sigma=0.5, log_sigma=-1.0)
    with pytest.raises(ValueError, match=r".*"):
        log_snr_from_log_sigma_per_variance_preserving()


def test_linear_interpolation() -> None:
    log_snr = torch.linspace(-2.0, 4.0, 10)
    log_sigma = log_sigma_from_log_snr_per_rectified_flow(log_snr)
    log_alpha = compute_log_alpha(log_snr, log_sigma)
    alpha, sigma = torch.exp(log_alpha), torch.exp(log_sigma)
    torch.testing.assert_close(alpha, 1 - sigma, rtol=1e-5, atol=1e-6)


def test_rectified_flow_inverse_with_sigma() -> None:
    sigma = torch.tensor([0.1, 0.5, 0.9])
    log_snr = log_snr_from_log_sigma_per_rectified_flow(sigma=sigma)
    log_sigma = log_sigma_from_log_snr_per_rectified_flow(log_snr)
    sigma_reconstructed = torch.exp(log_sigma)
    torch.testing.assert_close(sigma, sigma_reconstructed, rtol=1e-5, atol=1e-5)


def test_rectified_flow_inverse_with_log_sigma() -> None:
    log_sigma_in = torch.tensor([-2.0, -1.0, -0.5])
    log_snr = log_snr_from_log_sigma_per_rectified_flow(log_sigma=log_sigma_in)
    log_sigma = log_sigma_from_log_snr_per_rectified_flow(log_snr)
    torch.testing.assert_close(log_sigma_in, log_sigma, rtol=1e-5, atol=1e-5)


def test_rectified_flow_inverse_error() -> None:
    with pytest.raises(ValueError, match=r".*"):
        log_snr_from_log_sigma_per_rectified_flow(sigma=0.5, log_sigma=-1.0)
    with pytest.raises(ValueError, match=r".*"):
        log_snr_from_log_sigma_per_rectified_flow()


def test_log_snr_from_log_time_per_logtan_basic() -> None:
    log_t = torch.tensor([0.0, 0.5, 1.0]).log()
    log_snr = log_snr_from_log_time_per_logtan(log_t)
    assert torch.all(log_snr[:-1] >= log_snr[1:])


def test_log_snr_from_log_time_per_truncnormicdf_basic() -> None:
    log_t = torch.tensor([0.0, 0.5, 1.0]).log()
    log_snr = log_snr_from_log_time_per_truncnormicdf(log_t)
    assert torch.all(log_snr[:-1] >= log_snr[1:])


def test_input_conditioning_rectified_flow() -> None:
    x = torch.randn(3, 2, 4)
    log_snr = torch.tensor([[[0.0]], [[1.0]], [[-1.0]]])
    result = input_conditioning_rectified_flow(x, log_snr)
    assert result.shape == x.shape


def test_input_conditioning_identity() -> None:
    x = torch.randn(2, 3, 4)
    log_snr = torch.tensor([0.0, 1.0, -1.0])
    result = input_conditioning_identity(x, log_snr)
    torch.testing.assert_close(result, x)


def test_log_snr_from_log_time_per_logit_roundtrip() -> None:
    log_t = torch.linspace(0.01, 0.99, 10).log()
    log_snr = log_snr_from_log_time_per_logit(log_t, low=-100, high=100)
    log_t_recovered = log_time_from_log_snr_per_logit(log_snr)
    torch.testing.assert_close(log_t_recovered, log_t)


def test_log_time_from_log_snr_per_logtan_roundtrip() -> None:
    log_t = torch.linspace(0.01, 0.99, 10).log()
    log_snr = log_snr_from_log_time_per_logtan(log_t, low=-100, high=100)
    log_t_recovered = log_time_from_log_snr_per_logtan(
        log_snr,
        low=-100,
        high=100,
    )
    torch.testing.assert_close(log_t_recovered, log_t)


def test_log_time_from_log_snr_per_truncnormicdf_roundtrip() -> None:
    log_t = torch.linspace(0.01, 0.99, 10).log()
    log_snr = log_snr_from_log_time_per_truncnormicdf(
        log_t,
        low=-100,
        high=100,
    )
    log_t_recovered = log_time_from_log_snr_per_truncnormicdf(
        log_snr,
        low=-100,
        high=100,
    )
    torch.testing.assert_close(log_t_recovered, log_t, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
