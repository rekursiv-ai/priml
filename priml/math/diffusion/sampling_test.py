from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import islice

import math

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.diffusion import SampleOneStepResult
from priml.math.diffusion.sampling import (
    SampleResult,
    ddpm,
    ddpm_ddim_step,
    rescale_classifier_free_guidance,
    sample,
    sample_iter,
)
from priml.math.diffusion.schedule import (
    compute_log_alpha,
    log_sigma_from_log_snr_per_rectified_flow,
)
from priml.math.diffusion.target import (
    TargetFn,
    target_rectified_flow,
)


def _reference_ddpm_step(
    model: Tensor,
    x_curr: Tensorable,
    log_snr_curr: Tensorable,
    log_snr_next: Tensorable,
    *,
    corruption_fn: Callable[
        [Tensorable],
        Tensor,
    ] = log_sigma_from_log_snr_per_rectified_flow,
    target_fn: TargetFn = target_rectified_flow,
    eta: float = 1,
) -> SampleOneStepResult:
    """Reference DDPM/DDIM step in linear space.

    Direct implementation of Song et al. 2020 Eq 12 to validate
    the numerically stable log-space version.
    """
    model, x_t, snr_t, snr_s = convert_to_tensor(
        model,
        x_curr,
        log_snr_curr,
        log_snr_next,
    )
    # s = next (less noisy), t = curr (noisier)
    log_sig_s = corruption_fn(snr_s)
    log_sig_t = corruption_fn(snr_t)
    sigma_s = torch.exp(log_sig_s)
    sigma_t = torch.exp(log_sig_t)
    alpha_s = torch.exp(compute_log_alpha(snr_s, log_sig_s))
    alpha_t = torch.exp(compute_log_alpha(snr_t, log_sig_t))
    predicted_x, predicted_eps = target_fn(model, x_t, snr_t, log_sig_t)[-2:]

    # DDPM posterior variance
    snr_ratio = (alpha_t / sigma_t) ** 2 / (alpha_s / sigma_s) ** 2
    sigma_ddpm = sigma_s * (1 - snr_ratio).clamp(min=0) ** 0.5

    # η interpolation
    sigma_stoch = eta * sigma_ddpm
    sigma_determ = (sigma_s**2 - sigma_stoch**2).clamp(min=0) ** 0.5
    posterior_mean = alpha_s * predicted_x + sigma_determ * predicted_eps

    log_std = torch.where(
        sigma_stoch > 0,
        torch.log(sigma_stoch),
        torch.tensor(
            -math.inf,
            dtype=posterior_mean.dtype,
            device=posterior_mean.device,
        ),
    )

    return SampleOneStepResult(predicted_x, posterior_mean, log_std)


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(101)


def test_sample_model_fn_protocol():
    """Test SampleModelFn protocol with various return types."""

    def model_fn_simple(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x

    log_snr = torch.tensor([2.0, 0.3, -1.5])
    x = torch.randn(2, 3)

    result = model_fn_simple(
        x,
        log_snr[0],
        torch.tensor(0),
        2,
        log_snr[1],
    )
    assert isinstance(result, Tensor)


@pytest.mark.parametrize("eta", [0, 0.33, 0.71, 1.0])
def test_ddpm(eta: float) -> None:
    x_shape = 4, 3, 6, 5
    _mx = torch.randn(size=(2, *x_shape))
    model: Tensor = _mx[0]
    x1: Tensor = _mx[1]
    log_snr_curr = torch.tensor([-4.0, 0.5, 2.3, 1.0]).reshape(-1, 1, 1, 1)
    log_snr_next = torch.tensor([-0.5, 3.5, 5.0, 4.2]).reshape(-1, 1, 1, 1)
    assert torch.all(log_snr_curr < log_snr_next)
    new_x0, new_mean, new_log_std = ddpm(
        model,
        x1,
        log_snr_curr,
        log_snr_next,
        eta=eta,
    )
    ref_x0, ref_mean, ref_log_std = _reference_ddpm_step(
        model,
        x1,
        log_snr_curr,
        log_snr_next,
        eta=eta,
    )
    assert torch.isfinite(ref_mean).all(), ref_mean
    assert torch.isfinite(new_mean).all(), new_mean
    torch.testing.assert_close(new_log_std, ref_log_std, rtol=5e-4, atol=1e-6)
    torch.testing.assert_close(new_mean, ref_mean, rtol=5e-3, atol=1e-6)
    torch.testing.assert_close(new_x0, ref_x0, rtol=5e-4, atol=1e-6)


@pytest.mark.parametrize("eta", [0, 0.33, 0.71, 1.0])
def test_ddpm_step_matches_ddpm(eta: float) -> None:
    """Verify ddpm_step produces identical results to ddpm."""
    x_shape = 3, 2, 5, 4
    _mx = torch.randn(size=(2, *x_shape))
    model: Tensor = _mx[0]
    x1: Tensor = _mx[1]
    log_snr1 = np.reshape([-3.5, 0.2, 1.8], [-1] + [1] * 3)
    log_snr0 = np.reshape([-0.7, 3.1, 4.6], [-1] + [1] * 3)

    # ddpm (wrapper)
    x0_ddpm, mean_ddpm, log_std_ddpm = ddpm(model, x1, log_snr1, log_snr0, eta=eta)

    # ddpm_step (manual decompose + step)
    log_snr1_t, log_snr0_t = convert_to_tensor(log_snr1, log_snr0)
    log_sigma_curr = log_sigma_from_log_snr_per_rectified_flow(log_snr1_t)
    log_sigma_next = log_sigma_from_log_snr_per_rectified_flow(log_snr0_t)
    model_t, x1_t = convert_to_tensor(model, x1)
    result = target_rectified_flow(
        model_t,
        x1_t,
        log_snr1_t,
        log_sigma_curr,
    )
    mean_step, log_std_step = ddpm_ddim_step(
        result.x_clean,
        result.eps_clean,
        log_snr1_t,
        log_snr0_t,
        log_sigma_next,
        eta=eta,
    )

    # x_clean from ddpm wrapper vs target_fn
    torch.testing.assert_close(x0_ddpm, result.x_clean)
    torch.testing.assert_close(mean_ddpm, mean_step)
    torch.testing.assert_close(log_std_ddpm, log_std_step)


def test_ddim_is_deterministic() -> None:
    """Test that DDIM (eta=0) produces deterministic sampling."""
    x_shape = 4, 3, 6, 5
    _mx = torch.randn(size=(2, *x_shape))
    model: Tensor = _mx[0]
    x1: Tensor = _mx[1]
    log_snr1 = np.reshape([-4.2, 0.3, -1.7, 2.4], [-1] + [1] * 3)
    log_snr0 = np.reshape([-1.1, 3.8, 0.9, 5.0], [-1] + [1] * 3)
    assert all(log_snr1 < log_snr0)

    _, _, log_std = ddpm(model, x1, log_snr1, log_snr0, eta=0)

    assert torch.all(torch.isinf(log_std) & (log_std < 0)), (
        "DDIM (eta=0) should produce log_std=-inf for "
        "deterministic sampling, "
        f"but got {log_std}"
    )


@pytest.mark.parametrize(
    ("log_snr_curr", "log_snr_next"),
    [
        (-20, 20),
        (-10, -5),
        (5, 10),
        (-5, 5),
    ],
)
def test_numerical_stability_extreme_log_snr(
    log_snr_curr: float,
    log_snr_next: float,
) -> None:
    """Test numerical stability for extreme log_snr values."""
    model = torch.randn(2, 3)
    x1 = torch.randn(2, 3)

    x0, mean, log_std = ddpm(model, x1, log_snr_curr, log_snr_next)

    assert torch.isfinite(x0).all(), (
        f"x0 not finite for log_snr {log_snr_curr}->{log_snr_next}"
    )
    assert torch.isfinite(mean).all(), (
        f"mean not finite for log_snr {log_snr_curr}->{log_snr_next}"
    )
    assert torch.isfinite(log_std).all(), (
        f"log_std not finite for log_snr {log_snr_curr}->{log_snr_next}"
    )


def test_sample_without_stateful() -> None:
    def model_fn(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x.clone()

    log_snr = torch.tensor([3.0, 1.0, -0.5, -2.0])
    x = torch.tensor([[1.2, -0.7, 0.3], [-0.9, 2.4, 0.1]])
    with torch.no_grad():
        r = sample(log_snr, model_fn, x)
    assert isinstance(r, SampleResult)
    assert x.shape == r.x_clean.shape, r.x_clean.shape


def test_sample_with_insufficient_steps():
    """Test sample raises ValueError with insufficient steps."""

    def model_fn(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x

    log_snr_empty = torch.tensor([])
    x = torch.randn(2, 3)
    with pytest.raises(
        ValueError,
        match="Expected log_snr to have leading size",
    ):
        sample(log_snr_empty, model_fn, x)

    log_snr_single = torch.tensor([1.0])
    with pytest.raises(
        ValueError,
        match="Expected log_snr to have leading size",
    ):
        sample(log_snr_single, model_fn, x)


def test_sample_iter_yields_each_step():
    """sample_iter yields one SampleResult per diffusion step."""

    def model_fn(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x.clone()

    log_snr = torch.tensor([2.0, 0.3, -1.5])
    x = torch.randn(2, 3)

    with torch.no_grad():
        gen = sample_iter(log_snr, model_fn, x)
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

        results = list(gen)
        assert len(results) == 2


def _identity_model_fn(
    x: Tensor,
    log_snr: Tensor,
    it: Tensor,
    num_steps: int,
    log_snr_next: Tensor,
) -> Tensor:
    del log_snr, it, num_steps, log_snr_next
    return x.clone()


def test_sample_wrap_steps_sees_every_step():
    """wrap_steps is applied to the per-step iterator before draining."""
    log_snr = torch.tensor([2.0, 0.3, -1.5])  # 2 steps
    x = torch.randn(2, 3)
    seen: list[SampleResult] = []

    def spy(it: Iterable[SampleResult]) -> Iterable[SampleResult]:
        for step in it:
            seen.append(step)
            yield step

    with torch.no_grad():
        sample(log_snr, _identity_model_fn, x, wrap_steps=spy)
    assert len(seen) == 2  # one per diffusion step


def test_sample_wrap_steps_default_does_not_change_result():
    """A pass-through wrap_steps returns the same final step as no wrap.

    Sampling draws per-step noise, so both runs are seeded identically to
    isolate the wrap_steps effect from RNG.
    """
    log_snr = torch.tensor([-2.0, 0.3, 2.0])
    x = torch.randn(2, 3)
    with torch.no_grad():
        torch.manual_seed(0)
        plain = sample(log_snr, _identity_model_fn, x)
        torch.manual_seed(0)
        wrapped = sample(log_snr, _identity_model_fn, x, wrap_steps=lambda it: it)
    torch.testing.assert_close(wrapped.x_curr, plain.x_curr)


def test_sample_wrap_steps_can_truncate_the_run():
    """wrap_steps controls the drain: islice stops the run early.

    The returned step is the last one the wrapped iterator yielded. A spy
    records how many steps were actually consumed, proving islice cut the
    3-step run short and the returned step is the truncation point.
    """
    log_snr = torch.tensor([2.0, 0.3, -1.5, -3.0])  # 3 steps
    x = torch.randn(2, 3)
    consumed: list[SampleResult] = []

    def take_two(it: Iterable[SampleResult]) -> Iterable[SampleResult]:
        for step in islice(it, 2):  # stop after the 2nd of 3 steps
            consumed.append(step)
            yield step

    with torch.no_grad():
        result = sample(log_snr, _identity_model_fn, x, wrap_steps=take_two)

    assert len(consumed) == 2  # the 3rd step was never generated
    # sample() returns exactly the last step the wrapped stream yielded
    # (identity check, robust to a degenerate model_fn producing non-finite
    # values: this asserts the drain plumbing, not the diffusion numerics).
    assert result.x_curr is consumed[-1].x_curr


def test_sample_default_last_step_returns_clean_mean():
    """Default (noise_last_step=False): final x_curr is the noise-free mean."""

    def model_fn(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x.clone()

    log_snr = torch.tensor([-2.0, 0.3, 2.0])
    x = torch.randn(2, 3)
    with torch.no_grad():
        result = sample(log_snr, model_fn, x)
    # Ho et al. 2020 Alg. 2: sigma=0 at t=0, so the final step is noise-free.
    torch.testing.assert_close(result.x_curr, result.mean)


def test_sample_noise_last_step_true_restores_noise():
    """noise_last_step=True: final x_curr adds sampling noise to the mean."""

    def model_fn(
        x: Tensor,
        log_snr: Tensor,
        it: Tensor,
        num_steps: int,
        log_snr_next: Tensor,
    ) -> Tensor:
        del log_snr, it, num_steps, log_snr_next
        return x.clone()

    log_snr = torch.tensor([-2.0, 0.3, 2.0])
    x = torch.randn(2, 3)
    with torch.no_grad():
        result = sample(log_snr, model_fn, x, noise_last_step=True)
    # log_std is finite here, so the noisy update differs from the bare mean.
    assert not torch.allclose(result.x_curr, result.mean)


@pytest.mark.parametrize("eta", [0.0, 0.01, 0.5, 1.0])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
@pytest.mark.parametrize(
    ("log_snr_curr", "log_snr_next"),
    [
        (-20.0, 20.0),
        (-10.0, -9.999),
        (-1.0, 1.0),
    ],
)
def test_ddpm_no_nan_in_value_or_gradient(
    eta: float,
    log_snr_curr: float,
    log_snr_next: float,
    dtype: torch.dtype,
) -> None:
    """Verify ddpm produces finite values and gradients."""
    model = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    x1 = torch.randn(2, 3, dtype=dtype)
    x0, mean, log_std = ddpm(
        model,
        x1,
        log_snr_curr,
        log_snr_next,
        eta=eta,
    )
    assert torch.isfinite(x0).all(), f"x0 not finite: {x0}"
    assert torch.isfinite(mean).all(), f"mean not finite: {mean}"
    # log_std can be -inf when low precision rounds away small
    # step differences. Must never be NaN or +inf.
    assert not torch.isnan(log_std).any(), f"log_std is NaN: {log_std}"
    assert not (torch.isinf(log_std) & (log_std > 0)).any(), (
        f"log_std is +inf: {log_std}"
    )
    loss = mean.sum()
    loss.backward()
    assert model.grad is not None
    assert torch.isfinite(model.grad).all(), f"grad not finite: {model.grad}"


def test_rescale_cfg_default_rho():
    """Test rescale_classifier_free_guidance with default rho is a no-op."""
    guided = torch.randn(2, 3, 4, 5, 6)
    unguided = torch.randn(2, 3, 4, 5, 6)

    result = rescale_classifier_free_guidance(
        guided,
        unguided,
        spatial_dims=(-4, -3, -2, -1),
    )
    torch.testing.assert_close(result, guided)


def test_rescale_cfg_nonzero_rho():
    """Test rescale_classifier_free_guidance with non-zero rho."""
    guided = torch.randn(2, 3, 4, 5, 6)
    unguided = torch.randn(2, 3, 4, 5, 6)

    result = rescale_classifier_free_guidance(
        guided,
        unguided,
        strength=0.5,
        spatial_dims=(-4, -3, -2, -1),
    )
    assert result.shape == guided.shape
    assert not torch.allclose(result, guided)


def test_rescale_cfg_custom_dim():
    """Test rescale_classifier_free_guidance with custom dimensions."""
    guided = torch.randn(4, 8, 16, 16)
    unguided = torch.randn(4, 8, 16, 16)

    result = rescale_classifier_free_guidance(
        guided,
        unguided,
        strength=0.7,
        spatial_dims=(-3, -2, -1),
    )
    assert result.shape == guided.shape


def test_rescale_cfg_eps_parameter():
    """Test rescale_classifier_free_guidance eps prevents division by zero."""
    guided = torch.ones(2, 3, 4, 5, 6) * 1e-10
    unguided = torch.randn(2, 3, 4, 5, 6)

    result = rescale_classifier_free_guidance(
        guided,
        unguided,
        strength=0.5,
        spatial_dims=(-4, -3, -2, -1),
    )
    assert result.shape == guided.shape
    assert torch.isfinite(result).all()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
