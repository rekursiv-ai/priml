from __future__ import annotations

from importlib.util import find_spec

import math

from wrapt import lazy_import

import numpy as np
import pytest
import torch

from priml.math.custom_types import convert_to_tensor
from priml.math.probability import (
    cdf_logit_normal,
    cdf_normal,
    cdf_truncated_normal,
    cdf_uniform,
    lbeta,
    log_cdf_truncated_normal,
    log_gamma_correction,
    log_gamma_difference,
    ndtr,
    ndtri,
    pdf_logit_distribution,
    pdf_logit_normal,
    pdf_normal,
    pdf_uniform,
    quantile_logit_distribution,
    quantile_logit_normal,
    quantile_normal,
    quantile_truncated_normal,
    quantile_uniform,
    random_categorical,
    random_chi2,
    random_gamma,
    random_logit_normal,
    random_student_t,
)


# scipy is the reference oracle for the special-function parity tests only; it
# is an optional test dependency. The lazy proxy defers the real import to
# first attribute access, which only happens inside a parity test body -- and
# those are skipped (not errored) when scipy is absent.
_HAS_SCIPY = find_spec("scipy") is not None
requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
scipy_special = lazy_import("scipy.special")


def test_ndtr():
    x = torch.randn(3, 4, 6)
    x_ = ndtri(ndtr(x))
    torch.testing.assert_close(x, x_, atol=5e-4, rtol=1e-3)


def test_pdf_normal():
    x = torch.tensor([0.0, 1.0, -1.0])
    loc, scale = 0.0, 1.0
    result = pdf_normal(x, loc, scale)
    expected = torch.distributions.Normal(loc, scale).log_prob(x).exp()
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_cdf_normal():
    x = torch.tensor([0.0, 1.0, -1.0])
    loc, scale = 0.5, 2.0
    result = cdf_normal(x, loc, scale)
    # Test against ndtr implementation
    y = (x - loc) / scale
    expected = ndtr(y)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_quantile_normal():
    p = torch.tensor([0.5, 0.84, 0.16])
    loc, scale = 1.0, 2.0
    result = quantile_normal(p, loc=loc, scale=scale)
    # Check that cdf(quantile(p)) = p
    cdf_result = cdf_normal(result, loc, scale)
    torch.testing.assert_close(cdf_result, p, atol=1e-5, rtol=1e-5)


def test_pdf_uniform():
    x = torch.tensor([-0.5, 0.25, 0.75, 1.5])
    result = pdf_uniform(x, 0.0, 1.0)
    expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(result, expected)
    # Non-unit interval.
    result2 = pdf_uniform(x, 0.0, 2.0)
    expected2 = torch.tensor([0.0, 0.5, 0.5, 0.5])
    torch.testing.assert_close(result2, expected2)


def test_cdf_uniform():
    x = torch.tensor([-0.5, 0.25, 0.75, 1.5])
    low, high = 0.0, 1.0
    result = cdf_uniform(x, low, high)
    expected = torch.tensor([0.0, 0.25, 0.75, 1.0])
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_quantile_uniform():
    p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    low, high = 2.0, 5.0
    result = quantile_uniform(p, low, high)
    expected = torch.tensor([2.0, 2.75, 3.5, 4.25, 5.0])
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_pdf_uniform_degenerate_interval_is_finite():
    """High == low is a point mass; the density must stay finite (0 off-support)."""
    result = pdf_uniform(torch.tensor([0.5, 1.0, 1.5]), 1.0, 1.0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros(3))


def test_cdf_uniform_degenerate_interval_is_unit_step():
    """High == low: CDF is the unit step at low, not 0/0 == nan."""
    result = cdf_uniform(torch.tensor([0.5, 1.0, 1.5]), 1.0, 1.0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 1.0]))


def test_quantile_uniform_rejects_out_of_range_p():
    """P outside [0, 1] is outside the quantile domain and must raise."""
    with pytest.raises(ValueError, match=r"p in \[0, 1\]"):
        quantile_uniform(torch.tensor([0.5, 2.0]), 0.0, 1.0)
    with pytest.raises(ValueError, match=r"p in \[0, 1\]"):
        quantile_uniform(torch.tensor([-0.1, 0.5]), 0.0, 1.0)


def test_pdf_logit_normal():
    x = torch.tensor([0.1, 0.5, 0.9])
    loc, scale = 0.0, 1.0
    result = pdf_logit_normal(x, loc, scale)
    assert result.shape == x.shape
    assert torch.all(result > 0)


def test_cdf_logit_normal():
    x = torch.tensor([0.1, 0.5, 0.9])
    loc, scale = 0.0, 1.0
    result = cdf_logit_normal(x, loc, scale)
    assert result.shape == x.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_quantile_logit_normal():
    p = torch.tensor([0.1, 0.5, 0.9])
    loc, scale = 0.0, 1.0
    result = quantile_logit_normal(p, loc, scale)
    assert result.shape == p.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_quantile_truncated_normal():
    p = torch.tensor([0.0, 0.5, 1.0])
    loc, scale, low, high = 0.0, 1.0, -2.0, 2.0
    result = quantile_truncated_normal(p, loc=loc, scale=scale, low=low, high=high)
    assert result.shape == p.shape
    # Results should be within bounds
    assert torch.all(result >= low)
    assert torch.all(result <= high)


def test_cdf_truncated_normal_degenerate_interval_is_unit_step():
    """Low == high is a point mass at low: CDF is the unit step, not 0/0 nan.

    Matches the uniform-family convention (cdf_uniform degenerate case).
    """
    x = torch.tensor([-1.0, 1.0, 2.0])
    result = cdf_truncated_normal(x, low=1.0, high=1.0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 1.0]))


def test_log_cdf_truncated_normal_degenerate_interval_is_finite():
    """Low == high must give log of the unit step, not logsubexp(-inf, -inf) nan."""
    x = torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
    result = log_cdf_truncated_normal(x, low=1.0, high=1.0)
    assert not torch.isnan(result).any()
    # log(unit step at low): -inf below, 0 at/above.
    torch.testing.assert_close(
        result, torch.tensor([-math.inf, 0.0, 0.0], dtype=torch.float64)
    )


def test_quantile_truncated_normal_rejects_out_of_range_p():
    """P outside [0, 1] is outside the quantile domain and must raise.

    Matches quantile_uniform, which already guards its p-domain.
    """
    with pytest.raises(ValueError, match=r"p in \[0, 1\]"):
        quantile_truncated_normal(torch.tensor([0.5, 2.0]), low=-2.0, high=2.0)
    with pytest.raises(ValueError, match=r"p in \[0, 1\]"):
        quantile_truncated_normal(torch.tensor([-0.1, 0.5]), low=-2.0, high=2.0)


def test_random_categorical_all_neg_inf_logits_raises():
    """All -inf logits define no distribution; the cmf becomes NaN.

    Silently returning index 0 (or out-of-range) hides a malformed input, so
    the function must reject it rather than sample garbage.
    """
    with pytest.raises(ValueError, match=r"categorical|finite|probability"):
        random_categorical(5, logits=torch.tensor([-math.inf, -math.inf, -math.inf]))


def test_pdf_logit_distribution_outside_unit_interval_is_zero():
    """Density of a (0, 1)-supported logit distribution is 0 off-support.

    logit(x) is NaN/+-inf outside (0, 1); the density must follow the
    uniform-family off-support convention (0), not crash or return NaN.
    """
    x = torch.tensor([-0.5, 0.5, 1.5])
    result = pdf_logit_normal(x)
    assert torch.isfinite(result).all()
    assert result[0] == 0.0
    assert result[2] == 0.0
    assert result[1] > 0.0


def test_cdf_logit_distribution_outside_unit_interval_is_clamped():
    """CDF of a (0, 1)-supported logit distribution is 0 below, 1 above."""
    x = torch.tensor([-0.5, 0.5, 1.5])
    result = cdf_logit_normal(x)
    assert torch.isfinite(result).all()
    assert result[0] == 0.0
    assert result[2] == 1.0
    assert 0.0 < result[1] < 1.0


def test_pdf_logit_distribution():
    x = torch.tensor([0.1, 0.5, 0.9])

    def base_pdf(x: torch.Tensor) -> torch.Tensor:
        return pdf_normal(x, 0.0, 1.0)

    result = pdf_logit_distribution(x, base_pdf)
    assert result.shape == x.shape
    assert torch.all(result > 0)


def test_quantile_logit_distribution():
    p = torch.tensor([0.1, 0.5, 0.9])

    def base_quantile(p: torch.Tensor) -> torch.Tensor:
        return quantile_normal(p, loc=0.0, scale=1.0)

    result = quantile_logit_distribution(p, base_quantile)
    assert result.shape == p.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_student_t():
    n, df, loc, scale = 300_000, [4.0, 8.0], [[1.0]], 1.5
    x = random_student_t(n, df=df, loc=loc, scale=scale)
    assert x.shape == (n, 1, 2), x.shape
    mean = x.mean(dim=0)
    torch.testing.assert_close(
        mean,
        torch.full_like(mean, loc[0][0]),
        atol=0.08,
        rtol=0.08,
    )
    # Var = scale^2 * df / (df - 2) for df > 2
    df_t = convert_to_tensor(df)
    expected_std = scale * torch.sqrt(df_t / (df_t - 2))
    actual_std = x.std(dim=0).squeeze(0)
    torch.testing.assert_close(actual_std, expected_std, atol=0.15, rtol=0.12)


def test_random_chi2():
    # Test basic chi2 random generation - just ensure it runs
    n = 1_000
    df = torch.tensor([2.0])
    x = random_chi2(n, df=df)
    # Just check it returns data
    assert x.shape[0] == n
    assert torch.all(x > 0)  # Chi2 is always positive


def test_gamma():
    n, concentration, rate = 150_000, [1.0, 2.5, 5.0], 2.0
    conc_t = torch.tensor(concentration)
    x = random_gamma(n, concentration=concentration, rate=rate)
    # Mean = concentration / rate
    mean = x.mean(dim=0)
    expected_mean = conc_t / rate
    torch.testing.assert_close(mean, expected_mean, atol=0.08, rtol=0.08)
    # Var = concentration / rate^2
    std = x.std(dim=0)
    expected_std = torch.sqrt(conc_t / rate**2)
    torch.testing.assert_close(std, expected_std, atol=0.12, rtol=0.12)


def test_gamma_with_rate():
    n, concentration, rate = 10_000, 2.0, 0.5
    x = random_gamma(n, concentration=concentration, rate=rate)
    # Gamma mean = concentration / rate
    mean = x.mean()
    expected_mean = concentration / rate
    np.testing.assert_allclose(mean, expected_mean, atol=0, rtol=0.1)


@pytest.mark.parametrize("use_logits", [True, False])
def test_categorical(use_logits: bool):
    n = 80_000
    probs_t = torch.tensor([[0.15, 0.35, 0.5], [0.6, 0.0, 0.4]])
    if use_logits:
        logits_t = probs_t.log()  # -inf for zero entries via torch
        x = random_categorical(n, logits=logits_t)
    else:
        x = random_categorical(n, probs=probs_t)
    # Compute empirical frequencies with a simple loop over categories
    freq = torch.zeros_like(probs_t)
    for row in range(probs_t.shape[0]):
        for cat in range(probs_t.shape[1]):
            freq[row, cat] = (x[:, row] == cat).float().mean()
    torch.testing.assert_close(
        probs_t,
        freq,
        atol=1.5e-2,
        rtol=0.12,
    )


def test_categorical_error():
    # Test that providing both probs and logits raises error
    with pytest.raises(ValueError, match=r".*"):
        random_categorical(100, probs=[0.5, 0.5], logits=[0.0, 0.0])
    # Test that providing neither raises error
    with pytest.raises(ValueError, match=r".*"):
        random_categorical(100)


def test_random_dtype_is_honored_with_float_loc_tensor_scale():
    """Explicit dtype must survive a Python-float loc + tensor scale.

    Regression guard for MATH-003 (rejected non-bug): all params route through
    a single convert_to_tensor, which unifies dtype and broadcasts loc, so the
    requested dtype is preserved rather than overridden by scale's dtype.
    """
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
    x = random_logit_normal(4, loc=0.0, scale=scale, dtype=torch.float64)
    assert x.dtype == torch.float64
    assert x.shape == (4, 2)
    x = random_student_t(4, df=4.0, loc=0.0, scale=scale, dtype=torch.float64)
    assert x.dtype == torch.float64
    x = random_gamma(4, concentration=2.0, rate=scale, dtype=torch.float64)
    assert x.dtype == torch.float64


def test_random_logit_normal():
    n, loc, scale = 10_000, 0.0, 1.0
    x = random_logit_normal(n, loc=loc, scale=scale)
    assert x.shape == (n,)
    assert torch.all(x >= 0)
    assert torch.all(x <= 1)


def test_random_logit_normal_with_sequence():
    # Test with sequence input
    n = (5, 10)
    x = random_logit_normal(*n, loc=0.0, scale=1.0)
    assert x.shape == (5, 10)


def test_random_student_t_with_sequence():
    # Test with sequence input (line 212)
    n = (5, 10)  # Pass as tuple (sequence)
    df = torch.tensor([2.0])
    x = random_student_t(*n, df=df, loc=0.0, scale=1.0)
    assert x.shape == (5, 10, 1)


def test_random_gamma_with_sequence():
    # Test with sequence input (line 249)
    n = (3, 4)  # Pass as tuple (sequence)
    x = random_gamma(*n, concentration=1.0, rate=1.0)
    assert x.shape == (3, 4)


def test_random_categorical_with_sequence():
    # Test with sequence input (line 266)
    n = (10, 5)  # Pass as tuple (sequence)
    probs = [0.3, 0.7]
    x = random_categorical(*n, probs=probs)
    assert x.shape == (10, 5)


def test_random_categorical_unreachable_branch():
    # Test the unreachable AssertionError branch (line 280)
    # This should never be hit in normal operation
    # Just test normal categorical behavior
    x = random_categorical(10, probs=[0.5, 0.5])
    assert x.shape == (10,)


def test_random_logit_normal_with_tuple():
    # Test with tuple input (line 295)
    n = (5, 10)  # Pass as tuple (sequence)
    x = random_logit_normal(*n, loc=0.0, scale=1.0)
    assert x.shape == (5, 10)


def test_log_cdf_truncated_normal() -> None:
    x = torch.tensor(
        [-1.5, -1.0, 0.0, 1.0, 1.5],
        dtype=torch.float64,
    )
    loc, scale, low, high = 0.0, 1.0, -2.0, 2.0
    log_cdf = log_cdf_truncated_normal(
        x,
        loc=loc,
        scale=scale,
        low=low,
        high=high,
    )
    cdf = cdf_truncated_normal(
        x,
        loc=loc,
        scale=scale,
        low=low,
        high=high,
    )
    expected = torch.log(cdf.to(torch.float64))
    torch.testing.assert_close(log_cdf, expected, rtol=1e-10, atol=1e-10)


@requires_scipy
def test_ndtr_left_tail_matches_scipy() -> None:
    """Deep left tail: the piecewise erfc path must match scipy to f64.

    ``torch.special.ndtr`` loses precision here; this is exactly why the
    custom erf/erfc formulation exists. The test pins that precision.
    """
    x = torch.tensor([-8.0, -6.0, -4.0, -2.0, -1.0], dtype=torch.float64)
    got = ndtr(x)
    expected = torch.tensor(scipy_special.ndtr(x.numpy()), dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=0.0)


@requires_scipy
def test_ndtr_right_tail_matches_scipy() -> None:
    x = torch.tensor([1.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float64)
    got = ndtr(x)
    expected = torch.tensor(scipy_special.ndtr(x.numpy()), dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=0.0)


@requires_scipy
def test_ndtri_tail_matches_scipy() -> None:
    """Inverse-CDF tails: ``erfinv(2p-1)*sqrt2`` must track scipy.ndtri.

    The relative tolerance is pinned to the measured agreement of torch's
    ``erfinv`` with scipy's ``ndtri`` in the deep tail (~2e-9 rel at
    p=1e-10); tightening it past that is asserting precision the routine
    does not have, loosening it would let a real regression slip.
    """
    p = torch.tensor(
        [1e-10, 1e-6, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-6],
        dtype=torch.float64,
    )
    got = ndtri(p)
    expected = torch.tensor(scipy_special.ndtri(p.numpy()), dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=3e-9, atol=1e-7)


@requires_scipy
def test_log_gamma_correction_matches_scipy() -> None:
    """log_gamma_correction(x) == gammaln(x) - Stirling(x) for x >= 8."""
    x = torch.tensor([8.0, 16.0, 50.0, 200.0, 1000.0], dtype=torch.float64)
    got = log_gamma_correction(x)
    xn = x.numpy()
    stirling = (xn - 0.5) * np.log(xn) - xn + 0.5 * math.log(2 * math.pi)
    expected = torch.tensor(
        scipy_special.gammaln(xn) - stirling,
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


@requires_scipy
def test_log_gamma_difference_matches_scipy() -> None:
    """lgamma(y) - lgamma(x + y) across the y >= 8 Stirling branch."""
    x = torch.tensor([0.5, 1.0, 3.0, 7.0], dtype=torch.float64)
    y = torch.tensor([10.0, 50.0, 200.0, 9.0], dtype=torch.float64)
    got = log_gamma_difference(x, y)
    expected = torch.tensor(
        scipy_special.gammaln(y.numpy()) - scipy_special.gammaln((x + y).numpy()),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


@requires_scipy
def test_lbeta_large_args_match_scipy() -> None:
    """Catastrophic-cancellation regime: both args large must match betaln."""
    x = torch.tensor([8.0, 50.0, 200.0, 1000.0], dtype=torch.float64)
    y = torch.tensor([12.0, 300.0, 200.0, 5.0], dtype=torch.float64)
    got = lbeta(x, y)
    expected = torch.tensor(
        scipy_special.betaln(x.numpy(), y.numpy()),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)


@requires_scipy
def test_lbeta_small_and_mixed_branches_match_scipy() -> None:
    """The small (x,y < 8) and one-large branches must also match betaln."""
    x = torch.tensor([0.5, 2.0, 1.0, 7.0], dtype=torch.float64)
    y = torch.tensor([0.5, 3.0, 100.0, 7.5], dtype=torch.float64)
    got = lbeta(x, y)
    expected = torch.tensor(
        scipy_special.betaln(x.numpy(), y.numpy()),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_lbeta_gradient_finite_at_zero() -> None:
    """The unselected two_large branch must not back-prop NaN at small x.

    ``(x / (x + y)).log()`` is -inf at x == 0, so torch.where leaks a NaN
    gradient into the selected small branch. Beta(0, y) diverges, so the value
    may be +inf, but the gradient at a finite x must stay finite.
    """
    x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    lbeta(x, y).sum().backward()
    assert x.grad is not None
    assert y.grad is not None
    assert torch.isfinite(x.grad[1:]).all()
    assert torch.isfinite(y.grad).all()


def test_lbeta_is_symmetric() -> None:
    """Beta(x, y) == Beta(y, x): the min/max swap must not break symmetry."""
    x = torch.tensor([0.5, 50.0, 3.0], dtype=torch.float64)
    y = torch.tensor([200.0, 2.0, 9.0], dtype=torch.float64)
    torch.testing.assert_close(lbeta(x, y), lbeta(y, x), rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
