from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest.mock import Mock, call

import math


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from priml.math.custom_types import Tensorable
from torch import Tensor
from wrapt import lazy_import

import numpy as np
import pytest
import torch

from priml.math import probability
from priml.math.probability import (
    _normal_cdf_difference,
    _unpack_size,
    cdf_logit_distribution,
    cdf_logit_normal,
    cdf_normal,
    cdf_truncated_normal,
    cdf_uniform,
    gumbel_max,
    lbeta,
    log_cdf_truncated_normal,
    log_gamma_correction,
    log_gamma_difference,
    log_prob_discretized_logistic,
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
    random_gumbel,
    random_logit_normal,
    random_student_t,
)
from priml.memory import convert_to_tensor


class _ScipySpecial(Protocol):
    def ndtr(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def ndtri(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def gammaln(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def betaln(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


# ``scipy`` is the reference oracle for the special-function parity tests only; it
# is an optional test dependency. The lazy proxy defers the real import to
# first attribute access, which only happens inside a parity test body -- and
# those are skipped (not errored) when scipy is absent.
_HAS_SCIPY: Final = find_spec("scipy") is not None
requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
scipy_special = cast(_ScipySpecial, lazy_import("scipy.special"))


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
    # Test against ndtr implementation.
    y = (x - loc) / scale
    expected = ndtr(y)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_quantile_normal():
    p = torch.tensor([0.5, 0.84, 0.16])
    loc, scale = 1.0, 2.0
    result = quantile_normal(p, loc=loc, scale=scale)
    # Check that cdf(quantile(p)) = p.
    cdf_result = cdf_normal(result, loc, scale)
    torch.testing.assert_close(cdf_result, p, atol=1e-5, rtol=1e-5)


def test_quantile_normal_defaults_are_zero_and_one() -> None:
    p = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    torch.testing.assert_close(quantile_normal(p), torch.special.ndtri(p))


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


def test_uniform_density_uses_positive_subunit_span() -> None:
    x = torch.tensor([-0.25, 0.0, 0.25, 0.5, 0.75])
    torch.testing.assert_close(
        pdf_uniform(x, low=0.0, high=0.5),
        torch.tensor([0.0, 2.0, 2.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        cdf_uniform(x, low=0.0, high=0.5),
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
    )


def test_cdf_uniform_degenerate_interval_is_unit_step():
    """High == low: CDF is the unit step at low, not 0/0 == nan."""
    result = cdf_uniform(torch.tensor([0.5, 1.0, 1.5]), 1.0, 1.0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 1.0]))


def test_uniform_functions_use_nonzero_support_bounds() -> None:
    x = torch.tensor([1.0, 2.0, 3.5, 5.0, 6.0])
    torch.testing.assert_close(
        pdf_uniform(x, low=2.0, high=5.0),
        torch.tensor([0.0, 1 / 3, 1 / 3, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        cdf_uniform(x, low=2.0, high=5.0),
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        quantile_uniform(torch.tensor([0.0, 0.25, 0.75, 1.0]), low=2.0, high=5.0),
        torch.tensor([2.0, 2.75, 4.25, 5.0]),
    )


def test_quantile_uniform_out_of_domain_is_nan():
    p = torch.tensor([-0.1, 0.5, 2.0, math.nan])
    result = quantile_uniform(p, 0.0, 1.0)
    torch.testing.assert_close(
        result,
        torch.tensor([math.nan, 0.5, math.nan, math.nan]),
        equal_nan=True,
    )


def test_pdf_logit_normal():
    x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    loc, scale = 0.6, 1.7
    result = pdf_logit_normal(x, loc, scale)
    z = torch.logit(x)
    expected = torch.exp(
        -0.5 * ((z - loc) / scale).square()
        - math.log(scale * (2 * math.pi) ** 0.5)
        - torch.log(x)
        - torch.log1p(-x),
    )
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_cdf_logit_normal():
    x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    loc, scale = 0.6, 1.7
    result = cdf_logit_normal(x, loc, scale)
    expected = torch.special.ndtr((torch.logit(x) - loc) / scale)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_quantile_logit_normal():
    p = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    loc, scale = 0.6, 1.7
    result = quantile_logit_normal(p, loc, scale)
    expected = torch.sigmoid(loc + scale * torch.special.ndtri(p))
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_quantile_truncated_normal():
    p = torch.tensor([0.0, 0.5, 1.0])
    loc, scale, low, high = 0.0, 1.0, -2.0, 2.0
    result = quantile_truncated_normal(p, loc=loc, scale=scale, low=low, high=high)
    assert result.shape == p.shape
    # Results should be within bounds.
    assert torch.all(result >= low)
    assert torch.all(result <= high)


def test_truncated_normal_cdf_matches_asymmetric_reference() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    loc, scale, low, high = 0.4, 1.3, -0.7, 2.2
    result = cdf_truncated_normal(x, loc=loc, scale=scale, low=low, high=high)
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    low_cdf = torch.special.ndtr(torch.tensor(std_low, dtype=x.dtype))
    high_cdf = torch.special.ndtr(torch.tensor(std_high, dtype=x.dtype))
    expected = (
        (torch.special.ndtr((x - loc) / scale) - low_cdf) / (high_cdf - low_cdf)
    ).clamp(0, 1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_truncated_normal_quantile_inverts_asymmetric_cdf() -> None:
    p = torch.tensor([0.1, 0.35, 0.8], dtype=torch.float64)
    low, high, loc, scale = -0.6, 1.8, 0.7, 1.4
    quantile = quantile_truncated_normal(p, loc=loc, scale=scale, low=low, high=high)
    observed = cdf_truncated_normal(quantile, loc=loc, scale=scale, low=low, high=high)
    torch.testing.assert_close(observed, p, rtol=1e-12, atol=1e-12)


def test_truncated_quantile_defaults_use_zero_location_unit_scale() -> None:
    p = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    low, high = -0.5, 1.5
    low_cdf = torch.special.ndtr(torch.tensor(low, dtype=p.dtype))
    high_cdf = torch.special.ndtr(torch.tensor(high, dtype=p.dtype))
    expected = torch.special.ndtri(low_cdf + p * (high_cdf - low_cdf))
    torch.testing.assert_close(
        quantile_truncated_normal(p, low=low, high=high),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_truncated_normal_defaults_use_zero_location_and_unit_scale() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    low, high = -0.5, 1.5
    result = cdf_truncated_normal(x, low=low, high=high)
    low_cdf = torch.special.ndtr(torch.tensor(low, dtype=x.dtype))
    high_cdf = torch.special.ndtr(torch.tensor(high, dtype=x.dtype))
    expected = ((torch.special.ndtr(x) - low_cdf) / (high_cdf - low_cdf)).clamp(0, 1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_truncated_cdf_branches_have_exact_values() -> None:
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    low, high = -1.0, 1.0
    cdf = cdf_truncated_normal(x, low=low, high=high)
    log_cdf = log_cdf_truncated_normal(x, low=low, high=high)
    low_cdf = torch.special.ndtr(torch.tensor(low, dtype=x.dtype))
    span = torch.special.ndtr(torch.tensor(high, dtype=x.dtype)) - low_cdf
    expected = (torch.special.ndtr(x) - low_cdf) / span
    torch.testing.assert_close(cdf, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(log_cdf, expected.log(), rtol=1e-12, atol=1e-12)
    assert cdf[0] == 0
    assert cdf[-1] == 1
    assert log_cdf[0] == -math.inf
    assert log_cdf[-1] == 0


def test_normal_cdf_difference_switches_at_zero_with_exact_values() -> None:
    b = torch.tensor([-0.25, 0.0, 0.25, 0.99], dtype=torch.float64)
    a = torch.tensor([0.25, 0.5, 0.75, 0.990000000001], dtype=torch.float64)
    expected = torch.stack(
        [
            torch.special.ndtr(a[0]) - torch.special.ndtr(b[0]),
            torch.special.ndtr(-b[1]) - torch.special.ndtr(-a[1]),
            torch.special.ndtr(-b[2]) - torch.special.ndtr(-a[2]),
            torch.special.ndtr(-b[3]) - torch.special.ndtr(-a[3]),
        ],
    )
    assert torch.equal(_normal_cdf_difference(a, b), expected)


@pytest.mark.parametrize("a", [1e-3, 0.3, 2.0])
def test_normal_cdf_difference_zero_uses_exact_flipped_reference(a: float) -> None:
    value = torch.tensor(a, dtype=torch.float64)
    result = _normal_cdf_difference(value, torch.tensor(0.0, dtype=torch.float64))
    expected = 0.5 - torch.special.ndtr(-value)
    assert torch.equal(result, expected)


def test_ndtr_signbit_rewrite_matches_old_formula_bit_exactly() -> None:
    grid = torch.linspace(-10.0, 10.0, 20_001, dtype=torch.float64)
    x = torch.cat((grid, torch.tensor([0.0, -0.0], dtype=grid.dtype)))
    t = x * 0.5**0.5
    z = t.abs()
    expected = 0.5 * torch.where(
        z < 0.5**0.5,
        1 + torch.erf(t),
        torch.where(t > 0.0, 2 - torch.erfc(z), torch.erfc(z)),
    )
    assert torch.equal(ndtr(x), expected)


def test_ndtr_exactly_matches_both_piecewise_branches_and_boundary() -> None:
    boundary = torch.tensor(1.0, dtype=torch.float64)
    x = torch.stack(
        [
            torch.nextafter(-boundary, torch.zeros_like(boundary)),
            -boundary,
            torch.nextafter(-boundary, torch.tensor(-math.inf)),
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.tensor(math.inf)),
            torch.tensor(0.3, dtype=torch.float64),
            torch.tensor(0.6, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
        ],
    )
    scaled = x * 0.5**0.5
    z = scaled.abs()
    expected = 0.5 * torch.where(
        z < 0.5**0.5,
        1 + torch.erf(scaled),
        torch.where(scaled > 0, 2 - torch.erfc(z), torch.erfc(z)),
    )
    torch.testing.assert_close(ndtr(x), expected, rtol=0, atol=0)
    at_zero = ndtr(torch.tensor([0.0], dtype=torch.float64))
    torch.testing.assert_close(
        at_zero,
        torch.tensor([0.5], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_unpack_size_accepts_tuple_or_positional_dimensions() -> None:
    assert _unpack_size((2, 3)) == (2, 3)
    assert _unpack_size(2, 3) == (2, 3)
    assert _unpack_size(2) == (2,)


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
        result,
        torch.tensor([-math.inf, 0.0, 0.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "function",
    [cdf_uniform, pdf_uniform, cdf_truncated_normal, log_cdf_truncated_normal],
)
def test_degenerate_probability_gradients_are_finite_and_zero(
    function: Callable[..., Tensor],
) -> None:
    x = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    low = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    high = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    function(x, low=low, high=high).sum().backward()

    for value in (x.grad, low.grad, high.grad):
        if value is not None:
            assert torch.isfinite(value).all()
            assert torch.equal(value, torch.zeros_like(value))


def test_quantile_truncated_normal_out_of_domain_is_nan():
    p = torch.tensor([-0.1, 0.5, 2.0, math.nan])
    result = quantile_truncated_normal(p, low=-2.0, high=2.0)
    assert result[[0, 2, 3]].isnan().all()
    torch.testing.assert_close(result[1], torch.tensor(0.0), atol=1e-6, rtol=0)


def test_quantile_truncated_normal_sanitizes_out_of_domain_gradient() -> None:
    p = torch.tensor([-0.1, 0.4, 1.1], requires_grad=True)
    result = quantile_truncated_normal(p, low=-2.0, high=2.0)
    result[1].backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad[1])
    torch.testing.assert_close(result[1], torch.tensor(-0.24158715), atol=1e-6, rtol=0)


def test_random_categorical_all_neg_inf_logits_propagate_nan():
    result = random_categorical(
        5,
        logits=torch.tensor([-math.inf, -math.inf, -math.inf]),
    )
    assert result.shape == (5,)
    assert torch.equal(result, torch.zeros(5, dtype=torch.long))


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


def test_logit_cdf_passes_finite_logits_and_matches_sigmoid() -> None:
    x = torch.tensor([-0.1, 0.0, 0.2, 0.7, 1.0, 1.1])
    base_cdf = Mock(wraps=torch.sigmoid)

    result = cdf_logit_distribution(x, base_cdf)

    logits = base_cdf.call_args.args[0]
    assert isinstance(logits, Tensor)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[x <= 0], torch.zeros_like(logits[x <= 0]))
    assert torch.equal(logits[x >= 1], torch.zeros_like(logits[x >= 1]))
    torch.testing.assert_close(result, x.clamp(0, 1))


def test_logit_pdf_passes_finite_logits_and_has_exact_jacobian() -> None:
    x = torch.tensor([-0.1, 0.0, 0.2, 0.7, 1.0, 1.1], dtype=torch.float64)
    base_pdf = Mock(wraps=torch.ones_like)

    result = pdf_logit_distribution(x, base_pdf)

    logits = base_pdf.call_args.args[0]
    assert isinstance(logits, Tensor)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[x <= 0], torch.zeros_like(logits[x <= 0]))
    assert torch.equal(logits[x >= 1], torch.zeros_like(logits[x >= 1]))
    expected = torch.where(
        (x > 0) & (x < 1),
        1 / (x * (1 - x)),
        torch.zeros_like(x),
    )
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_pdf_logit_distribution():
    x = torch.tensor([0.1, 0.5, 0.9])

    def base_pdf(x: Tensorable) -> Tensor:
        return pdf_normal(x, 0.0, 1.0)

    result = pdf_logit_distribution(x, base_pdf)
    assert result.shape == x.shape
    assert torch.all(result > 0)


def test_quantile_logit_distribution():
    p = torch.tensor([0.1, 0.5, 0.9])

    def base_quantile(p: Tensorable) -> Tensor:
        return quantile_normal(p, loc=0.0, scale=1.0)

    result = quantile_logit_distribution(p, base_quantile)
    assert result.shape == p.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_random_logit_normal_default_parameters_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standard_normal = torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    randn = Mock(return_value=standard_normal)
    monkeypatch.setattr(torch, "randn", randn)

    result = random_logit_normal(1, 2, dtype=torch.float64)

    torch.testing.assert_close(result, torch.sigmoid(standard_normal))
    assert result.dtype is torch.float64
    assert randn.call_args == call(
        1,
        2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


def test_random_logit_normal_uses_requested_location_and_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loc = torch.tensor([[0.3], [-0.4]])
    scale = torch.tensor([0.5, 1.7, 2.0])
    normal = torch.tensor([[0.2, -0.1, 0.0], [1.0, 0.7, -0.5]])
    randn = Mock(return_value=normal)
    monkeypatch.setattr(torch, "randn", randn)
    device = torch.device("cpu")

    result = random_logit_normal(1, loc=loc, scale=scale, device=device)

    torch.testing.assert_close(
        result,
        torch.sigmoid(normal * scale + loc).unsqueeze(0),
    )
    assert randn.call_args == call(1, 2, 3, dtype=scale.dtype, device=device)


def test_random_gamma_default_rate_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    sampled = torch.tensor([2.0, 3.0])
    sampler = Mock(return_value=sampled)
    monkeypatch.setattr(torch, "_standard_gamma", sampler)

    result = random_gamma(2, concentration=torch.tensor([1.5, 2.5]))

    torch.testing.assert_close(result, sampled)
    torch.testing.assert_close(
        sampler.call_args.args[0],
        torch.tensor([[1.5, 2.5], [1.5, 2.5]]),
    )


def test_random_gamma_converter_receives_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled: list[Tensor] = []
    gamma_draw = torch.ones(1, dtype=torch.float64)

    def standard_gamma(concentration: Tensor) -> Tensor:
        sampled.append(concentration)
        return gamma_draw

    monkeypatch.setattr(torch, "_standard_gamma", standard_gamma)
    concentration = torch.tensor([2.0])
    device = torch.device("cpu")

    result = random_gamma(
        1,
        concentration=concentration,
        dtype=torch.float64,
        device=device,
    )

    assert result.dtype is torch.float64
    assert len(sampled) == 1
    assert sampled[0].dtype is torch.float64
    assert sampled[0].device == device
    torch.testing.assert_close(sampled[0], torch.tensor([[2.0]], dtype=torch.float64))


def test_random_gamma_uses_rate_and_clamps_underflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concentration = torch.tensor([2.0, 4.0, 6.0])
    rate = torch.tensor([2.0, 0.5, 1.0])
    standard_gamma = torch.tensor([[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    sampler = Mock(return_value=standard_gamma)
    monkeypatch.setattr(torch, "_standard_gamma", sampler)

    result = random_gamma(2, concentration=concentration, rate=rate)

    # random_gamma expands batch size over the concentration vector.
    expected_concentration = concentration.expand(2, 3)
    sampled_concentration = sampler.call_args.args[0]
    assert isinstance(sampled_concentration, Tensor)
    assert torch.equal(sampled_concentration, expected_concentration)
    torch.testing.assert_close(result, standard_gamma / rate)

    sampler.return_value = torch.zeros(1)
    underflow = random_gamma(1, concentration=1.0, rate=1e38)
    assert torch.equal(
        underflow,
        torch.full_like(underflow, torch.finfo(torch.float32).tiny),
    )


def test_random_chi2_delegates_half_df_and_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = torch.tensor([2.0, 6.0])
    gamma_result = torch.tensor([[4.0, 8.0], [5.0, 9.0]])
    gamma = Mock(return_value=gamma_result)
    monkeypatch.setattr(probability, "random_gamma", gamma)

    result = random_chi2(2, df=df)

    assert gamma.call_args.args == (2,)
    torch.testing.assert_close(gamma.call_args.kwargs["concentration"], df / 2)
    assert gamma.call_args.kwargs["rate"] == 0.5
    assert result is gamma_result


def test_random_chi2_converter_receives_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gamma = Mock(return_value=torch.ones(1, dtype=torch.float64))
    monkeypatch.setattr(probability, "random_gamma", gamma)
    df = torch.tensor([4.0])
    device = torch.device("cpu")

    result = random_chi2(1, df=df, dtype=torch.float64, device=device)

    assert result.dtype is torch.float64
    concentration = gamma.call_args.kwargs["concentration"]
    assert isinstance(concentration, Tensor)
    assert concentration.dtype is torch.float64
    assert concentration.device == device
    torch.testing.assert_close(concentration, df.to(dtype=torch.float64) / 2)
    assert gamma.call_args.kwargs["rate"] == 0.5


def test_student_t_default_location_and_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = torch.tensor([4.0, 8.0])
    normal = torch.tensor([[0.5, -1.0]])
    chi2 = torch.tensor([[2.0, 8.0]])
    randn = Mock(return_value=normal)
    gamma = Mock(return_value=chi2)
    monkeypatch.setattr(torch, "randn", randn)
    monkeypatch.setattr(probability, "random_gamma", gamma)
    device = torch.device("cpu")

    result = random_student_t(1, df=df, dtype=torch.float64, device=device)

    torch.testing.assert_close(result, normal * torch.rsqrt(chi2))
    assert randn.call_args == call(1, 2, dtype=torch.float64, device=device)
    expected_half_df = df.to(dtype=torch.float64) / 2
    torch.testing.assert_close(
        gamma.call_args.kwargs["concentration"],
        expected_half_df,
    )
    torch.testing.assert_close(gamma.call_args.kwargs["rate"], expected_half_df)


def test_student_t_fixed_draws_apply_df_location_and_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = torch.tensor([4.0, 8.0, 12.0])
    loc = torch.tensor([0.2, -0.3, 1.1])
    scale = torch.tensor([0.5, 1.4, 2.0])
    normal = torch.tensor([[0.2, -0.1, 0.7], [1.0, 0.7, -0.2]])
    chi2 = torch.tensor([[3.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    randn = Mock(return_value=normal)
    gamma = Mock(return_value=chi2)
    monkeypatch.setattr(torch, "randn", randn)
    monkeypatch.setattr(probability, "random_gamma", gamma)

    result = random_student_t(2, df=df, loc=loc, scale=scale)

    expected = loc + scale * normal * torch.rsqrt(chi2)
    torch.testing.assert_close(result, expected)
    assert randn.call_args == call(2, 3, dtype=df.dtype, device=df.device)
    torch.testing.assert_close(gamma.call_args.kwargs["concentration"], df / 2)
    torch.testing.assert_close(gamma.call_args.kwargs["rate"], df / 2)


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
    # Var = scale^2 * df / (df - 2) for df > 2.
    df_t = convert_to_tensor(df)
    expected_std = scale * torch.sqrt(df_t / (df_t - 2))
    actual_std = x.std(dim=0).squeeze(0)
    torch.testing.assert_close(actual_std, expected_std, atol=0.15, rtol=0.12)


def test_random_chi2():
    # Test basic chi2 random generation - just ensure it runs.
    n = 1_000
    df = torch.tensor([2.0])
    x = random_chi2(n, df=df)
    # Just check it returns data.
    assert x.shape[0] == n
    assert torch.all(x > 0)  # Chi2 is always positive.


def test_gamma():
    n, concentration, rate = 150_000, [1.0, 2.5, 5.0], 2.0
    conc_t = torch.tensor(concentration)
    x = random_gamma(n, concentration=concentration, rate=rate)
    # Mean = concentration / rate.
    mean = x.mean(dim=0)
    expected_mean = conc_t / rate
    torch.testing.assert_close(mean, expected_mean, atol=0.08, rtol=0.08)
    # Var = concentration / rate^2.
    std = x.std(dim=0)
    expected_std = torch.sqrt(conc_t / rate**2)
    torch.testing.assert_close(std, expected_std, atol=0.12, rtol=0.12)


def test_gamma_with_rate():
    n, concentration, rate = 10_000, 2.0, 0.5
    x = random_gamma(n, concentration=concentration, rate=rate)
    # Gamma mean = concentration / rate.
    mean = x.mean()
    expected_mean = concentration / rate
    np.testing.assert_allclose(mean, expected_mean, atol=0, rtol=0.1)


@pytest.mark.parametrize("use_logits", [True, False])
def test_categorical(use_logits: bool):
    n = 80_000
    probs_t = torch.tensor([[0.15, 0.35, 0.5], [0.6, 0.0, 0.4]])
    if use_logits:
        logits_t = probs_t.log()  # -inf for zero entries via torch.
        x = random_categorical(n, logits=logits_t)
    else:
        x = random_categorical(n, probs=probs_t)
    # Compute empirical frequencies with a simple loop over categories.
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
    error = r"\ASpecify exactly one of probs or logits\.\Z"
    with pytest.raises(ValueError, match=error):
        random_categorical(100, probs=[0.5, 0.5], logits=[0.0, 0.0])
    with pytest.raises(ValueError, match=error):
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


def test_random_categorical_uses_cumulative_probability_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.4, 0.0]])
    # Pin the same boundary probes on each independent batch row.
    draws = torch.tensor([0.1, 0.2, 0.5, 0.6, 0.9]).reshape(5, 1, 1).expand(5, 2, 1)
    rand = Mock(return_value=draws)
    monkeypatch.setattr(torch, "rand", rand)

    samples = random_categorical(5, probs=probabilities)

    assert torch.equal(
        samples,
        torch.tensor([[0, 0], [1, 0], [2, 0], [2, 1], [2, 1]]),
    )
    assert rand.call_args == call(
        5,
        2,
        1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )


def test_random_categorical_normalizes_unnormalized_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = torch.tensor([2.0, 3.0])
    draw = torch.tensor([[0.5]])
    monkeypatch.setattr(torch, "rand", Mock(return_value=draw))

    samples = random_categorical(1, probs=probabilities)

    assert torch.equal(samples, torch.tensor([1]))


def test_random_categorical_all_negative_infinity_singleton_propagates_nan() -> None:
    result = random_categorical(1, logits=torch.tensor([-math.inf]))
    assert torch.equal(result, torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("use_logits", [False, True])
def test_random_categorical_conversion_uses_requested_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
    use_logits: bool,
) -> None:
    draws: list[
        tuple[tuple[int, ...], torch.dtype | None, torch.device | str | None]
    ] = []

    def random_draw(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        draws.append((size, dtype, device))
        return torch.zeros(*size, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "rand", random_draw)
    values = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected_device = torch.device("cpu")
    inputs = {"logits": values.log()} if use_logits else {"probs": values}

    samples = random_categorical(
        2,
        **inputs,
        dtype=torch.float32,
        device=expected_device,
    )

    assert samples.shape == (2,)
    assert draws == [((2, 1), torch.float32, expected_device)]


def test_random_categorical_logits_match_probability_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    # random_categorical draws have the production sample-by-category shape.
    draws = torch.tensor([0.1, 0.2, 0.5, 0.9]).reshape(4, 1)
    monkeypatch.setattr(torch, "rand", Mock(return_value=draws))

    samples = random_categorical(4, logits=probabilities.log() + 2.7)

    assert torch.equal(samples, torch.tensor([0, 1, 2, 2]))


def test_random_logit_normal():
    n, loc, scale = 10_000, 0.0, 1.0
    x = random_logit_normal(n, loc=loc, scale=scale)
    assert x.shape == (n,)
    assert torch.all(x >= 0)
    assert torch.all(x <= 1)


def test_random_logit_normal_with_sequence():
    # Test with sequence input.
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
    # Just test normal categorical behavior.
    x = random_categorical(10, probs=[0.5, 0.5])
    assert x.shape == (10,)


def test_random_logit_normal_with_tuple():
    # Test with tuple input (line 295)
    n = (5, 10)  # Pass as tuple (sequence)
    x = random_logit_normal(*n, loc=0.0, scale=1.0)
    assert x.shape == (5, 10)


def test_log_cdf_truncated_normal() -> None:
    x = torch.tensor([-0.7, -0.3, 0.4, 1.1, 2.2], dtype=torch.float64)
    loc, scale, low, high = 0.4, 1.3, -0.7, 2.2
    log_cdf = log_cdf_truncated_normal(
        x,
        loc=loc,
        scale=scale,
        low=low,
        high=high,
    )
    std_x = (x - loc) / scale
    std_low = torch.tensor((low - loc) / scale, dtype=x.dtype)
    std_high = torch.tensor((high - loc) / scale, dtype=x.dtype)
    numerator = torch.special.ndtr(std_x) - torch.special.ndtr(std_low)
    denominator = torch.special.ndtr(std_high) - torch.special.ndtr(std_low)
    expected = torch.log(numerator / denominator)
    torch.testing.assert_close(log_cdf, expected, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize(
    ("loc", "log_scale"),
    [(0.3, -3.0), (-0.99, -5.0), (0.0, 1.0), (1.4, -6.0)],
)
def test_discretized_logistic_masses_sum_to_one(loc: float, log_scale: float) -> None:
    values = torch.arange(256)
    log_prob = log_prob_discretized_logistic(values, loc, log_scale)
    torch.testing.assert_close(
        torch.logsumexp(log_prob.double(), 0),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )


def test_discretized_logistic_matches_the_cdf_difference() -> None:
    x = torch.tensor([0, 100, 255])
    loc = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float64)
    scale = torch.tensor([-2.0, -1.0, -3.0], dtype=torch.float64).exp()
    centre = x.double() / 127.5 - 1
    upper = torch.sigmoid((centre + 1 / 255 - loc) / scale)
    lower = torch.sigmoid((centre - 1 / 255 - loc) / scale)
    mass = torch.stack([upper[0], upper[1] - lower[1], 1 - lower[2]])
    torch.testing.assert_close(
        log_prob_discretized_logistic(x, loc, scale.log()),
        mass.log(),
    )


def test_discretized_logistic_is_accurate_far_in_the_tail() -> None:
    """A bin 40 scales above the location: the naive CDF difference is 0."""
    loc, log_scale = torch.tensor(-0.5), torch.tensor(-4.0)
    log_prob = log_prob_discretized_logistic(torch.tensor(100), loc, log_scale)
    centre, half, scale = 100 / 127.5 - 1, 1 / 255, math.exp(-4.0)
    # log(σ(-l) - σ(-u)) with -l, -u < 0, where σ(z) ≈ e^z to 1e-17.
    upper, lower = (centre + half + 0.5) / scale, (centre - half + 0.5) / scale
    expected = -lower + math.log1p(-math.exp(lower - upper))
    torch.testing.assert_close(log_prob, torch.tensor(expected), rtol=1e-5, atol=0)


def test_discretized_logistic_gradient_is_finite_in_the_tails() -> None:
    x = torch.tensor([0, 3, 128, 252, 255])
    loc = torch.tensor([5.0, -5.0, 0.0, 5.0, -5.0], requires_grad=True)
    log_scale = torch.full((5,), -6.0, requires_grad=True)
    log_prob_discretized_logistic(x, loc, log_scale).sum().backward()
    assert loc.grad is not None
    assert log_scale.grad is not None
    assert torch.isfinite(loc.grad).all()
    assert torch.isfinite(log_scale.grad).all()


def test_random_gumbel_is_minus_log_minus_log_of_the_generators_uniforms() -> None:
    uniform = torch.rand(2, 3, generator=torch.Generator().manual_seed(7))
    noise = random_gumbel(2, 3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(noise, -(-uniform.log()).log())


def test_gumbel_max_draws_the_bits_of_subtracting_log_minus_log_uniforms() -> None:
    logits = torch.randn(3, 5, generator=torch.Generator().manual_seed(0))
    logits[0, 1] = float("-inf")
    ours, theirs = torch.Generator().manual_seed(8), torch.Generator().manual_seed(8)
    # The spelling gumbel_max replaced: one float32 draw of the logits' shape,
    # whose log(-log U) is subtracted rather than its negation added.
    uniform = torch.rand(logits.shape, generator=theirs, dtype=torch.float32)
    scores = logits - (-uniform.log()).log()
    noise = random_gumbel(3, 5, generator=torch.Generator().manual_seed(8))
    assert torch.equal(logits + noise, scores)
    assert torch.equal(gumbel_max(logits, generator=ours), scores.argmax(-1))
    assert torch.equal(ours.get_state(), theirs.get_state())


def test_a_zero_uniform_is_minus_infinite_noise_that_never_makes_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uniform = torch.tensor([[0.0, 0.25, 0.0], [0.5, 0.0, 0.75]])

    def fixed(size: tuple[int, ...], **_: object) -> Tensor:
        return uniform.reshape(size)

    monkeypatch.setattr(torch, "rand", fixed)
    noise = random_gumbel(2, 3)
    assert torch.equal(noise.isneginf(), uniform == 0)
    logits = torch.tensor([[float("-inf"), 1.0, 0.0], [float("-inf"), 2.0, 0.0]])
    assert not (logits + noise).isnan().any()
    # Each row's masked entry and its zero-uniform entries lose to the rest.
    assert gumbel_max(logits).tolist() == [1, 2]


def test_gumbel_max_frequencies_match_softmax() -> None:
    logits = torch.tensor([0.5, -1.0, 2.0, 0.0, float("-inf"), 1.0])
    draws = 20_000
    samples = gumbel_max(
        logits.expand(draws, -1),
        generator=torch.Generator().manual_seed(4),
    )
    counts = torch.bincount(samples, minlength=6).double()
    assert counts[4] == 0
    expected = logits.softmax(-1).double() * draws
    finite = expected > 0
    chi_square = ((counts - expected)[finite] ** 2 / expected[finite]).sum()
    # 18.47 is the 0.999 quantile of chi-square with 4 degrees of freedom.
    assert float(chi_square) < 18.47, float(chi_square)


def test_probability_distribution_defaults_match_standard_parameters() -> None:
    normal_x = torch.tensor([-1.0, 0.25, 1.5], dtype=torch.float64)
    normal = torch.distributions.Normal(0.0, 1.0)
    torch.testing.assert_close(pdf_normal(normal_x), normal.log_prob(normal_x).exp())
    torch.testing.assert_close(cdf_normal(normal_x), torch.special.ndtr(normal_x))

    unit_x = torch.tensor([-0.1, 0.25, 0.75, 1.0, 1.2], dtype=torch.float64)
    torch.testing.assert_close(
        pdf_uniform(unit_x),
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0], dtype=unit_x.dtype),
    )
    torch.testing.assert_close(cdf_uniform(unit_x), unit_x.clamp(0.0, 1.0))

    logit_x = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
    logits = torch.logit(logit_x)
    torch.testing.assert_close(cdf_logit_normal(logit_x), torch.special.ndtr(logits))
    torch.testing.assert_close(
        pdf_logit_normal(logit_x),
        normal.log_prob(logits).exp() / (logit_x * (1 - logit_x)),
    )
    probabilities = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
    torch.testing.assert_close(
        quantile_logit_normal(probabilities),
        torch.sigmoid(torch.special.ndtri(probabilities)),
    )
    torch.testing.assert_close(quantile_uniform(probabilities), probabilities)


def test_unpack_size_reports_invalid_dimension_type() -> None:
    with pytest.raises(TypeError) as error:
        _unpack_size(2, (3, 4))

    assert str(error.value) == "sample dimensions must be integers"


def test_lbeta_uses_exact_stirling_branch_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def correction(value: Tensor) -> Tensor:
        return value.square()

    def difference(x: Tensor, y: Tensor) -> Tensor:
        del y
        return torch.full_like(x, 1000.0)

    monkeypatch.setattr(probability, "log_gamma_correction", correction)
    monkeypatch.setattr(probability, "log_gamma_difference", difference)
    x = torch.tensor([8.0, 8.5, 7.5], dtype=torch.float64)
    y = torch.tensor([10.0, 10.0, 8.0], dtype=torch.float64)

    result = lbeta(x, y)

    safe_x = x.clamp_min(8.0)
    two_large = (
        0.5 * math.log(2 * math.pi)
        - 0.5 * y.log()
        + correction(safe_x)
        + correction(y)
        - correction(safe_x + y)
        + (safe_x - 0.5) * (safe_x / (safe_x + y)).log()
        - y * torch.log1p(safe_x / y)
    )
    one_large = torch.lgamma(x[2:]) + difference(x[2:], y[2:])
    torch.testing.assert_close(
        result,
        torch.cat((two_large[:2], one_large)),
        rtol=0,
        atol=0,
    )


def test_log_gamma_correction_matches_lgamma_at_the_stirling_boundary() -> None:
    values = torch.tensor([8.0, 8.5, 16.0, 50.0], dtype=torch.float64)
    expected = torch.tensor(
        [
            math.lgamma(value)
            - ((value - 0.5) * math.log(value) - value + 0.5 * math.log(2 * math.pi))
            for value in (8.0, 8.5, 16.0, 50.0)
        ],
        dtype=values.dtype,
    )

    torch.testing.assert_close(
        log_gamma_correction(values),
        expected,
        rtol=1e-12,
        atol=2e-14,
    )


def test_log_gamma_correction_preserves_the_input_device() -> None:
    values = torch.empty((2,), dtype=torch.float64, device="meta")

    assert log_gamma_correction(values).device == values.device


def test_log_gamma_difference_uses_stirling_branch_at_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def correction(value: Tensor) -> Tensor:
        return value.square()

    monkeypatch.setattr(probability, "log_gamma_correction", correction)
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    y = torch.tensor([8.0, 8.5], dtype=torch.float64)

    result = log_gamma_difference(x, y)

    cancelled_stirling = -(x + y - 0.5) * torch.log1p(x / y) - x * y.log() + x
    correction_term = correction(y) - correction(x + y)
    torch.testing.assert_close(
        result,
        cancelled_stirling + correction_term,
        rtol=0,
        atol=0,
    )


def test_log_gamma_difference_small_arguments_match_lgamma() -> None:
    x = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    y = torch.tensor([0.5, 2.0, 7.5], dtype=torch.float64)
    expected = torch.tensor(
        [
            math.lgamma(right) - math.lgamma(left + right)
            for left, right in ((0.5, 0.5), (1.0, 2.0), (3.0, 7.5))
        ],
        dtype=x.dtype,
    )

    torch.testing.assert_close(
        log_gamma_difference(x, y),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )


def test_discretized_logistic_uses_the_stable_left_tail_branch() -> None:
    x = torch.tensor([128.0], dtype=torch.float32)
    loc = torch.tensor([100.0], dtype=torch.float32)
    log_scale = torch.tensor([-2.0], dtype=torch.float32)
    half = 1 / 255
    centre = x * (2 * half) - 1
    inverse = torch.exp(-log_scale)
    upper = (centre - loc + half) * inverse
    lower = (centre - loc - half) * inverse
    gap = torch.nn.functional.logsigmoid(lower) - torch.nn.functional.logsigmoid(upper)
    expected = torch.nn.functional.logsigmoid(upper) + torch.log(-torch.expm1(gap))

    result = log_prob_discretized_logistic(x, loc, log_scale)

    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_discretized_logistic_tail_clamp_keeps_collapsed_edges_finite() -> None:
    result = log_prob_discretized_logistic(
        torch.tensor([128]),
        torch.tensor([1.0e8]),
        torch.tensor([0.0]),
    )

    assert torch.isfinite(result).all()


def test_random_categorical_accepts_negative_infinity_before_finite_logit() -> None:
    samples = random_categorical(8, logits=torch.tensor([-math.inf, 0.0]))

    assert torch.equal(samples, torch.ones_like(samples))


def test_random_sampling_forwards_requested_conversion_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_convert = convert_to_tensor
    requested = torch.device("cpu")
    observed: list[torch.device | str | None] = []

    def convert(
        *values: Tensorable,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        observed.append(device)
        return real_convert(
            *values,
            dtype=dtype,
            device=device,
            dtype_hint=dtype_hint,
        )

    monkeypatch.setattr(probability, "convert_to_tensor", convert)
    start = len(observed)
    random_categorical(1, probs=torch.tensor([0.25, 0.75]), device=requested)
    assert observed[start] == requested

    start = len(observed)
    random_categorical(1, logits=torch.tensor([0.25, 0.75]), device=requested)
    assert observed[start] == requested

    start = len(observed)
    random_chi2(1, df=4.0, device=requested)
    assert observed[start] == requested

    start = len(observed)
    random_gamma(1, concentration=4.0, device=requested)
    assert observed[start] == requested

    start = len(observed)
    random_logit_normal(1, device=requested)
    assert observed[start] == requested

    start = len(observed)
    random_student_t(1, df=4.0, device=requested)
    assert observed[start] == requested


def test_random_logit_normal_default_dtype_hint_is_float32() -> None:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result = random_logit_normal(2)
    finally:
        torch.set_default_dtype(previous)

    assert result.dtype is torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ndtri_exact_tail_cases(dtype: torch.dtype) -> None:
    p = torch.tensor([1e-30, 1e-10, 1e-8, 0.5, 1 - 1e-6, 0.0, 1.0], dtype=dtype)
    expected = torch.special.ndtri(p)
    torch.testing.assert_close(
        ndtri(p),
        expected,
        rtol=1e-6 if dtype == torch.float32 else 1e-14,
        atol=0,
    )


@pytest.mark.parametrize("use_logits", [False, True])
def test_categorical_batch_rows_are_independent(use_logits: bool) -> None:
    probs = torch.full((2, 3), 1 / 3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        samples = (
            random_categorical(300, logits=probs.log())
            if use_logits
            else random_categorical(300, probs=probs)
        )
    agreement = (samples[:, 0] == samples[:, 1]).float().mean()
    assert 0.2 < agreement < 0.5


@pytest.mark.parametrize(
    "probs",
    [[0.0, 0.0], [-1.0, 2.0], [math.nan, 1.0], [math.inf, 1.0]],
)
def test_categorical_invalid_probability_rows_propagate_nan(
    probs: list[float],
) -> None:
    result = random_categorical(2, probs=probs)
    assert result.shape == (2,)


@pytest.mark.parametrize("reverse", [False, True])
def test_truncated_cdf_saturates_at_support(reverse: bool) -> None:
    low, high = (1.0, 0.0) if reverse else (0.0, 1.0)
    x = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    expected = torch.tensor([1.0, 0.0] if reverse else [0.0, 1.0], dtype=x.dtype)
    torch.testing.assert_close(
        cdf_truncated_normal(x, low=low, high=high),
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        log_cdf_truncated_normal(x, low=low, high=high),
        expected.log(),
        rtol=0,
        atol=0,
    )


def test_point_mass_log_cdf_boundary_gradient() -> None:
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    low = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    high = low.detach().clone().requires_grad_()
    log_cdf_truncated_normal(x, low=low, high=high).sum().backward()
    for value in (x, low, high):
        assert value.grad is not None
        torch.testing.assert_close(value.grad, torch.zeros_like(value), rtol=0, atol=0)


def test_discretized_logistic_collapsed_float16_edges() -> None:
    result = log_prob_discretized_logistic(
        torch.tensor([128.0, 129.0], dtype=torch.float16),
        torch.tensor(100.0, dtype=torch.float16),
        torch.tensor(0.0, dtype=torch.float16),
    )
    assert result.isfinite().all()


@pytest.mark.parametrize("function", [quantile_uniform, quantile_truncated_normal])
def test_quantile_fullgraph(function: Callable[..., Tensor]) -> None:
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    p = torch.tensor([0.25, 0.75])
    torch.testing.assert_close(compiled(p), function(p))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
