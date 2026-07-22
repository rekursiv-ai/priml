from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import functools
import math

from torch import Tensor, distributions

import torch

from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.numeric import logsubexp


# Adapted from tensorflow/probability.
def pdf_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """PDF of a normal distribution.

    Returns:
      pdf: Probability density at x.

    """
    x, loc, scale = convert_to_tensor(x, loc, scale)
    return distributions.Normal(loc, scale).log_prob(x).exp()


# Adapted from tensorflow/probability.
def cdf_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """CDF of a normal distribution.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, loc, scale = convert_to_tensor(x, loc, scale)
    return ndtr((x - loc) / scale)


# Adapted from tensorflow/probability.
def quantile_normal(
    p: Tensorable,
    *,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """Quantile function ("inverse CDF") of a normal distribution.

    Returns:
      quantile: Value at which CDF equals p.

    """
    p, loc, scale = convert_to_tensor(p, loc, scale)
    return ndtri(p) * scale + loc


def pdf_uniform(
    x: Tensorable,
    low: Tensorable = 0.0,
    high: Tensorable = 1.0,
) -> Tensor:
    """PDF of a uniform distribution.

    Returns:
      pdf: Probability density at x.

    """
    x, low, high = convert_to_tensor(x, low, high)
    span = high - low
    # Guard the degenerate interval (high == low) so the unused density branch
    # never produces inf; the support condition already excludes that case.
    safe_span = torch.where(span > 0, span, 1.0)
    return torch.where((x >= low) & (x < high), 1.0 / safe_span, 0.0)


def cdf_uniform(
    x: Tensorable,
    low: Tensorable = 0.0,
    high: Tensorable = 1.0,
) -> Tensor:
    """CDF of a uniform distribution.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, low, high = convert_to_tensor(x, low, high)
    span = high - low
    y = x.max(low).min(high)
    # Degenerate interval (high == low) is a point mass at ``low``: the CDF is
    # the unit step there. Guard the zero denominator to avoid 0/0 == nan.
    safe_span = torch.where(span > 0, span, 1.0)
    return torch.where(span > 0, (y - low) / safe_span, (x >= low).to(x.dtype))


def quantile_uniform(
    p: Tensorable,
    low: Tensorable = 0.0,
    high: Tensorable = 1.0,
) -> Tensor:
    """Quantile function of a uniform distribution.

    Returns:
      quantile: Value at which CDF equals p.

    """
    p, low, high = convert_to_tensor(p, low, high)
    if not torch.all((p >= 0) & (p <= 1)):
        raise ValueError("quantile_uniform requires p in [0, 1].")
    return p * (high - low) + low


def pdf_logit_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """PDF of a logit-normal distribution.

    Returns:
      pdf: Probability density at x.

    """
    x, loc, scale = convert_to_tensor(x, loc, scale)
    pdf = functools.partial(pdf_normal, loc=loc, scale=scale)
    return pdf_logit_distribution(x, pdf)


def cdf_logit_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """CDF of a logit-normal distribution.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, loc, scale = convert_to_tensor(x, loc, scale)
    cdf = functools.partial(cdf_normal, loc=loc, scale=scale)
    return cdf_logit_distribution(x, cdf)


def quantile_logit_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """Quantile function of a logit-normal distribution.

    Returns:
      quantile: Value at which CDF equals p.

    """
    x, loc, scale = convert_to_tensor(x, loc, scale)
    quantile = functools.partial(quantile_normal, loc=loc, scale=scale)
    return quantile_logit_distribution(x, quantile)


def cdf_truncated_normal(
    x: Tensorable,
    *,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    low: Tensorable = -math.inf,
    high: Tensorable = math.inf,
) -> Tensor:
    """CDF of a truncated normal distribution.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, loc, scale, low, high = convert_to_tensor(x, loc, scale, low, high)
    std_x = (x - loc) / scale
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    span = ndtr(std_high) - ndtr(std_low)
    # Degenerate interval (high == low) is a point mass at ``low``: the CDF is
    # the unit step there. Guard only the exactly-zero denominator (span != 0
    # also admits the deliberately reversed low > high direction) to avoid
    # 0/0 == nan, matching cdf_uniform's degenerate-interval convention.
    nondegenerate = span != 0
    safe_span = torch.where(nondegenerate, span, 1.0)
    return torch.where(
        nondegenerate,
        (ndtr(std_x) - ndtr(std_low)) / safe_span,
        (x >= low).to(x.dtype),
    )


def log_cdf_truncated_normal(
    x: Tensorable,
    *,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    low: Tensorable = -math.inf,
    high: Tensorable = math.inf,
) -> Tensor:
    """Log CDF of a truncated normal distribution.

    Args:
      x: Value at which to evaluate.
      loc: Location parameter.
      scale: Scale parameter.
      low: Lower truncation bound.
      high: Upper truncation bound.

    Returns:
      log_cdf: log P(X ≤ x | low ≤ X ≤ high).

    """
    x, loc, scale, low, high = convert_to_tensor(x, loc, scale, low, high)
    std_x = (x - loc) / scale
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    # Degenerate interval (high == low) is a point mass at ``low``: the log CDF
    # is the log of the unit step (-inf below, 0 at/above), not
    # logsubexp(-inf, -inf) == nan. Guard only exact equality so the
    # deliberately reversed low > high direction still flows through logsubexp
    # (which orders its operands internally). Mirror cdf_truncated_normal.
    nondegenerate = std_high != std_low
    safe_high = torch.where(nondegenerate, std_high, std_low + 1.0)
    log_cdf = logsubexp(
        torch.special.log_ndtr(std_x),
        torch.special.log_ndtr(std_low),
    ) - logsubexp(
        torch.special.log_ndtr(safe_high),
        torch.special.log_ndtr(std_low),
    )
    step = torch.where(x >= low, 0.0, -math.inf)
    return torch.where(nondegenerate, log_cdf, step)


def quantile_truncated_normal(
    p: Tensorable,
    *,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    low: Tensorable = -math.inf,
    high: Tensorable = math.inf,
) -> Tensor:
    """Quantile function ("inverse CDF") of a truncated normal distribution.

    Returns:
      quantile: Value at which CDF equals p.

    """
    p, loc, scale, low, high = convert_to_tensor(p, loc, scale, low, high)
    if not torch.all((p >= 0) & (p <= 1)):
        raise ValueError("quantile_truncated_normal requires p in [0, 1].")
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    y = ndtri(ndtr(std_low) + p * _normal_cdf_difference(std_high, std_low))
    return y * scale + loc


def pdf_logit_distribution(
    x: Tensorable,
    base_pdf: Callable[[Tensor], Tensor],
) -> Tensor:
    """Density of Y = sigmoid(X) at y, given density of X.

    Args:
      x: Evaluation point in (0, 1).
      base_pdf: Density of X.

    Returns:
      pdf: p_X(logit(y)) / (y(1-y)).

    Derivation:
      By change-of-variables: p_Y(y) = p_X(logit(y)) / |dy/dx|
      where dy/dx = y(1-y), so p_Y(y) = p_X(logit(y)) / (y(1-y)).

    """
    x = convert_to_tensor(x)
    # The support is (0, 1); logit(x) is NaN/+-inf outside it (and at the exact
    # boundary), which would crash a validating base_pdf. Evaluate on a masked
    # x pinned into the support, then zero the density off-support -- the
    # uniform-family off-support convention.
    inside = (x > 0) & (x < 1)
    z = torch.logit(torch.where(inside, x, 0.5))
    # |dy/dx|⁻¹ = 1/(y(1-y)) = exp(softplus(z) + softplus(-z))
    neg_log_jac = torch.nn.functional.softplus(z) + torch.nn.functional.softplus(-z)
    pdf = base_pdf(z) * torch.exp(neg_log_jac)
    return torch.where(inside, pdf, 0.0)


def cdf_logit_distribution(
    x: Tensorable,
    base_cdf: Callable[[Tensor], Tensor],
) -> Tensor:
    """CDF of Y = sigmoid(X), given CDF of X.

    Returns:
      cdf: base_cdf(logit(x)).

    """
    x = convert_to_tensor(x)
    # The support is (0, 1); clamp x off-support to 0 below / 1 above so the
    # CDF is the saturating step there, mirroring the uniform-family
    # convention, and feed base_cdf a finite logit it can validate.
    inside = (x > 0) & (x < 1)
    cdf = base_cdf(torch.logit(torch.where(inside, x, 0.5)))
    return torch.where(inside, cdf, (x >= 1).to(cdf.dtype))


def quantile_logit_distribution(
    x: Tensorable,
    base_quantile: Callable[[Tensor], Tensor],
) -> Tensor:
    """Quantile function of Y = sigmoid(X), given quantile of X.

    Returns:
      quantile: sigmoid(base_quantile(x)).

    """
    x = convert_to_tensor(x)
    return torch.sigmoid(base_quantile(x))


def random_student_t(
    *samples_size: int,
    df: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from Student's t-distribution. Non-differentiable wrt df.

    Returns:
      samples: Tensor of shape (*samples_size, *params_size).

    """
    # StudentT(df) = Normal(0,1) / sqrt(Chi2(df)/df), Chi2(df) = Gamma(df/2, 1/2).
    samples_size = _unpack_size(*samples_size)
    df, loc, scale = convert_to_tensor(df, loc, scale, dtype=dtype, device=device)
    params_size: tuple[int, ...] = torch.broadcast_shapes(
        df.shape,
        loc.shape,
        scale.shape,
    )
    x = torch.randn(*samples_size, *params_size, dtype=df.dtype, device=df.device)
    half_df = 0.5 * df
    z = random_gamma(
        *samples_size,
        concentration=torch.broadcast_to(half_df, params_size),
        rate=half_df,
    )
    return loc + scale * x * torch.rsqrt(z)


def random_chi2(
    *samples_size: int,
    df: Tensorable,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from chi-squared distribution. Non-differentiable wrt df.

    Returns:
      samples: Tensor of shape (*samples_size, *df.shape).

    """
    df_tensor = convert_to_tensor(df, dtype=dtype, device=device)
    return random_gamma(
        *samples_size,
        concentration=0.5 * df_tensor,
        rate=0.5,
    )


# Adapted from tensorflow/probability.
def random_gamma(
    *samples_size: int,
    concentration: Tensorable,
    rate: Tensorable = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from Gamma(concentration, rate). Non-differentiable wrt concentration.

    Returns:
      samples: Tensor of shape (*samples_size, *params_size).

    """
    samples_size = _unpack_size(*samples_size)
    concentration, rate = convert_to_tensor(
        concentration,
        rate,
        dtype=dtype,
        device=device,
    )
    params_size: tuple[int, ...] = torch.broadcast_shapes(
        concentration.shape,
        rate.shape,
    )
    concentration = torch.broadcast_to(concentration, samples_size + params_size)
    # detach: _standard_gamma's gradient is incorrect (not reparameterizable).
    y = torch._standard_gamma(concentration).detach() / rate  # noqa: SLF001
    y = y.clamp_(min=torch.finfo(y.dtype).tiny)
    return y


# Adapted from tensorflow/probability.
def random_categorical(
    *samples_size: int,
    probs: Tensorable | None = None,
    logits: Tensorable | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from a categorical distribution.

    Args:
      *samples_size: Output sample shape.
      probs: Unnormalized probabilities. Exactly one of probs/logits required.
      logits: Log-odds. Exactly one of probs/logits required.
      dtype: Output dtype.
      device: Output device.

    Returns:
      samples: Integer tensor of shape (*samples_size, *batch_shape).

    Raises:
      ValueError: If both or neither of probs and logits are provided.

    """
    samples_size = _unpack_size(*samples_size)
    if (probs is None) == (logits is None):
        raise ValueError("Specify exactly one of probs or logits.")
    if probs is None:
        logits = convert_to_tensor(logits, dtype=dtype, device=device)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        cmf = torch.logcumsumexp(logits, dim=-1)
        if not torch.all(torch.isfinite(cmf[..., -1:])):
            raise ValueError(
                "random_categorical requires at least one finite logit per row.",
            )
        cmf = torch.exp(cmf - cmf[..., -1:])
    else:
        probs = convert_to_tensor(probs, dtype=dtype, device=device)
        cmf = torch.cumsum(probs, dim=-1)
        cmf = cmf / cmf[..., -1:]
    z = torch.rand(*samples_size, *[1] * cmf.ndim, dtype=cmf.dtype, device=cmf.device)
    return (z >= cmf).sum(dim=-1)


def random_logit_normal(
    *samples_size: int,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from a logit-normal distribution.

    Returns:
      samples: Tensor in (0, 1) of shape (*samples_size, *params_size).

    """
    samples_size = _unpack_size(*samples_size)
    loc, scale = convert_to_tensor(
        loc,
        scale,
        dtype_hint=torch.float32,
        dtype=dtype,
        device=device,
    )
    params_size: tuple[int, ...] = torch.broadcast_shapes(loc.shape, scale.shape)
    scale = torch.broadcast_to(scale, samples_size + tuple(params_size))
    z = torch.randn(*scale.shape, dtype=scale.dtype, device=scale.device) * scale + loc
    return torch.sigmoid(z)


def ndtr(x: Tensorable) -> Tensor:
    """Normal distribution function.

    Adapted from ``tfp.internal.special_math.ndtr`` (tensorflow/probability).

    Returns:
      ndtr: Φ(x) = 0.5 (1 + erf(x / √2)).

    Note: torch.special.ndtr exists but loses precision at extreme
    values. This piecewise erf/erfc formulation is more accurate
    in the tails.

    References:
      tfp.math.ndtr

    """
    t = convert_to_tensor(x)
    t = t * 0.5**0.5
    z = t.abs()
    return 0.5 * torch.where(
        z < 0.5**0.5,
        1 + torch.erf(t),
        torch.where(t > 0.0, 2 - torch.erfc(z), torch.erfc(z)),
    )


def ndtri(p: Tensorable) -> Tensor:
    """Function inverse of ndtr.

    Returns:
      x: Value such that ndtr(x) = p.

    Note: torch.special.ndtri exists but is less accurate in the
    tails than erfinv(2p - 1) * √2.

    References:
      tfp.math.ndtri

    """
    p = convert_to_tensor(p)
    return torch.erfinv(2 * p - 1) * 2**0.5


def _normal_cdf_difference(a: Tensorable, b: Tensorable) -> Tensor:
    """Computes ndtr(a) - ndtr(b) assuming a >= b.

    Adapted from ``tfp.distributions.truncated_normal._normal_cdf_difference``.

    When both a, b > 0, ndtr values are near 1 so subtraction suffers
    cancellation. Using ndtr(-z) = 1 - ndtr(z), rewrite as
    ndtr(-b) - ndtr(-a), where both values are near 0.
    """
    a, b = convert_to_tensor(a, b)
    flip = b >= 0
    hi = torch.where(flip, -b, a)
    lo = torch.where(flip, -a, b)
    return ndtr(hi) - ndtr(lo)


def log_gamma_correction(x: Tensorable) -> Tensor:
    """Error of the Stirling approximation to lgamma(x) for x >= 8.

    lgamma(x) ≈ (x-0.5)*log(x) - x + 0.5*log(2π) + log_gamma_correction(x).

    Uses a rational minimax approximation (DiDonato & Morris 1988).

    References:
      DiDonato & Morris, "Significant Digit Computation of the
      Incomplete Beta Function Ratios", 1988. NSWC TR 88-365, Eq (32).
      tfp.math.log_gamma_correction

    """
    x = convert_to_tensor(x)
    # Minimax polynomial coefficients from DiDonato & Morris.
    c = torch.tensor(
        [
            0.833333333333333e-01,
            -0.277777777760991e-02,
            0.793650666825390e-03,
            -0.595202931351870e-03,
            0.837308034031215e-03,
            -0.165322962780713e-02,
        ],
        dtype=x.dtype,
        device=x.device,
    )
    inv_x = x.reciprocal()
    inv_x2 = inv_x * inv_x
    # Horner evaluation.
    acc = c[5]
    for i in range(4, -1, -1):
        acc = acc * inv_x2 + c[i]
    return acc * inv_x


def log_gamma_difference(x: Tensorable, y: Tensorable) -> Tensor:
    """lgamma(y) - lgamma(x + y), accurate for large y.

    For y >= 8, cancels Stirling terms analytically, leaving only
    the small correction terms.

    References:
      DiDonato & Morris, "Significant Digit Computation of the
      Incomplete Beta Function Ratios", 1988. NSWC TR 88-365.
      tfp.math.log_gamma_difference

    """
    x, y = convert_to_tensor(x, y)
    naive = torch.lgamma(y) - torch.lgamma(x + y)
    cancelled_stirling = -(x + y - 0.5) * torch.log1p(x / y) - x * y.log() + x
    correction = log_gamma_correction(y) - log_gamma_correction(x + y)
    return torch.where(y >= 8, cancelled_stirling + correction, naive)


def lbeta(x: Tensorable, y: Tensorable) -> Tensor:
    """Log Beta(x, y), accurate for large arguments.

    Naive lgamma(x) + lgamma(y) - lgamma(x+y) suffers catastrophic
    cancellation when x, y are large. This uses Stirling decomposition
    to cancel the large terms analytically.

    References:
      DiDonato & Morris, "Significant Digit Computation of the
      Incomplete Beta Function Ratios", 1988. NSWC TR 88-365.
      tfp.math.lbeta

    """
    x, y = convert_to_tensor(x, y)
    x, y = torch.minimum(x, y), torch.maximum(x, y)
    log2pi = math.log(2 * math.pi)
    # Double-where: the two_large branch is only selected for x >= 8, but
    # torch.where evaluates it everywhere. At x == 0, ``(x / (x + y)).log()``
    # is log(0) = -inf and leaks a NaN gradient into the selected branch, so
    # feed this branch an x pinned into its own (x >= 8) domain.
    safe_x = torch.where(x >= 8, x, 8.0)
    two_large = (
        0.5 * log2pi
        - 0.5 * y.log()
        + log_gamma_correction(safe_x)
        + log_gamma_correction(y)
        - log_gamma_correction(safe_x + y)
        + (safe_x - 0.5) * (safe_x / (safe_x + y)).log()
        - y * torch.log1p(safe_x / y)
    )
    # One large (x < 8, y >= 8).
    one_large = torch.lgamma(x) + log_gamma_difference(x, y)
    # Both small.
    small = torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
    return torch.where(
        x >= 8,
        two_large,
        torch.where(y >= 8, one_large, small),
    )


def _unpack_size(*samples_size: int) -> tuple[int, ...]:
    if len(samples_size) == 1 and isinstance(samples_size[0], Sequence):
        return tuple(cast(Sequence[int], samples_size[0]))
    return samples_size
