"""Probability and distribution utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools
import math

from torch import Tensor, distributions, nn

import torch

from priml.math.numeric import log1mexp, logsubexp
from priml.memory import convert_to_tensor


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.math.custom_types import Tensorable, TensorableFn


# Adapted from tensorflow/probability.
def pdf_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """PDF of a normal distribution.

    Args:
      x: Sample value at which to evaluate the density.
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.

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

    Args:
      x: Sample value at which to evaluate the CDF.
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.

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

    Args:
      p: Cumulative probability in [0, 1].
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.

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

    Args:
      x: Sample value at which to evaluate the density.
      low: Lower bound of the support.
      high: Upper bound of the support.

    Returns:
      pdf: Probability density at x.

    """
    x, low, high = convert_to_tensor(x, low, high)
    span = high - low
    # Guard the degenerate interval (high == low) so the unused density branch
    # never produces inf; the support condition already excludes that case.
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    return torch.where((x >= low) & (x < high), 1.0 / safe_span, 0.0)


def cdf_uniform(
    x: Tensorable,
    low: Tensorable = 0.0,
    high: Tensorable = 1.0,
) -> Tensor:
    """CDF of a uniform distribution.

    Args:
      x: Sample value at which to evaluate the CDF.
      low: Lower bound of the support.
      high: Upper bound of the support.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, low, high = convert_to_tensor(x, low, high)
    span = high - low
    y = x.max(low).min(high)
    # Degenerate interval (high == low) is a point mass at ``low``: the CDF is
    # the unit step there. Guard the zero denominator to avoid 0/0 == nan.
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    return torch.where(span > 0, (y - low) / safe_span, x >= low)


def quantile_uniform(
    p: Tensorable,
    low: Tensorable = 0.0,
    high: Tensorable = 1.0,
) -> Tensor:
    """Quantile function of a uniform distribution.

    Args:
      p: Cumulative probability in [0, 1].
      low: Lower bound of the support.
      high: Upper bound of the support.

    Returns:
      quantile: Value at which CDF equals p.

    """
    p, low, high = convert_to_tensor(p, low, high)
    return torch.where((p >= 0) & (p <= 1), p * (high - low) + low, math.nan)


def log_prob_discretized_logistic(
    x: Tensorable,
    loc: Tensorable,
    log_scale: Tensorable,
    *,
    num_bins: int = 256,
) -> Tensor:
    """Log mass of integer pixels under a logistic discretized onto ``[-1, 1]``.

    Value ``v`` in ``0..num_bins-1`` maps to the bin centred at
    ``2 v / (num_bins - 1) - 1`` with half-width ``1 / (num_bins - 1)``; the two
    end bins absorb the tails, so the masses sum to one.

    Args:
      x: Integer values in ``[0, num_bins)``, any integer or float dtype.
      loc: Logistic location, in ``[-1, 1]`` units.
      log_scale: Logistic log scale.
      num_bins: Number of values; 256 for 8-bit pixels.

    Returns:
      log_prob: Elementwise log mass, broadcast over the arguments.

    References:
      https://arxiv.org/abs/1701.05517
        Salimans et al. 2017, "PixelCNN++."

    Derivation:
      For an interior bin with standardized edges ``u > l``, the mass is
      ``σ(u) - σ(l)``. On whichever side of the location keeps both CDFs at
      most one half (reflect ``(u, l) -> (-l, -u)`` above it),

          log(σ(a) - σ(b)) = log σ(a) + log1mexp(log σ(b) - log σ(a)),

      so the difference never cancels between two numbers near one.

    """
    x, loc, log_scale = convert_to_tensor(x, loc, log_scale)
    half = 1 / (num_bins - 1)
    centre = x * (2 * half) - 1
    inverse = torch.exp(-log_scale)
    upper = (centre - loc + half) * inverse
    lower = (centre - loc - half) * inverse
    below_location = torch.signbit(upper + lower)
    a = torch.where(below_location, upper, -lower)
    b = torch.where(below_location, lower, -upper)
    log_a = nn.functional.logsigmoid(a)
    # Coincident rounded edges need a representable negative log gap.
    gap = (nn.functional.logsigmoid(b) - log_a).clamp_max(
        -torch.finfo(log_a.dtype).tiny,
    )
    return torch.where(
        x <= 0,
        nn.functional.logsigmoid(upper),
        torch.where(
            x >= num_bins - 1,
            nn.functional.logsigmoid(-lower),
            log_a + log1mexp(gap),
        ),
    )


def pdf_logit_normal(
    x: Tensorable,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
) -> Tensor:
    """PDF of a logit-normal distribution.

    Args:
      x: Sample value in (0, 1) at which to evaluate the density.
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.

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

    Args:
      x: Sample value in (0, 1) at which to evaluate the CDF.
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.

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

    Args:
      x: Cumulative probability in (0, 1).
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.

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

    Args:
      x: Sample value at which to evaluate the CDF.
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.
      low: Lower truncation bound.
      high: Upper truncation bound.

    Returns:
      cdf: Cumulative probability at x.

    """
    x, loc, scale, low, high = convert_to_tensor(x, loc, scale, low, high)
    bounded_x = x.clamp(min=torch.minimum(low, high), max=torch.maximum(low, high))
    std_x = (bounded_x - loc) / scale
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    span = ndtr(std_high) - ndtr(std_low)
    # Degenerate interval (high == low) is a point mass at ``low``: the CDF is
    # the unit step there. Guard only the exactly-zero denominator (span != 0
    # also admits the deliberately reversed low > high direction) to avoid
    # 0/0 == nan, matching cdf_uniform's degenerate-interval convention.
    nondegenerate = span != 0
    safe_span = torch.where(nondegenerate, span, torch.ones_like(span))
    return torch.where(
        nondegenerate,
        (ndtr(std_x) - ndtr(std_low)) / safe_span,
        x >= low,
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
    interior = (x > torch.minimum(low, high)) & (x < torch.maximum(low, high))
    # Mask all operands, not only the result: logsubexp(a, a) has an
    # infinite derivative that would leak through the unselected branch.
    std_x = torch.where(interior, (x - loc) / scale, 0.5)
    std_low = torch.where(interior, (low - loc) / scale, 0.0)
    std_high = torch.where(interior, (high - loc) / scale, 1.0)
    log_cdf = logsubexp(
        torch.special.log_ndtr(std_x),
        torch.special.log_ndtr(std_low),
    ) - logsubexp(
        torch.special.log_ndtr(std_high),
        torch.special.log_ndtr(std_low),
    )
    at_one = torch.where(high < low, x <= high, x >= high)
    step = torch.where(at_one, 0.0, -math.inf)
    return torch.where(interior, log_cdf, step)


def quantile_truncated_normal(
    p: Tensorable,
    *,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    low: Tensorable = -math.inf,
    high: Tensorable = math.inf,
) -> Tensor:
    """Quantile function ("inverse CDF") of a truncated normal distribution.

    Adapted from tensorflow_probability:
      tensorflow_probability/python/distributions/truncated_normal.py::_quantile

    Args:
      p: Cumulative probability in [0, 1].
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.
      low: Lower truncation bound.
      high: Upper truncation bound.

    Returns:
      quantile: Value at which CDF equals p.

    """
    p, loc, scale, low, high = convert_to_tensor(p, loc, scale, low, high)
    in_domain = (p >= 0) & (p <= 1)
    safe_p = torch.where(in_domain, p, 0.5)
    std_low = (low - loc) / scale
    std_high = (high - loc) / scale
    y = ndtri(ndtr(std_low) + safe_p * _normal_cdf_difference(std_high, std_low))
    return torch.where(in_domain, y * scale + loc, math.nan)


def pdf_logit_distribution(
    x: Tensorable,
    base_pdf: TensorableFn,
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
    base_cdf: TensorableFn,
) -> Tensor:
    """CDF of Y = sigmoid(X), given CDF of X.

    Args:
      x: Sample value in (0, 1) at which to evaluate the CDF.
      base_cdf: CDF of the underlying distribution.

    Returns:
      cdf: base_cdf(logit(x)).

    """
    x = convert_to_tensor(x)
    # The support is (0, 1); clamp x off-support to 0 below / 1 above so the
    # CDF is the saturating step there, mirroring the uniform-family
    # convention, and feed base_cdf a finite logit it can validate.
    inside = (x > 0) & (x < 1)
    cdf = base_cdf(torch.logit(torch.where(inside, x, 0.5)))
    return torch.where(inside, cdf, x >= 1)


def quantile_logit_distribution(
    x: Tensorable,
    base_quantile: TensorableFn,
) -> Tensor:
    """Quantile function of Y = sigmoid(X), given quantile of X.

    Args:
      x: Cumulative probability in (0, 1).
      base_quantile: Quantile function of the underlying distribution.

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

    Adapted from tensorflow_probability:
      tensorflow_probability/python/distributions/student_t.py::sample_n

    Args:
      df: Degrees of freedom (shape parameter).
      loc: Location shift.
      scale: Scale factor.
      dtype: Output tensor data type.
      device: Output tensor device.
      *samples_size: Shape of the sample batch.

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

    Args:
      df: Degrees of freedom (shape parameter).
      dtype: Output tensor data type.
      device: Output tensor device.
      *samples_size: Shape of the sample batch.

    Returns:
      samples: Tensor of shape (*samples_size, *df.shape).

    """
    df_tensor = convert_to_tensor(df, dtype=dtype, device=device)
    return random_gamma(
        *samples_size,
        concentration=0.5 * df_tensor,
        rate=0.5,
    )


# Adapted from tensorflow_probability:
#   tensorflow_probability/python/distributions/gamma.py::random_gamma.
def random_gamma(
    *samples_size: int,
    concentration: Tensorable,
    rate: Tensorable = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from Gamma(concentration, rate). Non-differentiable wrt concentration.

    Args:
      concentration: Shape parameter α > 0.
      rate: Rate parameter β > 0 (inverse scale).
      dtype: Output tensor data type.
      device: Output tensor device.
      *samples_size: Shape of the sample batch.

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
    # Detach: _standard_gamma's gradient is incorrect (not reparameterizable).
    y = torch._standard_gamma(concentration).detach() / rate  # noqa: SLF001 -- The probability helper tests the distribution's private numerical seam.
    return y.clamp_(min=torch.finfo(y.dtype).tiny)


# Adapted from tensorflow_probability:
#   tensorflow_probability/python/distributions/categorical.py::_sample_n.
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
      dtype: Floating-point dtype used for probabilities and random draws.
      device: Output device.

    Returns:
      samples: Int64 tensor of shape (*samples_size, *batch_shape).

    Raises:
      ValueError: If both or neither of probs and logits are provided.

    """
    samples_size = _unpack_size(*samples_size)
    # Each arm re-tests the other parameter rather than sharing one
    # exclusive-or guard: the xor proves the invariant at runtime but leaves
    # both names optional to the checker, so every use below needs the
    # membership test that narrows it.
    if probs is None:
        if logits is None:
            raise ValueError("Specify exactly one of probs or logits.")
        logits = convert_to_tensor(logits, dtype=dtype, device=device)
        cmf = torch.logcumsumexp(logits, dim=-1)
        cmf = torch.exp(cmf - cmf[..., -1:])
    else:
        if logits is not None:
            raise ValueError("Specify exactly one of probs or logits.")
        probs = convert_to_tensor(probs, dtype=dtype, device=device)
        cmf = torch.cumsum(probs, dim=-1)
        cmf = cmf / cmf[..., -1:]
    z = torch.rand(
        *samples_size,
        *cmf.shape[:-1],
        1,
        dtype=cmf.dtype,
        device=cmf.device,
    )
    return (z >= cmf).sum(dim=-1)


def random_logit_normal(
    *samples_size: int,
    loc: Tensorable = 0.0,
    scale: Tensorable = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from a logit-normal distribution.

    Args:
      loc: Mean of the underlying normal distribution.
      scale: Standard deviation of the underlying normal distribution.
      dtype: Output tensor data type.
      device: Output tensor device.
      *samples_size: Shape of the sample batch.

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


def random_gumbel(
    *samples_size: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Sample the standard Gumbel distribution as ``-log(-log U)``, ``U ~ U[0, 1)``.

    Args:
      *samples_size: Shape of the sample batch.
      generator: Source of the uniform draws; None is torch's default.
      dtype: Output tensor data type.
      device: Output tensor device.

    Returns:
      samples: Tensor of shape ``samples_size``; ``-inf`` wherever ``U`` is 0.

    """
    # No clamp: a draw of 0 gives -inf, which loses every argmax and keeps a
    # masked -inf logit at -inf, never NaN. Clamping U would move the bits of
    # every sample a Gumbel-max caller has pinned.
    uniform = torch.rand(
        _unpack_size(*samples_size),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return -(-uniform.log()).log()


def gumbel_max(logits: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
    """Sample one index per row of ``softmax(logits)`` by the Gumbel-max trick.

    Args:
      logits: Scores ``[..., V]``; ``-inf`` entries are never chosen.
      generator: Source of the noise, on ``logits``' device; None is torch's
        default.

    Returns:
      index: Sampled index into the last axis, long ``[...]``.

    References:
      https://arxiv.org/abs/1411.0030
        Maddison, Tarlow and Minka 2014, "A* Sampling."

    """
    # Float32 noise whatever the logits' dtype: a bfloat16 U stops at 1 - 2**-8,
    # which caps the noise at 5.5 (16.6 in float32) and biases the samples. Not
    # the trick's exponential spelling, ``argmax(logits - log E)``: E can be
    # exactly 0, and -inf - log(0) is NaN, which ``argmax`` picks.
    noise = random_gumbel(
        *logits.shape,
        generator=generator,
        dtype=torch.float32,
        device=logits.device,
    )
    return (logits + noise).argmax(-1)


def ndtr(x: Tensorable) -> Tensor:
    """Evaluate the normal distribution function.

    Adapted from tensorflow_probability, which reformulates the piecewise
    erf/erfc split for tail accuracy:
      tensorflow_probability/python/internal/special_math.py::ndtr

    Args:
      x: Sample value at which to evaluate the standard normal CDF.

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
        torch.where(torch.signbit(t), torch.erfc(z), 2 - torch.erfc(z)),
    )


def ndtri(p: Tensorable) -> Tensor:
    """Evaluate the function inverse of ndtr.

    Args:
      p: Cumulative probability in (0, 1).

    Returns:
      x: Value such that ndtr(x) = p.

    Uses the tail-aware inverse directly: forming ``2p - 1`` rounds to -1
    for small float32 probabilities and loses the finite left tail.

    References:
      tfp.math.ndtri

    """
    p = convert_to_tensor(p)
    return torch.special.ndtri(p)


def log_gamma_correction(x: Tensorable) -> Tensor:
    """Error of the Stirling approximation to lgamma(x) for x >= 8.

    lgamma(x) ≈ (x-0.5)*log(x) - x + 0.5*log(2π) + log_gamma_correction(x).

    Uses a rational minimax approximation (DiDonato & Morris 1988).

    Args:
      x: Input value at least 8.

    Returns:
      correction: Stirling approximation error for lgamma(x).

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

    Args:
      x: First argument to the difference.
      y: Second argument; must be >= 8 for the analytical cancellation.

    Returns:
      result: Log-space difference lgamma(y) - lgamma(x + y).

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

    Args:
      x: First argument.
      y: Second argument.

    Returns:
      log_beta: log Beta(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y).

    References:
      DiDonato & Morris, "Significant Digit Computation of the
      Incomplete Beta Function Ratios", 1988. NSWC TR 88-365.
      tfp.math.lbeta

    """
    x, y = convert_to_tensor(x, y)
    x, y = torch.minimum(x, y), torch.maximum(x, y)
    log2pi = math.log(2 * math.pi)
    # Two-large is evaluated for every input. Clamp its logarithm into its
    # domain so the unselected branch cannot leak a NaN gradient at x == 0.
    safe_x = x.clamp_min(8.0)
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


def _unpack_size(*samples_size: int | Sequence[int]) -> tuple[int, ...]:
    if len(samples_size) == 1:
        size = samples_size[0]
        if isinstance(size, int):
            return (size,)
        return tuple(int(dimension) for dimension in size)

    dimensions: list[int] = []
    for size in samples_size:
        if not isinstance(size, int):
            raise TypeError("sample dimensions must be integers")
        dimensions.append(size)
    return tuple(dimensions)


# Adapted from tensorflow_probability: tensorflow_probability/python/distributions/trunc
# ated_normal.py::_normal_cdf_difference
#
# When both a, b > 0, ndtr values are near 1 so subtraction suffers cancellation. Using
# ndtr(-z) = 1 - ndtr(z), rewrite as ndtr(-b) - ndtr(-a), where both values are near 0.
def _normal_cdf_difference(a: Tensorable, b: Tensorable) -> Tensor:
    """Compute ndtr(a) - ndtr(b) assuming a >= b."""
    a, b = convert_to_tensor(a, b)
    flip = b >= 0
    hi = torch.where(flip, -b, a)
    lo = torch.where(flip, -a, b)
    return ndtr(hi) - ndtr(lo)
