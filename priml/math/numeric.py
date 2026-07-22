from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import math

from torch import Tensor

import torch

from priml.math.basic import broadcast_sequences
from priml.math.custom_types import Tensorable, convert_to_tensor


def log_arctan_exp(x: Tensorable) -> Tensor:
    """Numerically stable log(arctan(exp(x))).

    Returns:
      result: log(arctan(exp(x))).

    Derivation:
      Uses two branches, both only computing exp of non-positive
      arguments:

        x >= 0: log(pi/2) + log1p(-2/pi * arctan(exp(-x)))
        x <  0: x + log(arctan(z) / z),  z = exp(-|x|)

    """
    x = convert_to_tensor(x)
    z = torch.exp(-x.abs()).clamp(min=torch.finfo(x.dtype).tiny)
    atan_z = torch.arctan(z)
    return torch.where(
        x < 0,
        x + torch.log(atan_z / z),
        math.log(math.pi / 2) + torch.log1p((-2 / math.pi) * atan_z),
    )


def log_tan_exp(x: Tensorable) -> Tensor:
    """Numerically stable log(tan(exp(x))) for x < log(pi/2).

    Returns:
      result: log(tan(exp(x))).

    Derivation:
      Since tan(z)/z >= 1 for z in (0, pi/2), this is simply
      x + log(tan(z)/z) — a non-negative correction to x with
      no cancellation.

    """
    x = convert_to_tensor(x)
    z = torch.exp(x.clamp(min=math.log(torch.finfo(x.dtype).tiny)))
    return x + torch.log(torch.tan(z) / z)


def log1mexp(x: Tensorable) -> Tensor:
    """log(1 - exp(x)) for x < 0.

    Assumes x < 0 (does not take the absolute value).

    Adapted from ``tfp.math.log1mexp`` (tensorflow/probability).

    References:
      Machler (2012), "Accurately computing log(1 - exp(-|a|))"
      https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    """
    x = convert_to_tensor(x)
    return torch.where(
        x > -math.log(2),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


@overload
def logsubexp(
    x: Tensorable,
    y: Tensorable,
    *,
    return_sign: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def logsubexp(
    x: Tensorable,
    y: Tensorable,
    *,
    return_sign: Literal[False] = ...,
) -> Tensor: ...
def logsubexp(
    x: Tensorable,
    y: Tensorable,
    *,
    return_sign: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Numerically stable ``log(exp(max(x,y)) - exp(min(x,y)))``.

    Adapted from ``tfp.math.log_sub_exp`` (tensorflow/probability).

    Args:
      x: Float tensor broadcastable with *y*.
      y: Float tensor broadcastable with *x*.
      return_sign: If True, also return a +/-1 tensor indicating
        ``sign(exp(x) - exp(y))``. Required when x < y is possible,
        since log-space cannot represent negatives.

    Returns:
      result: The log absolute difference.
      sign: (only if *return_sign*) +1 where x >= y, -1 otherwise.

    """
    x, y = convert_to_tensor(x, y)
    hi = torch.maximum(x, y)
    lo = torch.minimum(x, y)
    # IEEE 754 indicates lo-hi<=0.
    result = hi + log1mexp(lo - hi)
    if return_sign:
        return result, torch.where(x < y, -1, 1)
    return result


def softcap(x: Tensorable, cap: Tensorable = 1.0) -> Tensor:
    """Soft capping: tanh(x/cap) * cap.

    Returns:
      result: Capped value in [-cap, cap].

    References:
      https://arxiv.org/abs/1611.09940
        De Brébisson & Vincent 2016, "An Exploration of Softmax Alternatives."
      https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf
        Gemma Team 2024, "Gemma 2: Improving Open Language Models."

    """
    orig_dtype = torch.as_tensor(x).dtype
    x, cap = convert_to_tensor(x, cap, dtype=torch.float32)
    return (torch.tanh(x / cap) * cap).to(orig_dtype)


def logerfc(x: Tensorable) -> Tensor:
    """log(erfc(x)), numerically stable for large positive x.

    For x >= 0, erfc(x) underflows; we use erfcx to avoid this:
      log(erfc(x)) = log(erfcx(x)) - x².

    References:
      tfp.math.logerfc

    """
    x = convert_to_tensor(x)
    negative = x < 0
    # Double-where: each branch must see an operand inside its own safe domain
    # so the unselected branch cannot inject a NaN gradient. The erfcx branch
    # underflows for x < 0, and erfc(x) underflows for x >= 0, so feed each a
    # masked input pinned to its safe side.
    safe_neg = torch.where(negative, x, -1.0)
    safe_pos = torch.where(negative, 1.0, x)
    return torch.where(
        negative,
        torch.log(torch.erfc(safe_neg)),
        torch.special.erfcx(safe_pos).log() - safe_pos * safe_pos,
    )


def log_cosh(x: Tensorable) -> Tensor:
    """Numerically stable ``log(cosh(x))``.

    Overflow-free for large ``|x|`` and exact at ``x == 0``; the gradient is
    ``tanh(x)``, finite for every finite real ``x``. Returns NaN only when
    ``x`` is NaN, and ``+inf`` for ``x = +-inf``.

    Derivation::

        cosh(x) = (e^x + e^-x) / 2 = e^|x| (1 + e^-2|x|) / 2
        log cosh(x) = |x| + log(1 + e^-2|x|) - log 2
                    = |x| + softplus(-2|x|) - log 2

    Note::

        d/dx log_cosh(x) = tanh(x) = 2 sigmoid(2 x) - 1
        integral sigmoid(x) dx = log(1 + e^x) = softplus(x)

    ``softplus`` is stable for every ``|x|``, so the single closed form needs
    no Taylor branch, no crossover constant, and no ``torch.where`` (whose
    unselected branch would otherwise inject NaN gradients).

    Args:
      x: Real input, any shape and float dtype.

    Returns:
      result: ``log(cosh(x))``, broadcast to the shape of ``x``.

    """
    ax = convert_to_tensor(x).abs()
    return ax + torch.nn.functional.softplus(-2 * ax) - math.log(2)


def softplus_inverse(x: Tensorable) -> Tensor:
    """Inverse of softplus: y such that softplus(y) = x.

    Uses log(expm1(x)) = x + log(-expm1(-x)) for stability,
    with asymptotic branches for extreme values.

    Args:
      x: Input tensor. The domain is x > 0 (softplus has range (0, inf));
        x <= 0 returns log(x), i.e. -inf at 0 and NaN below it.

    References:
      tfp.math.softplus_inverse

    """
    x = convert_to_tensor(x)
    threshold = math.log(torch.finfo(x.dtype).eps) + 2
    is_small = x < math.exp(threshold)
    is_large = x > -threshold
    safe = torch.where(is_small | is_large, 1.0, x)
    y = safe + torch.log(-torch.expm1(-safe))
    return torch.where(is_small, x.log(), torch.where(is_large, x, y))


def log1psquare(x: Tensorable) -> Tensor:
    """log(1 + x²), numerically stable for large |x|.

    For large |x| (where x² overflows), uses 2*log(|x|).

    References:
      tfp.math.log1psquare

    """
    x = convert_to_tensor(x)
    threshold = torch.finfo(x.dtype).eps ** -0.5
    is_large = x.abs() > threshold
    return torch.where(
        is_large,
        2 * torch.where(is_large, x.abs(), 1.0).log(),
        torch.log1p(x * x),
    )


def logmeanexp(
    x: Tensorable,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """Log-mean-exp: logsumexp(x, dim) - log(size(dim)).

    Computes log(mean(exp(x))) in a numerically stable way.

    Args:
      x: Input tensor.
      dim: Dimension(s) to reduce.
      keepdim: Whether to keep the reduced dimension.

    Returns:
      result: log(mean(exp(x), dim)).

    """
    x = convert_to_tensor(x)
    dim_ = (dim,) if isinstance(dim, int) else tuple(dim)
    n = math.prod(x.shape[d] for d in dim_)
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(n)


def mesh_arange(
    end: int | Sequence[int],
    *,
    start: int | Sequence[int] = 0,
    step: int | Sequence[int] = 1,
    dtype: torch.dtype = torch.long,
    device: torch.device | str | None = None,
) -> Tensor:
    """N-D meshgrid of arange's, flattened to ``[prod(sizes), N]``.

    Args:
      end: Upper bounds for each dimension (exclusive).
      start: Lower bounds (default 0).
      step: Step sizes (default 1).
      dtype: Output dtype.
      device: Output device.

    Returns:
      result: Tensor of shape [prod(sizes), N] with all grid coordinates.

    """
    start_, end_, step_ = broadcast_sequences(start, end, step)
    grid = torch.meshgrid(
        *[
            torch.arange(s, e, st, dtype=dtype, device=device)
            for s, e, st in zip(start_, end_, step_, strict=False)
        ],
        indexing="ij",
    )
    return torch.stack(grid, dim=-1).reshape(-1, len(end_))


def matrix_signum_via_newtonschulz(
    x: Tensor,
    coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Newton-Schulz iteration for the matrix sign function (orthogonal factor).

    Computes an approximation to U V^T from the SVD X = U Σ V^T using
    a quintic polynomial iteration. The coefficients are chosen to
    maximize convergence slope at zero.

    Expects 3D input [batch, rows, cols]. If rows > cols, the matrix
    is transposed internally for efficiency.

    Args:
      x: Input tensor of shape [batch, rows, cols].
      coefficients: Quintic iteration coefficients (a, b, c).
      steps: Number of Newton-Schulz iterations.
      eps: Norm clamping epsilon.

    Returns:
      result: Approximate orthogonal factor, same shape as input.

    References:
      Bernstein & Wan 2024, "Old Optimizer, New Norm."
      Keller Jordan, Muon optimizer.
      https://docs.modula.systems/algorithms/newton-schulz/

    """
    a, b, c = coefficients
    orig_dtype = x.dtype
    x = x.bfloat16()
    x = x / x.norm(dim=[-2, -1], keepdim=True).clamp(min=eps)
    if transpose := x.shape[-2] > x.shape[-1]:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        correction = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, correction, x, beta=a, alpha=1)
    if transpose:
        x = x.transpose(-1, -2)
    return x.to(orig_dtype)


def log_cumsum_exp(x: Tensorable, dim: int = -1) -> Tensor:
    """Numerically stable cumulative logsumexp along a dimension.

    Computes log(cumsum(exp(x))) without overflow/underflow by
    subtracting a running maximum.

    Args:
      x: Input tensor.
      dim: Dimension along which to accumulate.

    Returns:
      result: log(cumsum(exp(x), dim)).

    References:
      tfp.math.log_cumsum_exp

    """
    x = convert_to_tensor(x)
    return torch.logcumsumexp(x, dim=dim)


def kahan_sum(x: Tensorable, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """Compensated summation (Kahan–Babuška–Neumaier) for FP accuracy.

    Reduces rounding error when summing many terms; useful for gradient
    accumulation and loss aggregation over large batches.

    Uses the Neumaier improvement over plain Kahan summation. Kahan (1965)
    carries the rounding error of ``running_total + term`` in a single
    compensation accumulator; Babuška (1969) studied the same scheme; both lose
    precision when an individual ``term`` is larger in magnitude than the
    running total (the lost low-order bits are then those of ``running_total``,
    not of ``term``). Neumaier (1974) fixes this by selecting which operand's
    low bits to recover per term, via the ``|total| >= |term|`` branch below.
    The classic example ``[1, 1e100, 1, -1e100]`` (true sum 2) returns 0 under
    plain Kahan but 2 under Neumaier.

    Neumaier (a.k.a. Kahan–Babuška–Neumaier, KBN) is chosen because it is
    strictly more accurate than plain Kahan at no extra per-term cost. A
    second-order variant (Klein 2006) carries a compensation-of-the-
    compensation and is more accurate still in adversarial cases, but it adds a
    second accumulator and arithmetic per term, so it is intentionally NOT used
    here; switch to it only if profiling shows KBN's residual error matters.

    Args:
      x: Input tensor.
      dim: Dimension to reduce. None reduces all elements.
      keepdim: Whether to keep the reduced dimension.

    Returns:
      result: Compensated sum.

    References:
      Kahan 1965, "Pracniques: further remarks on reducing truncation errors."
      Babuška 1969, "Numerical stability in mathematical analysis."
      Neumaier 1974, "Rundungsfehleranalyse einiger Verfahren zur Summation
      endlicher Summen." (the KBN variant implemented here).
      Klein 2006, "A generalized Kahan–Babuška-Summation-Algorithm."
      (second-order; more accurate but costlier — not used.)

    """
    x = convert_to_tensor(x)
    reduce_all = dim is None
    orig_ndim = x.ndim
    if reduce_all:
        x = x.reshape(-1)
        dim = 0
    # Move target dim to front for iteration.
    x = x.moveaxis(dim, 0)
    # An empty reduction sums to zero; ``x[0]`` would otherwise IndexError.
    template = x[0] if x.shape[0] > 0 else x.new_zeros(x.shape[1:])
    total = torch.zeros_like(template)
    compensation = torch.zeros_like(template)
    for i in range(x.shape[0]):
        term = x[i]
        t = total + term
        # Recover the low-order bits of whichever operand is larger: when the
        # running total dominates, ``(total - t) + term`` captures term's lost
        # bits; otherwise ``(term - t) + total`` captures total's lost bits.
        total_dominates = total.abs() >= term.abs()
        compensation = compensation + torch.where(
            total_dominates,
            (total - t) + term,
            (term - t) + total,
        )
        total = t
    total = total + compensation
    if keepdim:
        # When reducing all dims, restore the full original rank as size-1 dims.
        total = total.reshape((1,) * orig_ndim) if reduce_all else total.unsqueeze(dim)
    return total


def smoothstep(x: Tensorable) -> Tensor:
    """Hermite C¹ smoothstep: 3x² - 2x³, clamped to [0, 1].

    Has zero first derivative at x=0 and x=1.

    Args:
      x: Input tensor (values outside [0, 1] are clamped).

    Returns:
      result: Smoothly interpolated values in [0, 1].

    References:
      https://en.wikipedia.org/wiki/Smoothstep

    """
    x = convert_to_tensor(x)
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smoothstep_inverse(y: Tensorable) -> Tensor:
    """Inverse of smoothstep via Newton's method.

    Finds x such that 3x² - 2x³ = y for y in [0, 1].

    Args:
      y: Input tensor with values in [0, 1].

    Returns:
      result: Inverse-mapped values in [0, 1].

    References:
      https://en.wikipedia.org/wiki/Smoothstep

    """
    y = convert_to_tensor(y)
    # Newton's method: f(x) = 3x² - 2x³ - y, f'(x) = 6x(1-x).
    x = y.clone()
    for _ in range(12):
        f = x * x * (3.0 - 2.0 * x) - y
        df = 6.0 * x * (1.0 - x)
        # Avoid division by zero at boundaries.
        x = x - f / df.clamp(min=1e-12)
        x = torch.clamp(x, 0.0, 1.0)
    return x


def smootherstep(x: Tensorable) -> Tensor:
    """Perlin's C² smootherstep: 6x⁵ - 15x⁴ + 10x³, clamped to [0, 1].

    A smooth interpolant that has zero first and second derivatives at
    x=0 and x=1. Useful for noise schedules and smooth transitions.

    Args:
      x: Input tensor (values outside [0, 1] are clamped).

    Returns:
      result: Smoothly interpolated values in [0, 1].

    References:
      Perlin 2002, "Improving Noise."
      tfp.math.smootherstep

    """
    x = convert_to_tensor(x)
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def soft_threshold(x: Tensorable, threshold: Tensorable) -> Tensor:
    """Soft thresholding (proximal operator for L1 regularization).

    Computes sign(x) * max(|x| - threshold, 0). Used as the proximal
    operator in ISTA/FISTA and for sparse training.

    Args:
      x: Input tensor.
      threshold: Non-negative threshold value.

    Returns:
      result: Soft-thresholded tensor.

    References:
      tfp.math.soft_threshold

    """
    x, threshold = convert_to_tensor(x, threshold)
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def sqrt1pm1(x: Tensorable) -> Tensor:
    """Numerically stable sqrt(1 + x) - 1 for small x.

    For small |x|, direct computation suffers catastrophic cancellation.
    Uses the identity sqrt(1+x)-1 = x / (sqrt(1+x)+1).

    Args:
      x: Input tensor. The domain is x >= -1; at x = -1 the result is -1,
        and x < -1 (outside the domain) yields -1 from ``safe_sqrt`` rather
        than NaN.

    Returns:
      result: sqrt(1 + x) - 1.

    References:
      tfp.math.sqrt1pm1

    """
    x = convert_to_tensor(x)
    # For |x| < threshold, use the stable form x/(sqrt(1+x)+1).
    # For large |x|, direct computation is fine.
    threshold = torch.finfo(x.dtype).eps ** 0.25
    is_small = x.abs() < threshold
    safe_x = torch.where(is_small, x, 0.0)
    stable = safe_x / (safe_sqrt(1.0 + safe_x) + 1.0)
    direct = safe_sqrt(1.0 + x) - 1.0
    return torch.where(is_small, stable, direct)


def softsign(x: Tensorable) -> Tensor:
    """Softsign: x / (1 + |x|).

    Maps R to (-1, 1). Smoother saturation than tanh with lighter tails.

    Args:
      x: Input tensor.

    Returns:
      result: x / (1 + |x|).

    """
    x = convert_to_tensor(x)
    return x / (1.0 + torch.abs(x))


def softsign_inverse(y: Tensorable) -> Tensor:
    """Inverse softsign: y / (1 - |y|).

    Maps (-1, 1) to R. Inverse of softsign.

    Args:
      y: Input tensor with values in (-1, 1).

    Returns:
      result: y / (1 - |y|).

    """
    y = convert_to_tensor(y)
    return y / (1.0 - torch.abs(y))


def power_transform(x: Tensorable, power: float) -> Tensor:
    """Box-Cox-like power transform: (1 + x * c)^(1/c).

    Generalizes exp (power → 0) and linear (power = 1).
    Domain: x >= -1/power when power > 0.

    Args:
      x: Input tensor.
      power: Power parameter c. When 0, returns exp(x).

    Returns:
      result: (1 + x * power) ** (1 / power), or exp(x) if power == 0.

    References:
      Box & Cox 1964, "An Analysis of Transformations."
      tfp.bijectors.PowerTransform

    """
    x = convert_to_tensor(x)
    if power == 0:
        return torch.exp(x)
    # Domain is 1 + x*power > 0; safe_pow returns 0 outside it without
    # producing NaN values or NaN gradients (matches tfp.PowerTransform).
    return safe_pow(1.0 + x * power, 1.0 / power)


def power_transform_inverse(y: Tensorable, power: float) -> Tensor:
    """Inverse Box-Cox power transform: (y^c - 1) / c.

    Args:
      y: Input tensor (must be positive).
      power: Power parameter c. When 0, returns log(y).

    Returns:
      result: (y ** power - 1) / power, or log(y) if power == 0.

    """
    y = convert_to_tensor(y)
    if power == 0:
        return torch.log(y)
    return (torch.pow(y, power) - 1.0) / power


def softmax_centered(x: Tensorable, dim: int = -1) -> Tensor:
    """Bijective softmax via zero-pivot augmentation.

    Appends a zero coordinate, then applies softmax. This gives a
    bijection from R^n to the (n+1)-simplex.

    Args:
      x: Input tensor of shape (..., n).
      dim: Dimension along which to apply softmax.

    Returns:
      result: Tensor of shape (..., n+1) on the probability simplex.

    References:
      tfp.bijectors.SoftmaxCentered

    """
    x = convert_to_tensor(x)
    # Append a zero pivot along the target dimension.
    zero = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
    augmented = torch.cat([x, zero], dim=dim)
    return torch.softmax(augmented, dim=dim)


def softmax_centered_inverse(y: Tensorable, dim: int = -1) -> Tensor:
    """Inverse of softmax_centered: log-ratio against last coordinate.

    Args:
      y: Tensor on the (n+1)-simplex, shape (..., n+1).
      dim: Dimension along which softmax was applied.

    Returns:
      result: Tensor of shape (..., n).

    """
    y = convert_to_tensor(y)
    # Split off the pivot (last element along dim).
    log_y = torch.log(y)
    pivot = log_y.select(dim, -1).unsqueeze(dim)
    # Drop the pivot coordinate and subtract it from the rest.
    return log_y.narrow(dim, 0, y.shape[dim] - 1) - pivot


def sinh_arcsinh(
    x: Tensorable,
    skewness: Tensorable = 0.0,
    tailweight: Tensorable = 1.0,
) -> Tensor:
    """Sinh-arcsinh transform for controlling skewness and tail weight.

    Computes sinh((arcsinh(x) + skewness) * tailweight).

    When skewness=0 and tailweight=1, this is the identity.
    tailweight > 1 produces heavier tails; tailweight < 1 produces lighter.
    skewness != 0 introduces asymmetry.

    Args:
      x: Input tensor.
      skewness: Skewness parameter (0 = symmetric).
      tailweight: Tail weight parameter (1 = unchanged).

    Returns:
      result: Transformed tensor.

    References:
      Jones & Pewsey 2009, "Sinh-arcsinh distributions."
      tfp.bijectors.SinhArcsinh

    """
    x, skewness, tailweight = convert_to_tensor(x, skewness, tailweight)
    return torch.sinh((torch.arcsinh(x) + skewness) * tailweight)


def sinh_arcsinh_inverse(
    y: Tensorable,
    skewness: Tensorable = 0.0,
    tailweight: Tensorable = 1.0,
) -> Tensor:
    """Inverse sinh-arcsinh: sinh(arcsinh(y) / tailweight - skewness).

    Args:
      y: Input tensor.
      skewness: Skewness parameter.
      tailweight: Tail weight parameter.

    Returns:
      result: Inverse-transformed tensor.

    """
    y, skewness, tailweight = convert_to_tensor(y, skewness, tailweight)
    return torch.sinh(torch.arcsinh(y) / tailweight - skewness)


def custom_grad(*, actual: Tensor, phantom: Tensor) -> Tensor:
    """Return actual in forward, phantom in backward (STE pattern).

    Adapted from ``tfp.math.custom_grad`` (tensorflow/probability).

    Returns:
      result: actual.detach() + (phantom - phantom.detach()).

    """
    # Parenthesized to ensure exact cancellation under floating-point arithmetic.
    return actual.detach() + (phantom - phantom.detach())


def ste_round(x: Tensorable, scale: Tensorable = 1.0) -> Tensor:
    """Round with straight-through estimator gradient.

    In the forward pass, returns ``round(x * scale) / scale``.
    In the backward pass, gradients pass through as if this were
    the identity function.

    References:
      Bengio et al. 2013, "Estimating or Propagating Gradients Through
        Stochastic Neurons for Conditional Computation."

    """
    x, scale = convert_to_tensor(x, scale)
    return custom_grad(actual=torch.round(x * scale) / scale, phantom=x)


def ste_clamp(
    input: Tensorable,
    min: Tensorable | None = None,
    max: Tensorable | None = None,
) -> Tensor:
    """Clamp with straight-through estimator gradient.

    In the forward pass, returns ``clamp(input, min, max)``.
    In the backward pass, gradients pass through as if this were
    the identity function.

    References:
      Bengio et al. 2013, "Estimating or Propagating Gradients Through
        Stochastic Neurons for Conditional Computation."

    """
    input = convert_to_tensor(input)
    return custom_grad(actual=torch.clamp(input, min, max), phantom=input)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownArgumentType]  # ty: ignore[no-matching-overload]


def safe_log(input: Tensorable) -> Tensor:
    """Logarithm that returns -inf instead of NaN for non-positive inputs.

    Uses the double-where trick to avoid NaN gradients.
    Adapted from tensorflow/probability.

    Returns:
      result: log(input) where input > 0, -inf otherwise.

    References:
      https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    """
    input = convert_to_tensor(input)
    valid = input > 0
    x = torch.where(valid, input, 1.0)
    return torch.where(valid, torch.log(x), -math.inf)


def safe_xlogy(
    input: Tensorable,
    other: Tensorable,
) -> Tensor:
    """X log(y) that returns 0 instead of NaN when x == 0.

    Adapted from tensorflow/probability.

    Returns:
      result: x * log(y), with 0 where x == 0.

    """
    input, other = convert_to_tensor(input, other)
    return torch.xlogy(input, torch.where(input == 0.0, 1.0, other))


def safe_sqrt(input: Tensorable) -> Tensor:
    """Square root that returns 0 instead of NaN for non-positive inputs.

    Uses the double-where trick to avoid NaN gradients.
    Adapted from tensorflow/probability.

    Returns:
      result: sqrt(input) where input > 0, 0 otherwise.

    References:
      https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    """
    input = convert_to_tensor(input)
    valid = input > 0
    x = torch.clamp(input, min=0.0)
    return torch.where(valid, torch.sqrt(x), 0.0)


def safe_rsqrt(input: Tensorable) -> Tensor:
    """Reciprocal square root with finite gradients on the non-positive domain.

    Uses the double-where trick to avoid NaN gradients. Conventions:

    - ``input > 0``: ordinary ``rsqrt(input)``.
    - ``input == 0``: ``+inf`` (the mathematical limit of ``1 / sqrt(x)`` as
      ``x -> 0+``).
    - ``input < 0``: ``0`` (the safe fallback where ``rsqrt`` is undefined).

    The forward value at ``input == 0`` is ``+inf``, but the gradient stays
    finite (the double-where keeps the autograd branch on a clamped operand).

    Returns:
      result: ``rsqrt(input)`` for ``input > 0``; ``+inf`` at ``0``; ``0`` for
        negative input.

    References:
      https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    """
    input = convert_to_tensor(input)
    valid = input > 0
    x = torch.where(valid, input, 1.0)
    fallback = torch.where(input == 0, math.inf, 0.0)
    return torch.where(valid, torch.rsqrt(x), fallback)


def safe_pow(base: Tensorable, exponent: Tensorable) -> Tensor:
    """Power function that avoids NaN for non-positive bases with fractional exponents.

    Uses the double-where trick to avoid NaN gradients. Conventions:

    - ``base > 0``: ordinary ``base ** exponent``.
    - ``base == 0, exponent == 0``: ``1`` (the standard ``0 ** 0 == 1``
      convention shared by IEEE-754 ``pow``, NumPy, and ``torch.pow``).
    - ``base == 0, exponent != 0`` or ``base < 0``: ``0`` (the safe fallback
      for the non-positive domain where a real fractional power is undefined).

    Returns:
      result: ``base ** exponent`` on the positive domain; ``1`` for ``0 ** 0``;
        ``0`` elsewhere.

    References:
      https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    """
    base, exponent = convert_to_tensor(base, exponent)
    valid = base > 0
    x = torch.where(valid, base, 1.0)
    fallback = torch.where((base == 0) & (exponent == 0), 1.0, 0.0)
    return torch.where(valid, torch.pow(x, exponent), fallback)
