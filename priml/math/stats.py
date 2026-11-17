"""Statistical functions and utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import math
import statistics

from torch import Tensor, linalg

import numpy as np
import torch

from priml.math.basic import reduction_dims
from priml.math.distributed import logmeanexp_all_to_all
from priml.math.numeric import logmeanexp
from priml.memory import convert_to_tensor


if TYPE_CHECKING:
    from collections.abc import Hashable

    from scipy.stats import t

    from priml.math.custom_types import Tensorable
else:
    from wrapt import lazy_import

    t = lazy_import("scipy.stats", "t")


type PcaDecompose = Callable[[Tensor], tuple[Tensor, Tensor]]
"""Factors a mean-centered ``(N, D)`` matrix into ascending eigenpairs.

The injection point of :func:`pca`: ``pca_eigh``, ``pca_svd``, and ``pca_power``
all satisfy it, and a caller can supply its own. Tuning knobs belong to one
implementation, so they ride in a ``partial`` rather than on ``pca``.

Returns:
  eigenvalues: Ascending, shape ``(K,)``; K is D for covariance decompositions
    and min(N, D) for a thin SVD.
  eigenvectors: Columns are the components, shape ``(D, K)``.
"""


def cov(
    x: Tensorable,
    y: Tensorable | None = None,
    bias: bool = True,
    rowvar: bool = False,
) -> Tensor:
    """Covariance matrix of x (and optionally cross-covariance with y).

    Args:
      x: Input tensor. Columns are variables by default.
      y: Optional second tensor for cross-covariance.
      bias: If True, normalize by N (biased). If False, by N-1.
      rowvar: If True, rows are variables instead of columns.

    Returns:
      covariance: Covariance matrix.

    References:
      numpy.cov

    """
    if y is None:
        x = convert_to_tensor(x)
    else:
        x, y = convert_to_tensor(x, y)
    if x.ndim == 1:
        x_centered = x - x.mean()
        y_values = x_centered if y is None else y - y.mean()
        observations = x.shape[0] - int(not bias)
        return (x_centered * y_values).sum() / observations
    obs_dim = x.ndim - (1 if rowvar else 2)
    x = x - x.mean(dim=obs_dim, keepdim=True)
    y = x if y is None else y - y.mean(dim=obs_dim, keepdim=True)
    observations = x.shape[obs_dim] - int(not bias)
    if rowvar:
        result = x @ y.transpose(-1, -2)
    else:
        result = x.transpose(-1, -2) @ y
    return result / observations


def entropy_logits(
    x: Tensorable,
    y: Tensorable | None = None,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """Cross-entropy H(softmax(x), softmax(y)), or entropy if y is None.

    Args:
      x: Logits; softmax(x) is the first distribution.
      y: Logits for cross-entropy; if None, computes H(softmax(x)).
      dim: Dimension(s) over which to compute softmax and sum.
      keepdim: If True, reduced dimensions are kept with size 1.

    Returns:
      entropy: Scalar or reduced tensor.

    """
    if y is None:
        x = convert_to_tensor(x)
        y = x
    else:
        x, y = convert_to_tensor(x, y)
    dims = reduction_dims(dim, ndim=x.ndim)
    p, log_q = _event_distributions(x, y, dims)
    return -torch.sum(
        p * torch.where(p == 0, 0.0, log_q),
        dim=dims,
        keepdim=keepdim,
    )


def entropy_probs(
    p: Tensorable,
    q: Tensorable | None = None,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """Cross-entropy H(p, q), or entropy H(p) if q is None.

    Args:
      p: Probability distribution (elements sum to 1 along dim).
      q: Reference distribution for cross-entropy; if None, computes H(p).
      dim: Dimension(s) over which to sum the entropy.
      keepdim: If True, reduced dimensions are kept with size 1.

    Returns:
      entropy: Scalar or reduced tensor.

    """
    if q is None:
        p = convert_to_tensor(p)
        q_t = p
    else:
        p, q_t = convert_to_tensor(p, q)
    safe_q = torch.where(p == 0, 1.0, q_t)
    dims = reduction_dims(dim, ndim=p.ndim)
    return -torch.sum(p * torch.log(safe_q), dim=dims, keepdim=keepdim)


def total_variation(
    p: Tensorable,
    q: Tensorable,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """Total-variation distance between two distributions, ``0.5 sum |p - q|``.

    Args:
      p: Probability distribution (elements sum to 1 along dim).
      q: Probability distribution, shaped like ``p``.
      dim: Dimension(s) over which to sum.
      keepdim: If True, reduced dimensions are kept with size 1.

    Returns:
      distance: In ``[0, 1]``, reduced over ``dim``.

    """
    p, q = convert_to_tensor(p, q)
    dims = reduction_dims(dim, ndim=p.ndim)
    return 0.5 * (p - q).abs().sum(dims, keepdim=keepdim)


def jsd(
    logp: Tensorable,
    *,
    ensemble_dim: int | Sequence[int] = 0,
    event_dim: int | Sequence[int] = -1,
) -> Tensor:
    """Jensen-Shannon divergence over an ensemble of log-probability distributions.

    JSD(p₁,...,pₖ) = H(mean(pᵢ)) - mean(H(pᵢ)).

    Args:
      logp: Log-probabilities, shape [..., K, ..., num_classes, ...].
      ensemble_dim: Dimension(s) indexing ensemble members (K).
      event_dim: Dimension(s) indexing the event space (num_classes).

    Returns:
      jsd: JSD values with ensemble and event dims squeezed.

    References:
      Lin 1991, "Divergence measures based on the Shannon entropy."

    """
    logp = convert_to_tensor(logp)
    ensemble_dim = reduction_dims(ensemble_dim, ndim=logp.ndim)
    event_dim = reduction_dims(event_dim, ndim=logp.ndim)

    def _entropy(lp: Tensor) -> Tensor:
        safe = torch.where(torch.isneginf(lp), torch.zeros_like(lp), lp)
        return -torch.sum(lp.exp() * safe, dim=event_dim, keepdim=True)

    # H(mean_p): entropy of the mixture.
    log_avg_p = logmeanexp(logp, dim=ensemble_dim, keepdim=True)
    h_avg = _entropy(log_avg_p)
    # mean(H(pᵢ)): average entropy of each member.
    avg_h = torch.mean(_entropy(logp), dim=ensemble_dim, keepdim=True)
    squeeze_dims = tuple(sorted({*ensemble_dim, *event_dim}))
    return (h_avg - avg_h).squeeze(dim=squeeze_dims)


def entropy_logits_mean_all_to_all(
    x: Tensorable,
    y: Tensorable | None = None,
    *,
    dim: int | Sequence[int] = -1,
    dim_mean: int | Sequence[int] | None = None,
    keepdim: bool = False,
    keepdim_mean: bool = False,
    world_size: int | None = None,
) -> Tensor:
    """Distributed cross-entropy of mean distributions.

    Computes H(mean_p, mean_q) where means are taken over dim_mean
    using all_gather for the log-mean. Analogous to the mean-teacher
    entropy used in semi-supervised learning.

    Args:
      x: Logits; softmax(x) is the first distribution.
      y: Logits for cross-entropy; if None, computes H(softmax(x)).
      dim: Dimension(s) for softmax and final summation.
      dim_mean: Dimensions over which to average before computing log-mean.
      keepdim: If True, reduced dimensions in dim are kept with size 1.
      keepdim_mean: If True, reduced dimensions in dim_mean are kept.
      world_size: Number of ranks in the all_gather; if None, auto-detect.

    Returns:
      entropy: Cross-entropy of the averaged distributions.

    """
    if y is None:
        x = convert_to_tensor(x)
        y = x
    else:
        x, y = convert_to_tensor(x, y)
    event_axes = reduction_dims(dim, ndim=x.ndim)
    mean_axes = (
        tuple(sorted(set(range(x.ndim)) - set(event_axes)))
        if dim_mean is None
        else reduction_dims(dim_mean, ndim=x.ndim)
    )
    p, log_q = _event_distributions(x, y, event_axes)
    mean_p = torch.mean(p, dim=mean_axes, keepdim=keepdim_mean)
    log_mean_q = logmeanexp_all_to_all(
        log_q,
        dim=mean_axes,
        keepdim=keepdim_mean,
        world_size=world_size,
    )
    if not keepdim_mean:
        event_axes = tuple(
            axis - sum(mean_axis < axis for mean_axis in mean_axes)
            for axis in event_axes
        )
    safe_log_q = torch.where(mean_p == 0, 0.0, log_mean_q)
    return -torch.sum(mean_p * safe_log_q, dim=event_axes, keepdim=keepdim)


def pca_eigh(x_centered: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose the covariance matrix with ``linalg.eigh`` (CUDA/CPU).

    MPS does not implement ``linalg.eigh``; use :func:`pca_power` there.

    Args:
      x_centered: Mean-centered observations of shape ``(N, D)``.

    Returns:
      eigenvalues: Ascending eigenvalues of shape ``(D,)``.
      eigenvectors: Columns are the corresponding eigenvectors ``(D, D)``.

    """
    sigma = (x_centered.T @ x_centered) / len(x_centered)
    return linalg.eigh(sigma)


def pca_svd(x_centered: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose the data matrix with ``linalg.svd`` (CUDA/CPU).

    MPS does not implement ``linalg.svd``; use :func:`pca_power` there.

    Args:
      x_centered: Mean-centered observations of shape ``(N, D)``.

    Returns:
      eigenvalues: Ascending eigenvalues of shape ``(min(N, D),)``, flipped from
        SVD's descending order to match the ``eigh`` convention.
      eigenvectors: Right singular vectors as columns, shape ``(D, min(N, D))``.

    Raises:
      RuntimeError: If the input lives on an MPS device.

    """
    if x_centered.device.type == "mps":
        raise RuntimeError("pca_svd is not supported on MPS; use pca_power instead.")
    _, s, vh = linalg.svd(x_centered, full_matrices=False)
    eigenvalues = s * s / len(x_centered)
    eigenvectors = vh.T
    return eigenvalues.flip(0), eigenvectors.flip(1)


def pca_power(
    x_centered: Tensor,
    num_iters: int = 200,
    tol: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Decompose by simultaneous power iteration with QR orthogonalization.

    Only uses matmul and basic arithmetic -- runs natively on MPS
    (no CPU fallback). Uses Householder QR for numerical stability.

    Args:
      x_centered: Mean-centered observations of shape ``(N, D)``.
      num_iters: Maximum power-iteration sweeps.
      tol: Subspace-convergence threshold; when > 0, iteration stops once the
        basis update changes the eigenvalue estimate by less than ``tol``.

    Returns:
      eigenvalues: Ascending eigenvalues of shape ``(D,)``.
      eigenvectors: Columns are the corresponding eigenvectors ``(D, D)``.

    """
    sigma = (x_centered.T @ x_centered) / len(x_centered)
    basis = torch.randn(*sigma.shape, device=sigma.device, dtype=sigma.dtype)
    basis, _ = _householder_qr(basis)
    prev = _rayleigh(sigma, basis)
    for _ in range(num_iters):
        basis, _ = _householder_qr(sigma @ basis)
        if tol > 0:
            eigenvalues = _rayleigh(sigma, basis)
            if (eigenvalues - prev).abs().max() < tol:
                break
            prev = eigenvalues
    eigenvalues = _rayleigh(sigma, basis)
    idx = eigenvalues.argsort()
    return eigenvalues[idx], basis[:, idx]


def pca(
    x: Tensorable,
    *,
    whiten: bool = False,
    eps: float = 0.0,
    decompose: PcaDecompose = pca_eigh,
) -> tuple[Tensor, Tensor]:
    """PCA decomposition via eigendecomposition of the covariance matrix.

    Computes eigenvectors (principal components) and eigenvalues from the
    population covariance (divides by N, not N-1) of ``x``. The input is
    treated as a matrix of observations (rows) x features (columns).
    Eigenvectors are returned in ascending eigenvalue order (last =
    largest variance). Input is cast to float32 for numerical stability.

    When ``whiten=True``, eigenvectors are scaled by ``1/sqrt(λ + eps)``
    so that projecting data onto them produces unit-variance components.

    This is a **fit** function -- it returns the decomposition, not
    transformed data. To apply::

        eigenvalues, eigenvectors = pca(x, whiten=True, eps=1e-5)
        projected = x @ eigenvectors            # PCA projection
        whitened = (x - x.mean(0)) @ eigenvectors  # PCA whitening
        # ZCA whitening (whiten in original basis):
        zca = (x - x.mean(0)) @ eigenvectors @ eigenvectors.T

    Args:
      x: Input tensor of shape ``(N, D)`` (observations x features).
      whiten: Scale eigenvectors by ``1/sqrt(λ + eps)``.
      eps: Regularization added to eigenvalues (only used when
        ``whiten=True``).
      decompose: Eigendecomposition of the mean-centered matrix. Defaults to
        :func:`pca_eigh`; :func:`pca_svd` decomposes the data matrix
        directly, and :func:`pca_power` runs natively on MPS. Tuning knobs
        belong to the implementation, so bind them at the call site::

            pca(x, decompose=functools.partial(pca_power, num_iters=50))

    Returns:
      eigenvalues: Shape ``(K,)``, ascending order; K is D for covariance
        decompositions and min(N, D) for a thin SVD.
      eigenvectors: Shape ``(D, K)``, columns are principal components.
        If ``whiten=True``, columns are scaled by ``1/sqrt(max(λ, 0) + eps)``;
        zero-variance columns are zero when eps is zero.

    """
    x_t = convert_to_tensor(x).float()
    centered = x_t - x_t.mean(dim=0)
    eigenvalues, eigenvectors = decompose(centered)
    if whiten:
        variance = eigenvalues.clamp(min=0) + eps
        positive = variance > 0
        inverse_scale = torch.where(
            positive,
            torch.rsqrt(torch.where(positive, variance, 1.0)),
            0.0,
        )
        eigenvectors = eigenvectors * inverse_scale.unsqueeze(0)
    return eigenvalues, eigenvectors


def quantile_normalize(x: Tensorable, q: float = 1e-3) -> Tensor:
    """Normalize x to [0, 1] using robust quantile bounds.

    Args:
      x: Input tensor.
      q: Quantile for robust min/max (default 0.001).

    Returns:
      normalized: Tensor scaled to approximately [0, 1].

    """
    x_t = convert_to_tensor(x)
    bounds = torch.quantile(
        x_t.reshape(-1),
        torch.tensor([q, 1 - q], dtype=x_t.dtype, device=x_t.device),
    )
    lo, hi = bounds[0], bounds[1]
    span = hi - lo
    # Constant (or near-constant) input collapses the range; map it to 0
    # rather than emitting inf/nan from a zero denominator. The safe denominator
    # keeps the unused ``where`` branch NaN-free so gradients stay clean.
    safe_span = torch.where(span > 0, span, 1.0)
    return torch.where(span > 0, (x_t - lo) / safe_span, 0.0)


def holm(p_values: Tensorable) -> Tensor:
    """Adjust a family of p-values by Holm's step-down procedure.

    The ``k``-th smallest of ``n`` p-values is multiplied by ``n - k + 1``,
    clamped at 1, and raised to the largest adjusted value ranked before it.
    Rejecting each test whose adjusted p-value is at most ``alpha`` holds the
    family-wise error rate at ``alpha``; the threshold stays the caller's.

    Args:
      p_values: Raw p-values ``[..., n]``, one family along the last axis;
        Python floats become float64.

    Returns:
      adjusted: Adjusted p-values, in the input's order and shape.

    References:
      Holm 1979, "A simple sequentially rejective multiple test procedure."

    """
    p = convert_to_tensor(p_values, dtype_hint=torch.float64)
    ordered, order = p.sort(dim=-1, stable=True)
    scale = torch.arange(p.shape[-1], 0, -1, dtype=p.dtype, device=p.device)
    adjusted = (ordered * scale).clamp(max=1).cummax(-1).values
    return torch.empty_like(adjusted).scatter_(-1, order, adjusted)


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    clusters: Sequence[Hashable],
    estimand: Literal["cluster_mean", "observation_mean"],
    resamples: int,
    seed: int,
    level: float,
) -> tuple[float, float, float]:
    """Estimate a mean paired difference with a percentile bootstrap over clusters.

    Each resample draws as many clusters as were observed, with replacement.
    The bounds index the sorted resample estimates at ``int(tail * resamples)``
    and ``int((1 - tail) * resamples)``, the latter clamped to the last
    resample, without interpolation.

    Args:
      differences: Complete paired differences, treated minus control. The
        caller removes or rejects incomplete pairs; none are imputed here.
      clusters: One cluster label per difference. Distinct labels treat the
        observations as independent.
      estimand: ``"cluster_mean"`` weighs clusters equally: the mean of
        cluster means. ``"observation_mean"`` weighs observations equally: the
        resampled clusters' difference totals over their observation counts.
      resamples: Bootstrap resamples of clusters.
      seed: Seed of the resampling generator.
      level: Two-sided interval coverage, strictly between 0 and 1.

    Returns:
      estimate: The estimand over the observed clusters.
      low: Lower percentile bound.
      high: Upper percentile bound.

    Raises:
      ValueError: No differences, a nonpositive resample count, a level
        outside the open unit interval, or not one label per difference.

    References:
      Efron 1979, "Bootstrap methods: another look at the jackknife."

    """
    if not differences:
        raise ValueError("No paired differences.")
    if resamples < 1:
        raise ValueError(f"Resamples must be positive, got {resamples}.")
    _check_level(level)
    grouped: dict[Hashable, list[float]] = {}
    for label, difference in zip(clusters, differences, strict=True):
        grouped.setdefault(label, []).append(difference)
    groups = list(grouped.values())
    # numpy's generator, not torch's: seeded intervals already reported must
    # keep their exact draws, which a torch generator would change.
    draws = np.random.default_rng(seed).integers(
        low=0,
        high=len(groups),
        size=(resamples, len(groups)),
    )
    if estimand == "cluster_mean":
        means = [statistics.fmean(group) for group in groups]
        estimate = statistics.fmean(means)
        boot = np.sort(np.array(means)[draws].mean(axis=1))
    else:
        totals = np.array([math.fsum(group) for group in groups])
        counts = np.array([len(group) for group in groups])
        estimate = statistics.fmean(differences)
        boot = np.sort(totals[draws].sum(axis=1) / counts[draws].sum(axis=1))
    tail = (1 - level) / 2
    high_index = min(int((1 - tail) * resamples), resamples - 1)
    return (
        estimate,
        cast(float, boot[int(tail * resamples)]),
        cast(float, boot[high_index]),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedTInterval:
    """Mean paired difference with its two-sided Student t interval.

    Attributes:
      count: Paired differences observed.
      mean: Their mean; NaN when there are none.
      bounds: Lower and upper bounds, or None below the minimum pair count.

    """

    count: int
    mean: float
    bounds: tuple[float, float] | None


def paired_t_interval(
    differences: Sequence[float],
    *,
    level: float,
    min_pairs: int,
) -> PairedTInterval:
    """Estimate a mean paired difference with a two-sided Student t interval.

    Args:
      differences: Complete paired differences, treated minus control.
      level: Two-sided interval coverage, strictly between 0 and 1.
      min_pairs: Fewest differences that get bounds. The interval has
        ``count - 1`` degrees of freedom, so meaningful values start at 2.

    Returns:
      interval: Count, mean and bounds; the bounds are None below
        ``min_pairs``.

    Raises:
      ValueError: A level outside the open unit interval.

    References:
      Student 1908, "The probable error of a mean."

    """
    _check_level(level)
    count = len(differences)
    if count < min_pairs:
        mean = statistics.fmean(differences) if differences else math.nan
        return PairedTInterval(count=count, mean=mean, bounds=None)
    mean = statistics.fmean(differences)
    se = statistics.stdev(differences) / cast(float, count**0.5)
    margin = float(t.ppf((1 + level) / 2, df=count - 1)) * se
    return PairedTInterval(
        count=count,
        mean=mean,
        bounds=(mean - margin, mean + margin),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PairedBinaryTable:
    """Paired binary outcomes of a reference and a candidate over the same units.

    Attributes:
      rescued: Units only the candidate succeeded on.
      lost: Units only the reference succeeded on.
      both: Units both succeeded on.
      neither: Units neither succeeded on.

    """

    rescued: int
    lost: int
    both: int
    neither: int

    @property
    def net(self) -> int:
        """Rescued minus lost units."""
        return self.rescued - self.lost


def paired_binary_table(
    reference: Sequence[bool],
    candidate: Sequence[bool],
) -> PairedBinaryTable:
    """Count rescued, lost, shared and shared-failure units of paired outcomes.

    Args:
      reference: Reference success per unit.
      candidate: Candidate success on the same units, in the same order.

    Returns:
      table: Counts per outcome pair; an interval on the net change is the
        bootstrap of ``candidate - reference``.

    Raises:
      ValueError: The two outcome vectors differ in length.

    """
    pairs = Counter(zip(reference, candidate, strict=True))
    return PairedBinaryTable(
        rescued=pairs[False, True],
        lost=pairs[True, False],
        both=pairs[True, True],
        neither=pairs[False, False],
    )


def exact_binomial_bounds(
    successes: int,
    *,
    count: int,
    tail: float,
) -> tuple[float, float]:
    """Return exact binomial rate bounds with ``tail`` probability in each tail.

    The bounds follow the definition: ``low`` is the rate at which
    ``P(X >= successes)`` equals ``tail`` (0 when ``successes`` is 0), and
    ``high`` the rate at which ``P(X <= successes)`` equals ``tail`` (1 when
    ``successes`` is ``count``). Each tail is a float64 ``logsumexp`` of
    ``lgamma`` log-pmfs, solved by bisection over the float64 bit patterns of
    the rate, which ends at adjacent floats within 64 steps.

    Args:
      successes: Observed successes, from 0 through ``count``.
      count: Observed trials, at least 1.
      tail: Probability outside each bound, strictly between 0 and 0.5, so
        coverage is ``1 - 2 * tail``.

    Returns:
      low: Lower rate bound.
      high: Upper rate bound.

    Raises:
      ValueError: Successes outside ``[0, count]``, no trials, or a tail
        outside the open interval (0, 0.5).

    References:
      Clopper & Pearson 1934, "The use of confidence or fiducial limits
        illustrated in the case of the binomial."

    """
    if count < 1 or successes < 0 or successes > count:
        raise ValueError(
            f"Successes must lie in [0, count] with count at least 1, "
            f"got {successes} of {count}.",
        )
    if math.isnan(tail) or tail <= 0 or 2 * tail >= 1:
        raise ValueError(f"Tail must lie strictly between 0 and 0.5, got {tail}.")
    low = (
        _binomial_tail_root(successes, count, count=count, tail=tail, rising=True)
        if successes
        else 0.0
    )
    high = (
        _binomial_tail_root(0, successes, count=count, tail=tail, rising=False)
        if successes < count
        else 1.0
    )
    return low, high


def rate_difference_lower_bound(
    candidate: int,
    *,
    reference: int,
    count: int,
    tail: float,
) -> tuple[float, float]:
    """Bound ``candidate_rate - reference_rate`` from below, conservatively.

    The bound is the candidate's exact lower bound minus the reference's exact
    upper bound, each with ``tail`` beyond it, so it holds with probability at
    least ``1 - 2 * tail``. Margins and splits of alpha across comparisons stay
    with the caller.

    Args:
      candidate: Candidate successes.
      reference: Reference successes.
      count: Trials in each arm.
      tail: Probability beyond each of the two exact bounds.

    Returns:
      lower: Lower bound on the rate difference.
      difference: Observed rate difference.

    References:
      Clopper & Pearson 1934, "The use of confidence or fiducial limits
        illustrated in the case of the binomial."

    """
    candidate_low, _ = exact_binomial_bounds(candidate, count=count, tail=tail)
    _, reference_high = exact_binomial_bounds(reference, count=count, tail=tail)
    return candidate_low - reference_high, (candidate - reference) / count


def ema_update(current: float, new_value: float, alpha: float = 0.3) -> float:
    """Exponential moving average step.

    Args:
      current: Current EMA value.
      new_value: New observation.
      alpha: Weight for new observation (0 to 1).

    Returns:
      value: alpha * new_value + (1 - alpha) * current.

    """
    return alpha * new_value + (1 - alpha) * current


def average_ranks(x: Tensorable, *, descending: bool = False) -> Tensor:
    """Return zero-based binary64 ranks, averaging equal values.

    Args:
      x: Ordered numeric values in one vector; NaNs are outside this contract.
      descending: Give the largest value the smallest rank.

    Returns:
      ranks: Complete ranks in the original input order, on the input device.

    """
    values = convert_to_tensor(x)
    if values.numel() == 0:
        return values.double()
    order = values.argsort(descending=descending)
    ordered = values[order]
    starts = torch.cat(
        (order.new_zeros(1), (ordered[1:] != ordered[:-1]).nonzero().flatten() + 1),
    )
    ends = torch.cat((starts[1:], order.new_full((1,), len(values))))
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.repeat_interleave(
        (starts + ends - 1).double() / 2,
        ends - starts,
    )
    return ranks


def ordinal_ranks(x: Tensorable, *, order: Tensor | None = None) -> Tensor:
    """Return zero-based ranks using a supplied complete sort permutation.

    Args:
      x: Numeric values in one vector.
      order: Complete sorted row indices, including the caller's tie policy.
        None uses Torch's ascending argsort.

    Returns:
      ranks: Binary64 ordinal ranks in the original row order.

    """
    values = convert_to_tensor(x)
    permutation = values.argsort() if order is None else order
    return permutation.argsort().double()


def spearman(x: Tensorable, y: Tensorable) -> Tensor:
    """Return binary64 centered-dot correlation of average ranks.

    Args:
      x: First complete numeric vector without NaNs.
      y: Paired values of the same length.

    Returns:
      correlation: Scalar binary64 correlation; NaN for fewer than two rows
        or a constant input. Direction and percentile normalization stay with
        the caller.

    """
    first, second = average_ranks(x), average_ranks(y)
    if first.numel() < 2:
        return first.new_full((), torch.nan)
    first, second = first - first.mean(), second - second.mean()
    return first @ second / (first.norm() * second.norm())


def binary_roc_auc(labels: Tensorable, scores: Tensorable) -> Tensor:
    """Integrate a complete tie-aware ROC curve in binary64.

    Args:
      labels: Binary membership vector in observation order.
      scores: Corresponding finite confidence scores, without a sigmoid.

    Returns:
      area: Binary64 trapezoid area after dropping collinear thresholds;
        NaN for an empty or single-class input.

    """
    membership, values = convert_to_tensor(labels), convert_to_tensor(scores)
    if values.numel() == 0:
        return values.new_full((), torch.nan, dtype=torch.float64)
    order = values.argsort(descending=True)
    ordered = values[order]
    ends = torch.cat(
        (
            (ordered[1:] != ordered[:-1]).nonzero().flatten(),
            order.new_full((1,), len(order) - 1),
        ),
    )
    true_count = membership[order].cumsum(0, dtype=torch.float64)[ends]
    false_count = ends + 1 - true_count
    if ends.numel() > 2:
        corners = torch.cat(
            (
                ends.new_ones(1, dtype=torch.bool),
                (false_count.diff(n=2) != 0) | (true_count.diff(n=2) != 0),
                ends.new_ones(1, dtype=torch.bool),
            ),
        )
        true_count, false_count = true_count[corners], false_count[corners]
    positives = membership.sum(dtype=torch.float64)
    tpr = torch.cat((true_count.new_zeros(1), true_count)) / positives
    fpr = torch.cat((false_count.new_zeros(1), false_count)) / (len(values) - positives)
    return torch.trapezoid(tpr, x=fpr)


class SlidingWindow:
    """Sliding window for computing throughput and other rates.

    Maintains (timestamp, cumulative_count) pairs and computes rates
    with Laplace smoothing.

    """

    def __init__(
        self,
        window_sec: float = 30.0,
        pseudocount: float = 2.0,
        pseudotime: float = 0.1,
    ):
        self.window_sec = window_sec
        self.pseudocount = pseudocount
        self.pseudotime = pseudotime
        self.samples: list[tuple[float, float]] = []

    def add(self, timestamp: float, cumulative_count: float) -> None:
        """Record an observation and prune expired entries.

        Args:
          timestamp: Timestamp.
          cumulative_count: Cumulative count.

        """
        self.samples.append((timestamp, cumulative_count))
        cutoff = timestamp - self.window_sec
        self.samples = [(t, c) for t, c in self.samples if t >= cutoff]

    def compute_rate(self, current_time: float, current_count: float) -> float:
        """Compute items/sec over the window with Laplace smoothing.

        Args:
          current_time: Timestamp in seconds.
          current_count: Cumulative count at current_time.

        Returns:
          rate: Items per second with Laplace smoothing added.

        """
        if len(self.samples) < 2:
            return 0.0
        t0, c0 = self.samples[0]
        elapsed = current_time - t0
        items = current_count - c0
        return (items + self.pseudocount) / (elapsed + self.pseudotime)


def _event_distributions(
    x: Tensor,
    y: Tensor,
    event_axes: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Normalize joint event dimensions while preserving their original axes."""
    if len(event_axes) == 1:
        return (
            torch.softmax(x, dim=event_axes[0]),
            torch.log_softmax(y, dim=event_axes[0]),
        )
    leading_axes = tuple(axis for axis in range(x.ndim) if axis not in event_axes)
    permutation = leading_axes + event_axes
    inverse = tuple(permutation.index(axis) for axis in range(x.ndim))
    leading_shape = tuple(x.shape[axis] for axis in leading_axes)
    event_shape = tuple(x.shape[axis] for axis in event_axes)
    x_flat = x.permute(permutation).reshape(*leading_shape, -1)
    y_flat = y.permute(permutation).reshape(*leading_shape, -1)
    p = torch.softmax(x_flat, dim=-1).reshape(*leading_shape, *event_shape)
    log_q = torch.log_softmax(y_flat, dim=-1).reshape(*leading_shape, *event_shape)
    return p.permute(inverse), log_q.permute(inverse)


# Returns (q, r) where q is orthogonal and r is upper triangular.
def _householder_qr(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Householder QR decomposition (MPS-native)."""
    m, n = mat.shape
    q = torch.eye(m, device=mat.device, dtype=mat.dtype)
    r = mat.clone()
    threshold = torch.finfo(mat.dtype).eps * mat.norm()
    for k in range(min(m, n)):
        x = r[k:, k]
        alpha = -torch.where(x[0] >= 0, 1.0, -1.0) * x.norm()
        v = x.clone()
        v[0] = v[0] - alpha
        v_norm = v.norm()
        if v_norm <= threshold:
            continue
        v = v / v_norm
        r[k:, k:] = r[k:, k:] - 2 * v.unsqueeze(0).T @ (v.unsqueeze(0) @ r[k:, k:])
        q[:, k:] = q[:, k:] - 2 * (q[:, k:] @ v.unsqueeze(0).T) @ v.unsqueeze(0)
    return q, r


def _rayleigh(sigma: Tensor, basis: Tensor) -> Tensor:
    """Estimate eigenvalues along the basis columns."""
    return (basis * (sigma @ basis)).sum(dim=0)


# ``rising`` says the tail mass grows with the rate. The returned float is the bisection
# end whose mass is at least ``tail``.
def _binomial_tail_root(
    first: int,
    last: int,
    *,
    count: int,
    tail: float,
    rising: bool,
) -> float:
    """Return the rate at which ``P(first <= X <= last)`` crosses ``tail``."""
    k = torch.arange(first, last + 1, dtype=torch.float64)
    n = torch.tensor(count, dtype=torch.float64)
    log_choose = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    log_tail = math.log(tail)
    bracket = torch.tensor((0.0, 1.0), dtype=torch.float64).view(torch.int64)
    low_bits, high_bits = int(bracket[0]), int(bracket[1])
    while high_bits - low_bits > 1:
        mid_bits = (low_bits + high_bits) // 2
        rate = float(torch.tensor(mid_bits).view(torch.float64))
        log_mass = torch.logsumexp(
            log_choose + k * math.log(rate) + (n - k) * math.log1p(-rate),
            dim=0,
        )
        if (float(log_mass) < log_tail) == rising:
            low_bits = mid_bits
        else:
            high_bits = mid_bits
    return float(torch.tensor(high_bits if rising else low_bits).view(torch.float64))


def _check_level(level: float) -> None:
    """Reject an interval coverage outside the open unit interval."""
    if math.isnan(level) or level <= 0 or level >= 1:
        raise ValueError(f"Level must lie strictly between 0 and 1, got {level}.")
