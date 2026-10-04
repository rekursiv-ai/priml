"""Statistical functions and utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from torch import Tensor, linalg

import torch

from priml.math.distributed import logmeanexp_all_to_all
from priml.math.numeric import logmeanexp
from priml.memory import convert_to_tensor


if TYPE_CHECKING:
    from priml.math.custom_types import Tensorable


type PcaDecompose = Callable[[Tensor], tuple[Tensor, Tensor]]
"""Factors a mean-centered ``(N, D)`` matrix into ascending eigenpairs.

The injection point of :func:`pca`: ``pca_eigh``, ``pca_svd``, and ``pca_power``
all satisfy it, and a caller can supply its own. Tuning knobs belong to one
implementation, so they ride in a ``partial`` rather than on ``pca``.

Returns:
  eigenvalues: Ascending, shape ``(D,)``.
  eigenvectors: Columns are the components, shape ``(D, D)``.
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
        observations = x.shape[0]
        if observations > 1 and not bias:
            observations -= 1
        return (x_centered * y_values).sum() / observations
    obs_dim = x.ndim - (1 if rowvar else 2)
    x = x - x.mean(dim=obs_dim, keepdim=True)
    y = x if y is None else y - y.mean(dim=obs_dim, keepdim=True)
    observations = x.shape[obs_dim]
    if observations > 1 and not bias:
        observations -= 1
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
    p = torch.softmax(x, dim=dim)
    log_q = torch.log_softmax(y, dim=dim)
    return -torch.sum(p * log_q, dim=dim, keepdim=keepdim)


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
    tiny = torch.finfo(q_t.dtype).tiny
    return -torch.sum(
        torch.where(p > 0, p * torch.log(q_t.clamp(min=tiny)), 0.0),
        dim=dim,
        keepdim=keepdim,
    )


def total_variation(
    p: Tensorable,
    q: Tensorable,
    dim: int | tuple[int, ...] = -1,
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
    return 0.5 * (p - q).abs().sum(dim, keepdim=keepdim)


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
    ensemble_dim = (
        (ensemble_dim,) if isinstance(ensemble_dim, int) else tuple(ensemble_dim)
    )
    event_dim = (event_dim,) if isinstance(event_dim, int) else tuple(event_dim)

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
    event_axes = (
        (dim % x.ndim,)
        if isinstance(dim, int)
        else tuple(axis % x.ndim for axis in dim)
    )
    if dim_mean is None:
        dim_mean = tuple(set(range(x.ndim)) - set(event_axes))
    elif not isinstance(dim_mean, int):
        dim_mean = tuple(dim_mean)
    if isinstance(dim, int):
        p = torch.softmax(x, dim=dim)
        log_q = torch.log_softmax(y, dim=dim)
    else:
        leading_axes = tuple(axis for axis in range(x.ndim) if axis not in event_axes)
        permutation = leading_axes + event_axes
        inverse = tuple(permutation.index(axis) for axis in range(x.ndim))
        leading_shape = tuple(x.shape[axis] for axis in leading_axes)
        event_shape = tuple(x.shape[axis] for axis in event_axes)
        x_flat = x.permute(permutation).reshape(*leading_shape, -1)
        y_flat = y.permute(permutation).reshape(*leading_shape, -1)
        p = torch.softmax(x_flat, dim=-1).reshape(*leading_shape, *event_shape)
        log_q = torch.log_softmax(y_flat, dim=-1).reshape(
            *leading_shape,
            *event_shape,
        )
        p = p.permute(inverse)
        log_q = log_q.permute(inverse)
    mean_p = torch.mean(p, dim=dim_mean, keepdim=keepdim_mean)
    log_mean_q = logmeanexp_all_to_all(
        log_q,
        dim=dim_mean,
        keepdim=keepdim_mean,
        world_size=world_size,
    )
    return -torch.sum(mean_p * log_mean_q, dim=dim, keepdim=keepdim)


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
    _U, s, vh = linalg.svd(x_centered, full_matrices=False)
    del _U
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
      eigenvalues: Shape ``(D,)``, ascending order.
      eigenvectors: Shape ``(D, D)``, columns are principal components.
        If ``whiten=True``, columns are scaled by ``1/sqrt(λ + eps)``.

    """
    x_t = convert_to_tensor(x).float()
    centered = x_t - x_t.mean(dim=0)
    eigenvalues, eigenvectors = decompose(centered)
    if whiten:
        eigenvectors = eigenvectors * torch.rsqrt(eigenvalues.unsqueeze(0) + eps)
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
    bounds = torch.quantile(x_t.reshape(-1), torch.tensor([q, 1 - q]))
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


# Returns (q, r) where q is orthogonal and r is upper triangular.
def _householder_qr(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Householder QR decomposition (MPS-native)."""
    m, n = mat.shape
    q = torch.eye(m, device=mat.device, dtype=mat.dtype)
    r = mat.clone()
    for k in range(min(m, n)):
        x = r[k:, k]
        alpha = -torch.sign(x[0]) * x.norm()
        v = x.clone()
        v[0] = v[0] - alpha
        v_norm = v.norm()
        # Skip the reflection when the column is already (near) axis-aligned;
        # scale the dtype epsilon by the row count for the accumulated error.
        if v_norm < torch.finfo(mat.dtype).eps * mat.shape[0]:
            continue
        v = v / v_norm
        r[k:, k:] = r[k:, k:] - 2 * v.unsqueeze(0).T @ (v.unsqueeze(0) @ r[k:, k:])
        q[:, k:] = q[:, k:] - 2 * (q[:, k:] @ v.unsqueeze(0).T) @ v.unsqueeze(0)
    return q, r


def _rayleigh(sigma: Tensor, basis: Tensor) -> Tensor:
    """Estimate eigenvalues along the basis columns."""
    return (basis * (sigma @ basis)).sum(dim=0)
