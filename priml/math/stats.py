"""Statistical functions and utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import Tensor

import torch
import torch.linalg

from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.distributed import logmeanexp_all_to_all
from priml.math.numeric import logmeanexp


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
    # A 1-D input is a single variable's observations; numpy.cov treats it as
    # one row and returns the scalar variance.
    scalar = x.ndim == 1
    if scalar:
        x = x.unsqueeze(0)
        if y is not None:
            y = y.unsqueeze(0)
        rowvar = True
    obs_dim = -1 if rowvar else -2
    x = x - x.mean(dim=obs_dim, keepdim=True)
    y = x if y is None else (y - y.mean(dim=obs_dim, keepdim=True))
    eq = "...ij,...kj->...ik" if rowvar else "...ji,...jk->...ik"
    n = x.shape[obs_dim]
    if n > 1 and not bias:
        n = n - 1
    result = torch.einsum(eq, x, y) / n
    return result.squeeze() if scalar else result


def entropy_logits(
    x: Tensorable,
    y: Tensorable | None = None,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """Cross-entropy H(softmax(x), softmax(y)), or entropy if y is None.

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
        safe = torch.where(torch.isneginf(lp), 0.0, lp)
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

    Returns:
      entropy: Cross-entropy of the averaged distributions.

    """
    if y is None:
        x = convert_to_tensor(x)
        y = x
    else:
        x, y = convert_to_tensor(x, y)
    if dim_mean is None:
        dim_iter = (dim,) if isinstance(dim, int) else dim
        dim_mean = tuple(set(range(x.ndim)) - {a % x.ndim for a in dim_iter})
    p = torch.softmax(x, dim=dim)
    assert dim_mean is not None
    mean_p = torch.mean(p, dim=dim_mean, keepdim=keepdim_mean)
    log_q = torch.log_softmax(y, dim=dim)
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
    return torch.linalg.eigh(sigma)


def pca_svd(x_centered: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose the data matrix with ``linalg.svd`` (CUDA/CPU).

    MPS does not implement ``linalg.svd``; use :func:`pca_power` there.

    Args:
      x_centered: Mean-centered observations of shape ``(N, D)``.

    Returns:
      eigenvalues: Ascending eigenvalues of shape ``(D,)``, flipped from
        SVD's descending order to match the ``eigh`` convention.
      eigenvectors: Columns are the corresponding eigenvectors ``(D, D)``.

    Raises:
      RuntimeError: If the input lives on an MPS device.

    """
    if x_centered.device.type == "mps":
        raise RuntimeError("pca_svd is not supported on MPS; use pca_power instead.")
    _U, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
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

    Only uses matmul and basic arithmetic — runs natively on MPS
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
    d = sigma.shape[0]
    basis = torch.randn(d, d, device=sigma.device, dtype=sigma.dtype)
    basis, _ = _householder_qr(basis)
    prev = (basis * (sigma @ basis)).sum(dim=0)
    for _ in range(num_iters):
        basis, _ = _householder_qr(sigma @ basis)
        if tol > 0:
            eigenvalues = (basis * (sigma @ basis)).sum(dim=0)
            if (eigenvalues - prev).abs().max() < tol:
                prev = eigenvalues
                break
            prev = eigenvalues
    eigenvalues = (basis * (sigma @ basis)).sum(dim=0)
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

    This is a **fit** function — it returns the decomposition, not
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
    eigenvalues, eigenvectors = decompose(x_t - x_t.mean(dim=0, keepdim=True))
    if whiten:
        eigenvectors = eigenvectors * torch.rsqrt(eigenvalues.unsqueeze(0) + eps)
    return eigenvalues, eigenvectors


def _householder_qr(mat: Tensor) -> tuple[Tensor, Tensor]:
    """Householder QR decomposition (MPS-native).

    Returns (q, r) where q is orthogonal and r is upper triangular.
    """
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
        """Record an observation and prune expired entries."""
        self.samples.append((timestamp, cumulative_count))
        cutoff = timestamp - self.window_sec
        self.samples = [(t, c) for t, c in self.samples if t >= cutoff]

    def compute_rate(self, current_time: float, current_count: float) -> float:
        """Compute items/sec over the window with Laplace smoothing."""
        if len(self.samples) < 2:
            return 0.0
        t0, c0 = self.samples[0]
        elapsed = current_time - t0
        items = current_count - c0
        return (items + self.pseudocount) / (elapsed + self.pseudotime)
