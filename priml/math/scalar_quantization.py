"""Fixed-rate scalar quantization: Lloyd-Max levels and nearest-level coding.

A quantizer is a sorted vector of reconstruction LEVELS; a value is coded as the
index of its nearest level, found against the midpoints between neighbours.
:func:`lloyd_max` fits levels that minimize mean squared error over a sample by
alternating the two Lloyd-Max conditions -- each boundary is the midpoint of
its neighbouring levels, each level is the mean of its cell -- until the
distortion stops falling. It converges to a local optimum, so the
initialization matters; :func:`gaussian_levels` supplies the standard-normal
optimum, which a per-channel mean and standard deviation turn into a start
close to the answer for any unimodal source.

Derivation:
  On SORTED samples ``x`` the cells are contiguous index ranges, so with prefix
  sums ``S[k] = x[0] + ... + x[k-1]`` and ``Q[k]`` of squares, a cell spanning
  ``[a, b)`` has mean ``(S[b] - S[a]) / (b - a)`` and squared error
  ``(Q[b] - Q[a]) - (S[b] - S[a])**2 / (b - a)``. Each iteration is then one
  ``searchsorted`` of the boundaries into ``x`` -- exact Lloyd on the sample,
  with no histogram, at ``O(levels * log n)`` per iteration.

References:
  W. R. Bennett. Spectra of quantized signals. Bell Syst. Tech. J.
    27(3):446-472, 1948.
  P. F. Panter and W. Dite. Quantization distortion in pulse-count
    modulation with nonuniform spacing of levels. Proc. IRE 39(1):44-48, 1951.
  S. P. Lloyd. Least squares quantization in PCM. IEEE Trans. Inf. Theory
    28(2):129-137, 1982.
  J. Max. Quantizing for minimum distortion. IRE Trans. Inf. Theory
    6(1):7-12, 1960.
  R. M. Gray and D. L. Neuhoff. Quantization. IEEE Trans. Inf. Theory
    44(6):2325-2383, 1998.

"""

from __future__ import annotations

from typing import cast

import math

from torch import Tensor

import torch


def lloyd_max(
    values: Tensor,
    init: Tensor,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
) -> Tensor:
    """Fit reconstruction levels minimizing squared error over a sample.

    An empty cell is refilled by splitting the cell with the largest squared
    error at its median, so every returned level holds data. A sample with
    fewer distinct values than levels returns those values, padded by
    repeating the largest.

    Args:
      values: 1-D sample of finite values.
      init: 1-D starting levels, one per output level.
      max_iterations: Upper bound on Lloyd iterations.
      tolerance: Stop once one iteration lowers the distortion by less than
        this fraction of it.

    Returns:
      levels: ``[len(init)]`` float64, non-decreasing; sums run in float64.

    Raises:
      ValueError: ``values`` is empty or not finite, or ``init`` is empty.

    """
    if values.ndim != 1 or init.ndim != 1:
        raise ValueError("lloyd_max expects 1-D values and levels.")
    if values.numel() == 0 or init.numel() == 0:
        raise ValueError("lloyd_max needs a nonempty sample and at least one level.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("lloyd_max needs finite values.")
    x = values.detach().to(torch.float64).sort().values
    num_levels = init.numel()
    distinct = cast("Tensor", torch.unique_consecutive(x))  # pyright: ignore[reportAny] -- torch stubs type unique_consecutive as Any.
    if distinct.numel() <= num_levels:
        pad = distinct[-1].expand(num_levels - distinct.numel())
        return torch.cat([distinct, pad])
    zero = x.new_zeros(1)
    total = torch.cat([zero, x.cumsum(0)])
    total_sq = torch.cat([zero, (x * x).cumsum(0)])
    levels = init.detach().to(torch.float64).sort().values
    previous = _distortion(x, levels, total, total_sq)
    for _ in range(max_iterations):
        bounds = _cell_bounds(x, levels)
        levels = _centroids(levels, bounds, total)
        levels = _split_empty_cells(levels, x, total, total_sq)
        distortion = _distortion(x, levels, total, total_sq)
        # Measured from the INITIAL distortion, not from infinity: ``inf - d <= tol *
        # inf`` holds, which stopped the loop after one step.
        if distortion == 0 or previous - distortion <= tolerance * previous:
            break
        previous = distortion
    return levels


def high_resolution_levels(
    values: Tensor,
    num_levels: int,
    *,
    bins: int | None = None,
) -> Tensor:
    """Place levels at the high-resolution optimum density, ``p(x) ** (1 / 3)``.

    For fixed-rate coding under squared error, the level density minimizing
    distortion as the level count grows is proportional to the cube root of the
    source density (Bennett 1948; Panter & Dite 1951; Gray & Neuhoff 1998,
    Sec. V). Estimated here from a histogram of the sample, it is a start from
    which :func:`lloyd_max` converges in a few hundred iterations where an
    equal-probability start needs tens of thousands -- and a better start also
    lands in a better local optimum (measured on 2**20 Laplace draws: 41.6 dB
    against 40.4 dB after 20,000 iterations from equal-probability levels).

    The histogram is deliberately coarse. The cube root of a SPARSE count is
    biased -- ``E[N ** (1/3)]`` is near ``E[N]`` when most bins hold 0 or 1 --
    so a fine histogram weights the tails like ``p`` rather than ``p ** (1/3)``
    (measured: 65,536 bins over 200,000 normal draws put level 48 of 64 at
    0.99 against the exact 1.21).

    Derivation:
      Levels sit at ``G^-1((k + 1/2) / L)`` for the normalized cumulative
      ``G`` of ``h ** (1 / 3)``, ``h`` the histogram counts. ``G`` starts at 0
      on the first bin's LEFT edge and is interpolated linearly within a bin,
      so a level never collapses onto a bin edge; an empty bin is flat in ``G``
      and receives no level.

    Args:
      values: 1-D finite sample.
      num_levels: Levels to place.
      bins: Histogram bins spanning the sample's range; ``None`` takes
        ``isqrt(n)``, which keeps the expected count per bin near ``sqrt(n)``.

    Returns:
      levels: ``[num_levels]`` float64, non-decreasing.

    Raises:
      ValueError: ``values`` is empty or not finite.

    """
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("high_resolution_levels expects a nonempty 1-D sample.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("high_resolution_levels needs finite values.")
    x = values.detach().to(torch.float64)
    low, high = float(x.min()), float(x.max())
    if low == high:
        return x.new_full((num_levels,), low)
    bins = math.isqrt(x.numel()) if bins is None else bins
    edges = torch.linspace(low, high, bins + 1, dtype=torch.float64)
    counts = torch.histc(x, bins=bins, min=low, max=high)
    density = torch.cat([x.new_zeros(1), (counts ** (1 / 3)).cumsum(0)])
    density = density / density[-1]
    target = (torch.arange(num_levels, dtype=torch.float64) + 0.5) / num_levels
    upper = torch.searchsorted(density, target).clamp(1, bins)
    below, above = density[upper - 1], density[upper]
    fraction = (target - below) / (above - below)
    return edges[upper - 1] + fraction * (edges[upper] - edges[upper - 1])


def gaussian_levels(num_levels: int, *, num_points: int = 1 << 20) -> Tensor:
    """Return the Lloyd-Max levels of the standard normal distribution.

    Fitted on ``num_points`` normal quantiles at equal probability spacing, a
    deterministic stand-in for the density that agrees with Max's (1960)
    tables to their four printed decimals.

    Derivation:
      The cube root of the standard normal density is proportional to a normal
      density of variance 3, so the high-resolution start is exactly
      ``sqrt(3) * ndtri((k + 1/2) / L)``; Lloyd iterations then refine the
      finite-``L`` optimum.

    Args:
      num_levels: Levels in the quantizer.
      num_points: Quantiles standing in for the density.

    Returns:
      levels: ``[num_levels]`` float64, symmetric about zero.

    """
    probability = (torch.arange(num_points, dtype=torch.float64) + 0.5) / num_points
    points = torch.special.ndtri(probability)
    start = 3**0.5 * torch.special.ndtri(
        (torch.arange(num_levels, dtype=torch.float64) + 0.5) / num_levels,
    )
    return lloyd_max(points, start, max_iterations=2_000, tolerance=1e-12)


def midpoints(levels: Tensor) -> Tensor:
    """Return the nearest-level decision boundaries between sorted levels.

    Args:
      levels: ``[..., L]`` sorted levels.

    Returns:
      thresholds: ``[..., L - 1]`` midpoints, in ``levels``' dtype.

    """
    return (levels[..., 1:] + levels[..., :-1]) / 2


def quantize(values: Tensor, thresholds: Tensor) -> Tensor:
    """Return each value's nearest-level index, one threshold row per row.

    A value equal to a threshold takes the lower cell, the convention
    :func:`lloyd_max` fits with. Values past the outer thresholds saturate to
    the first or last level.

    Args:
      values: ``[R, M]`` finite values.
      thresholds: ``[R, L - 1]`` sorted boundaries, one row per row of values.

    Returns:
      indices: ``[R, M]`` int64 in ``[0, L - 1]``.

    Raises:
      ValueError: A value is not finite.

    """
    if not bool(torch.isfinite(values).all()):
        raise ValueError("quantize needs finite values.")
    return torch.searchsorted(thresholds.contiguous(), values.contiguous())


def dequantize(indices: Tensor, levels: Tensor) -> Tensor:
    """Return the level each index names, one level row per row.

    Args:
      indices: ``[R, M]`` integer indices.
      levels: ``[R, L]`` levels.

    Returns:
      values: ``[R, M]`` in ``levels``' dtype.

    """
    return torch.gather(levels, 1, indices.long())


# Cell k holds the samples at or below boundary k and above boundary k-1, matching
# :func:`quantize`'s lower-cell tie rule; ``right=True`` counts ties into the lower
# cell.
def _cell_bounds(x: Tensor, levels: Tensor) -> Tensor:
    """Return the sample index each cell starts at, plus the end."""
    cut = torch.searchsorted(x, midpoints(levels), right=True)
    ends = x.new_tensor([x.numel()], dtype=torch.int64)
    return torch.cat([cut.new_zeros(1), cut, ends])


def _centroids(levels: Tensor, bounds: Tensor, total: Tensor) -> Tensor:
    """Move each nonempty cell's level to its mean; an empty one keeps its level."""
    counts = bounds[1:] - bounds[:-1]
    sums = total[bounds[1:]] - total[bounds[:-1]]
    means = sums / counts.clamp(min=1).to(torch.float64)
    return torch.where(counts > 0, means, levels)


# A level whose cell is empty codes nothing, so it is moved where it earns the most:
# into the cell with the largest squared error, split at its median.
def _split_empty_cells(
    levels: Tensor,
    x: Tensor,
    total: Tensor,
    total_sq: Tensor,
) -> Tensor:
    """Re-seat every empty cell's level inside the worst cell, then re-sort."""
    levels = levels.clone()
    for _ in range(levels.numel()):
        bounds = _cell_bounds(x, levels)
        counts = bounds[1:] - bounds[:-1]
        empty = (counts == 0).nonzero()
        if empty.numel() == 0:
            return levels
        error = _cell_errors(bounds, total, total_sq)
        error = torch.where(counts > 1, error, error.new_tensor(-1.0))
        worst = int(error.argmax())
        if error[worst] <= 0:
            return levels
        start, end = int(bounds[worst]), int(bounds[worst + 1])
        middle = (start + end) // 2
        lower = (total[middle] - total[start]) / (middle - start)
        upper = (total[end] - total[middle]) / (end - middle)
        levels[worst] = lower
        levels[int(empty[0])] = upper
        levels = levels.sort().values
    return levels


def _cell_errors(bounds: Tensor, total: Tensor, total_sq: Tensor) -> Tensor:
    """Return each cell's squared error about its own mean."""
    counts = (bounds[1:] - bounds[:-1]).to(torch.float64)
    sums = total[bounds[1:]] - total[bounds[:-1]]
    squares = total_sq[bounds[1:]] - total_sq[bounds[:-1]]
    return squares - sums * sums / counts.clamp(min=1)


def _distortion(x: Tensor, levels: Tensor, total: Tensor, total_sq: Tensor) -> float:
    """Return the mean squared error of coding ``x`` to its nearest level."""
    bounds = _cell_bounds(x, levels)
    counts = (bounds[1:] - bounds[:-1]).to(torch.float64)
    sums = total[bounds[1:]] - total[bounds[:-1]]
    squares = total_sq[bounds[1:]] - total_sq[bounds[:-1]]
    error = squares - 2 * levels * sums + counts * levels * levels
    return float(error.sum()) / x.numel()
