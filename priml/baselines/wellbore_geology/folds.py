"""Geographic folds and fold-safe relative-surface priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import math

from configgle import Fig
from numpy.typing import NDArray

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class Clustering:
    """Complete Lloyd result, including stable cluster numbering."""

    labels: NDArray[np.int64]
    centers: NDArray[np.float64]
    iterations: int


def kmeans(
    values: NDArray[np.float64],
    *,
    clusters: int,
    seed: int,
    extra_trials: int,
    tolerance: float,
    max_iterations: int,
) -> Clustering:
    """Cluster unscaled geographic coordinates with greedy distance seeding.

    Args:
      values: Complete centroid rows in input-well order.
      clusters: Number of geographic classes.
      seed: Independent MT19937 initialization seed.
      extra_trials: Seeding trials beyond the natural logarithm of the classes.
      tolerance: Coordinate-variance fraction ending the Lloyd iterations.
      max_iterations: Maximum complete assignment/update rounds.

    Returns:
      result: Complete labels, uncentered centers and iteration count.

    """
    origin = values.mean(axis=0)
    points = values - origin
    result = lloyd(
        points,
        centers=greedy_centers(
            points,
            clusters=clusters,
            seed=seed,
            extra_trials=extra_trials,
        ),
        tolerance=float(np.var(values, axis=0).mean()) * tolerance,
        max_iterations=max_iterations,
    )
    return Clustering(
        labels=result.labels,
        centers=result.centers + origin,
        iterations=result.iterations,
    )


def greedy_centers(
    points: NDArray[np.float64],
    *,
    clusters: int,
    seed: int,
    extra_trials: int,
) -> NDArray[np.float64]:
    """Choose each center by the lowest potential among distance-weighted trials.

    Args:
      points: Mean-centered float64 observations.
      clusters: Number of center rows to initialize.
      seed: Independent MT19937 draw stream.
      extra_trials: Trials beyond the natural logarithm of the classes.

    Returns:
      centers: Ordered initial center coordinates.

    """
    rng = np.random.RandomState(seed)
    centers = np.empty((clusters, points.shape[1]), dtype=np.float64)
    first = rng.choice(len(points), p=np.full(len(points), 1 / len(points)))
    centers[0] = points[first]
    squared_norms = cast("NDArray[np.float64]", np.einsum("ij,ij->i", points, points))
    nearest = cast(
        "NDArray[np.float64]",
        _squared_distances(centers[:1], points, squared_norms=squared_norms)[0],
    )
    potential = float(nearest.sum())
    for cluster in range(1, clusters):
        trial_mass = (
            rng.uniform(size=extra_trials + int(math.log(clusters))) * potential
        )
        candidate = np.searchsorted(np.cumsum(nearest), trial_mass)
        np.clip(candidate, None, len(points) - 1, out=candidate)
        candidate_distance = _squared_distances(
            points[candidate],
            points,
            squared_norms=squared_norms,
        )
        np.minimum(nearest, candidate_distance, out=candidate_distance)
        candidate_potential = candidate_distance.sum(axis=1)
        best = np.argmin(candidate_potential)
        centers[cluster] = points[candidate[best]]
        nearest = cast("NDArray[np.float64]", candidate_distance[best])
        potential = float(candidate_potential[best : best + 1].sum())
    return centers


def lloyd(
    points: NDArray[np.float64],
    *,
    centers: NDArray[np.float64],
    tolerance: float,
    max_iterations: int,
) -> Clustering:
    """Fit centers with stable ties, farthest-point rescue and variance tolerance.

    Args:
      points: Complete mean-centered observations.
      centers: Initial center rows; the input is left unchanged.
      tolerance: Largest accepted sum of squared center shifts.
      max_iterations: Maximum complete assignment/update rounds.

    Returns:
      result: Final assignment, centers, and complete iteration count.

    """
    clusters = len(centers)
    previous: NDArray[np.int64] | None = None
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        labels = _labels(points, centers)
        sums = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=clusters).astype(np.float64)
        np.add.at(sums, labels, points)
        empty = np.flatnonzero(np.equal(counts, 0))
        if empty.size:
            distance = np.square(points - centers[labels]).sum(axis=1)
            if np.max(distance) != 0:
                farthest = np.argsort(-distance, kind="stable")[: len(empty)]
                for offset in range(len(empty)):
                    missing = cast("np.int64", empty[offset])
                    index = cast("np.int64", farthest[offset])
                    donor = cast("np.int64", labels[index])
                    sums[donor] -= points[index]
                    sums[missing] = points[index]
                    counts[donor] -= 1
                    counts[missing] = 1
        largest = np.argmax(counts)
        updated = sums.copy()
        for cluster in range(clusters):
            if counts[cluster] > 0:
                updated[cluster] *= 1 / counts[cluster]
            else:
                updated[cluster] = sums[largest] / counts[largest]
        shift = np.sqrt(np.square(centers - updated).sum(axis=1))
        centers = updated
        if previous is not None and np.array_equal(labels, previous):
            return Clustering(labels=labels, centers=centers, iterations=iteration)
        if np.square(shift).sum() <= tolerance:
            break
        previous = labels.copy()
    return Clustering(
        labels=_labels(points, centers),
        centers=centers,
        iterations=iteration,
    )


def stratified_folds(
    labels: NDArray[np.int64],
    *,
    folds: int,
    seed: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Shuffle each first-appearance class after balanced round-robin allocation.

    Args:
      labels: Complete cluster labels in input-well order.
      folds: Number of disjoint validation partitions.
      seed: Independent MT19937 permutation stream.

    Returns:
      splits: Training and validation indices in ascending row order, per fold.

    """
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    encoded = np.argsort(first).argsort()[inverse]
    ordered = np.sort(encoded)
    classes = len(first)
    allocation = np.asarray(
        [
            np.bincount(ordered[index::folds], minlength=classes)
            for index in range(folds)
        ],
        dtype=np.int64,
    )
    validation_fold = np.empty(len(labels), dtype=np.int64)
    rng = np.random.RandomState(seed)
    for category in range(classes):
        assignments = np.arange(folds, dtype=np.int64).repeat(allocation[:, category])
        rng.shuffle(assignments)
        validation_fold[encoded == category] = assignments
    return [
        (
            np.flatnonzero(np.not_equal(validation_fold, index)),
            np.flatnonzero(np.equal(validation_fold, index)),
        )
        for index in range(folds)
    ]


class GeographicCV:
    """Keep the public repeated five-fold geographic training order."""

    class Config(Fig["GeographicCV"]):
        clusters: int = 20
        """Unscaled geographic KMeans classes."""

        folds: int = 5
        """Validation partitions per repeat."""

        repeats: int = 3
        """Independent consecutive split seeds."""

        cluster_seed: int = 0
        """Independent MT19937 seed for centroid initialization."""

        split_seed: int = 33
        """First shuffle seed; each repeat increments it by one."""

        extra_trials: int = 2
        """Seeding trials beyond the natural logarithm of the clusters."""

        tolerance: float = 1e-4
        """Coordinate-variance fraction ending the Lloyd iterations."""

        max_iterations: int = 300
        """Maximum complete Lloyd assignment/update rounds."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def split(
        self,
        xy: NDArray[np.float64],
    ) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Return every fold in repeat-major order without changing input rows.

        Args:
          xy: Complete unscaled geographic centroid rows in input-well order.

        Returns:
          splits: All repeated training and validation index arrays.

        """
        folds = min(self.config.folds, len(xy))
        if folds < 2:
            return [(np.arange(len(xy)), np.empty(0, dtype=np.int64))]
        clusters = min(self.config.clusters, len(xy))
        if clusters < 2:
            raise ValueError("Geographic CV requires at least two clusters.")
        labels = kmeans(
            xy,
            clusters=clusters,
            seed=self.config.cluster_seed,
            extra_trials=self.config.extra_trials,
            tolerance=self.config.tolerance,
            max_iterations=self.config.max_iterations,
        ).labels
        if np.any(np.bincount(labels, minlength=clusters) < folds):
            raise ValueError("Every geographic cluster needs one member per fold.")
        return [
            split
            for repeat in range(self.config.repeats)
            for split in stratified_folds(
                labels,
                folds=folds,
                seed=self.config.split_seed + repeat,
            )
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class WellTrajectory:
    """Full float32 trajectory with a contiguous hidden suffix and visible anchor."""

    well_id: str
    md: NDArray[np.float32]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    z: NDArray[np.float32]
    known_depth: NDArray[np.float32]
    depth: NDArray[np.float32] | None
    suffix_start: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RelativeSurface:
    """Integrated relative surface and all complete geometric diagnostic rows."""

    values: NDArray[np.float32]
    diagnostics: dict[str, NDArray[np.float32]]


@dataclass(frozen=True, slots=True, kw_only=True)
class SupportPoints:
    """Coordinates, normalized step equations, tangents, and path directions."""

    values: NDArray[np.float32]
    is_prefix: NDArray[np.bool_]


class AnisotropicMetric:
    """Rotate coordinates before scaling the along-axis neighbor distance."""

    class Config(Fig["AnisotropicMetric"]):
        angle_degrees: float = 50.0
        """Rotation angle in degrees."""

        ratio: float = 1.3
        """Length scale along the rotated first axis."""

    def __init__(self, config: Config) -> None:
        self.config = config
        theta = math.radians(config.angle_degrees)
        self.cosine, self.sine = math.cos(theta), math.sin(theta)

    def __call__(self, xy: NDArray[np.float32]) -> NDArray[np.float64]:
        """Transform point-neighbor coordinates in float64 precision."""
        coordinates = np.asarray(xy, dtype=np.float64)
        along = coordinates[:, 0] * self.cosine + coordinates[:, 1] * self.sine
        cross = -coordinates[:, 0] * self.sine + coordinates[:, 1] * self.cosine
        return np.column_stack((along / self.config.ratio, cross))


class RobustPlane:
    """Solve a local gradient with ridge, Tukey reweighting, and tangent fallback."""

    class Config(Fig["RobustPlane"]):
        ridge_fraction: float = 0.01
        """Ridge magnitude relative to the local weighted design trace."""

        min_eigenvalue_ratio: float = 0.05
        """Smallest accepted ratio of design eigenvalues."""

        max_gradient: float = 0.25
        """Largest fitted-plane coordinate gradient; tangent fallback is unbounded."""

        robust_scale: float = 0.1
        """Tukey residual cutoff in normalized surface-step units."""

        iterations: int = 1
        """Number of robust refinements after the initial weighted fit."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        weights: NDArray[np.float64],
        equations: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit every local gradient and refine it with Tukey residual weights.

        Args:
          weights: Per-query inverse-distance masses of every selected neighbor.
          equations: Normalized dx, dy, dS, tangent-x and tangent-y per neighbor.

        Returns:
          gradient: Fitted or fallback x/y surface derivatives for each query.

        """
        weights = weights / weights.sum(axis=1, keepdims=True)
        beta = self._fit(weights, equations)
        for _ in range(self.config.iterations):
            predicted = beta[:, None, 0] * equations[:, :, 0]
            predicted = predicted + beta[:, None, 1] * equations[:, :, 1]
            residual = np.abs(equations[:, :, 2] - predicted) / self.config.robust_scale
            factor = np.zeros_like(residual)
            keep = residual < 1
            factor[keep] = np.square(1 - np.square(residual[keep]))
            robust = weights * factor
            empty = robust.sum(axis=1) <= 1e-300
            robust[empty] = weights[empty]
            robust /= robust.sum(axis=1, keepdims=True)
            beta = self._fit(robust, equations)
        return beta

    def _fit(
        self,
        weights: NDArray[np.float64],
        equations: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights = weights / weights.sum(axis=1, keepdims=True)
        fallback = (equations[:, :, 3:5] * weights[:, :, None]).sum(axis=1)
        dx, dy, ds = (equations[:, :, index] for index in range(3))
        a00 = (weights * dx * dx).sum(axis=1)
        a01 = (weights * dx * dy).sum(axis=1)
        a11 = (weights * dy * dy).sum(axis=1)
        b0 = (weights * dx * ds).sum(axis=1)
        b1 = (weights * dy * ds).sum(axis=1)
        trace = a00 + a11
        ridge = np.maximum(trace * self.config.ridge_fraction, 1e-12)
        m00, m11 = a00 + ridge, a11 + ridge
        determinant = m00 * m11 - a01 * a01
        beta = fallback.copy()
        solvable = (trace > 1e-12) & (np.abs(determinant) > 1e-18)
        beta[solvable, 0] = (
            m11[solvable] * b0[solvable] - a01[solvable] * b1[solvable]
        ) / determinant[solvable]
        beta[solvable, 1] = (
            -a01[solvable] * b0[solvable] + m00[solvable] * b1[solvable]
        ) / determinant[solvable]
        discriminant = np.sqrt(np.maximum((a00 - a11) * (a00 - a11) + 4 * a01 * a01, 0))
        eigen_ratio = (0.5 * (trace - discriminant)) / np.maximum(
            0.5 * (trace + discriminant),
            1e-12,
        )
        bounded = (np.abs(beta[:, 0]) <= self.config.max_gradient) & (
            np.abs(beta[:, 1]) <= self.config.max_gradient
        )
        use_plane = (
            solvable & (eigen_ratio >= self.config.min_eigenvalue_ratio) & bounded
        )
        beta[~use_plane] = fallback[~use_plane]
        return beta


class GeographicPrior:
    """Build a relative surface from fold-local suffix and query-prefix support."""

    class Config(Fig["GeographicPrior"]):
        neighbor_wells: int = 12
        """Closest support-well centroids retained before point selection."""

        point_neighbors: int = 112
        """Closest support points retained for each query bin."""

        train_bin_size: int = 64
        """Original rows grouped into each suffix support point."""

        query_bin_size: int = 32
        """Original suffix rows grouped into each query point."""

        prefix_bin_size: int = 32
        """Original rows grouped into each visible-prefix support point."""

        prefix_support_rows: int = 512
        """Visible tail rows usable as same-well support."""

        surface_jump_limit: float = 0.15
        """Largest accepted absolute first difference of the known surface."""

        max_support_points: int = 12_000
        """Nearest support-point cap after selecting neighboring wells."""

        distance_epsilon: float = 1e-6
        """Minimum distance and coincidence threshold for inverse distance weights."""

        distance_power: float = 2.0
        """Power in the inverse-distance neighbor weight."""

        metric: AnisotropicMetric.Config = field(
            default_factory=AnisotropicMetric.Config,
        )
        """Point-neighbor coordinate geometry."""

        plane: RobustPlane.Config = field(default_factory=RobustPlane.Config)
        """Local robust surface-gradient fit."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.metric = config.metric.make()
        self.plane = config.plane.make()

    def build(
        self,
        support: Sequence[WellTrajectory],
        *,
        queries: Sequence[WellTrajectory],
        leave_one_out: bool,
    ) -> dict[str, RelativeSurface]:
        """Build every query surface without admitting held-out target support.

        Args:
          support: Only target-bearing training-fold wells.
          queries: Complete wells needing feature priors and diagnostics.
          leave_one_out: Exclude each query's own hidden suffix from support.

        Returns:
          prior: Complete relative surface and diagnostic arrays keyed by well.

        """
        points = {well.well_id: self._support(well, prefix=False) for well in support}
        return {
            well.well_id: self._query(
                well,
                support=[
                    other
                    for other in support
                    if not leave_one_out or other.well_id != well.well_id
                ],
                points=points,
            )
            for well in queries
        }

    def _support(self, well: WellTrajectory, *, prefix: bool) -> SupportPoints:
        region = np.zeros(len(well.md), dtype=np.bool_)
        if prefix:
            region[
                max(
                    0,
                    well.suffix_start - self.config.prefix_support_rows,
                ) : well.suffix_start
            ] = True
            surface = well.known_depth + well.z
            bin_size = self.config.prefix_bin_size
        else:
            if well.depth is None:
                raise ValueError("Expected well.depth is not None.")
            region[well.suffix_start :] = True
            surface = well.depth + well.z
            bin_size = self.config.train_bin_size
        ds, dx, dy = (np.full(len(well.md), np.nan, dtype=np.float32) for _ in range(3))
        ds[1:], dx[1:], dy[1:] = np.diff(surface), np.diff(well.x), np.diff(well.y)
        region &= np.abs(ds) <= np.float32(self.config.surface_jump_limit)
        distance_squared = dx * dx + dy * dy
        step = np.sqrt(distance_squared)
        along_x, along_y, gx, gy, normalized_ds = (
            np.full_like(ds, np.nan) for _ in range(5)
        )
        valid_step = distance_squared > 1e-10
        np.divide(dx, step, out=along_x, where=valid_step)
        np.divide(dy, step, out=along_y, where=valid_step)
        np.divide(ds * dx, distance_squared, out=gx, where=valid_step)
        np.divide(ds * dy, distance_squared, out=gy, where=valid_step)
        np.divide(ds, step, out=normalized_ds, where=valid_step)
        valid = (
            region
            & np.isfinite(well.x)
            & np.isfinite(well.y)
            & np.isfinite(surface)
            & np.isfinite(gx)
            & np.isfinite(gy)
            & valid_step
        )
        values: NDArray[np.float32] = np.column_stack(
            (well.x, well.y, along_x, along_y, normalized_ds, gx, gy, along_x, along_y),
        )
        selected = np.flatnonzero(valid)
        bins = selected // bin_size
        if selected.size == 0:
            averaged = np.empty((0, 9), dtype=np.float32)
        else:
            counts = np.bincount(bins).astype(np.float64)
            averaged64 = np.zeros((len(counts), 9), dtype=np.float64)
            for column in range(9):
                averaged64[:, column] = np.bincount(
                    bins,
                    weights=np.take(values[:, column], selected),
                    minlength=len(counts),
                )
            active = counts > 0
            averaged64[active] /= counts[active, None]
            averaged = averaged64[active].astype(np.float32)
        return SupportPoints(
            values=averaged,
            is_prefix=np.full(len(averaged), prefix, dtype=np.bool_),
        )

    def _query(
        self,
        well: WellTrajectory,
        *,
        support: Sequence[WellTrajectory],
        points: dict[str, SupportPoints],
    ) -> RelativeSurface:
        start = well.suffix_start
        anchor = well.known_depth[start - 1 : start] + well.z[start - 1 : start]
        values = np.full(len(well.md), np.float32("nan"))
        values[:start] = well.known_depth[:start] + well.z[:start] - anchor
        diagnostics = {
            name: np.zeros(len(well.md), dtype=np.float32)
            for name in (
                "geo_nbr_std",
                "geo_nbr_dS_std",
                "geo_radial_extrap_score",
                "geo_nbr_distance",
                "geo_nbr_path_alignment",
                "geo_prefix_weight_ratio",
            )
        }
        raw = np.arange(start, len(well.md))
        if len(raw) == 0:
            return RelativeSurface(values=values, diagnostics=diagnostics)
        bins = (raw - start) // self.config.query_bin_size
        counts = np.bincount(bins)
        xy = np.column_stack(
            (
                np.bincount(bins, weights=well.x[raw]) / counts,
                np.bincount(bins, weights=well.y[raw]) / counts,
            ),
        ).astype(np.float32)
        row_position = np.bincount(bins, weights=raw) / counts
        dxy = np.column_stack(
            (
                np.bincount(bins, weights=well.x[raw] - well.x[raw - 1]) / counts,
                np.bincount(bins, weights=well.y[raw] - well.y[raw - 1]) / counts,
            ),
        ).astype(np.float32)
        selected = self._neighbors(well, support=support, points=points, xy=xy)
        gradient, observed = self._predict(selected, xy=xy, dxy=dxy)
        raw_gradient = np.column_stack(
            [np.interp(raw, row_position, gradient[:, column]) for column in range(2)],
        ).astype(np.float32)
        for name, observed_values in observed.items():
            diagnostics[name][raw] = np.interp(raw, row_position, observed_values)
        increment = raw_gradient[:, 0] * (well.x[raw] - well.x[raw - 1])
        increment += raw_gradient[:, 1] * (well.y[raw] - well.y[raw - 1])
        values[raw] = np.cumsum(increment, dtype=np.float64)
        return RelativeSurface(values=values, diagnostics=diagnostics)

    def _neighbors(
        self,
        well: WellTrajectory,
        *,
        support: Sequence[WellTrajectory],
        points: dict[str, SupportPoints],
        xy: NDArray[np.float32],
    ) -> SupportPoints:
        if not support:
            raise ValueError("Geographic inference requires a support well.")
        centers = np.asarray(
            [
                (float(np.nanmean(other.x)), float(np.nanmean(other.y)))
                for other in support
            ],
            dtype=np.float64,
        )
        query_center = np.asarray(
            [(float(np.nanmean(well.x)), float(np.nanmean(well.y)))],
            dtype=np.float64,
        )
        _, nearest = _nearest(
            centers,
            queries=query_center,
            count=self.config.neighbor_wells,
        )
        own = self._support(well, prefix=True)
        parts = [own] if len(own.values) else []
        parts.extend(
            points[support[int(index)].well_id]
            for index in nearest[0]
            if len(points[support[int(index)].well_id].values)
        )
        if not parts:
            raise ValueError("Geographic support contains no valid surface steps.")
        values = np.vstack([part.values for part in parts]).astype(np.float32)
        flags = np.concatenate([part.is_prefix for part in parts])
        if len(values) > self.config.max_support_points:
            distance, _ = _nearest(
                xy.astype(np.float64),
                queries=values[:, :2].astype(np.float64),
                count=1,
            )
            selected = np.sort(
                np.argsort(distance[:, 0], kind="stable")[
                    : self.config.max_support_points
                ],
            )
            values, flags = values[selected], flags[selected]
        return SupportPoints(values=values, is_prefix=flags)

    def _predict(
        self,
        support: SupportPoints,
        *,
        xy: NDArray[np.float32],
        dxy: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        distance, indices = _nearest(
            self.metric(support.values[:, :2]),
            queries=self.metric(xy),
            count=self.config.point_neighbors,
        )
        weights = (
            1
            / np.maximum(distance, self.config.distance_epsilon)
            ** self.config.distance_power
        )
        weights[distance <= self.config.distance_epsilon] = 1e18
        weights /= weights.sum(axis=1, keepdims=True)
        neighbors = support.values[indices].astype(np.float64)
        gradient = self.plane(weights, neighbors[:, :, 2:7]).astype(np.float32)
        coordinate_mean = (neighbors[:, :, :2] * weights[:, :, None]).sum(axis=1)
        offset = xy.astype(np.float64)[:, None, :] - neighbors[:, :, :2]
        neighbor_distance = np.sqrt(
            (weights * np.sum(offset * offset, axis=2)).sum(axis=1),
        )
        centered = neighbors[:, :, :2] - coordinate_mean[:, None, :]
        variance = (weights * np.sum(centered * centered, axis=2)).sum(axis=1)
        standard_deviation = np.sqrt(np.maximum(variance, 0))
        query_offset = xy.astype(np.float64) - coordinate_mean
        radial = np.sqrt(np.sum(query_offset * query_offset, axis=1)) / np.maximum(
            standard_deviation,
            self.config.distance_epsilon,
        )
        query_direction = dxy.astype(np.float64)
        direction_dot = np.sum(
            neighbors[:, :, 7:9] * query_direction[:, None, :],
            axis=2,
        )
        direction_norm = np.linalg.norm(neighbors[:, :, 7:9], axis=2)
        direction_norm *= np.linalg.norm(query_direction, axis=1)[:, None]
        alignment = (
            weights
            * np.clip(
                np.abs(direction_dot)
                / np.maximum(direction_norm, self.config.distance_epsilon),
                0,
                1,
            )
        ).sum(axis=1)
        implied = (
            neighbors[:, :, 5] * dxy[:, None, 0] + neighbors[:, :, 6] * dxy[:, None, 1]
        )
        implied_mean = (implied * weights).sum(axis=1)
        implied_variance = (weights * np.square(implied - implied_mean[:, None])).sum(
            axis=1,
        )
        return gradient, {
            "geo_nbr_std": standard_deviation.astype(np.float32),
            "geo_nbr_dS_std": np.sqrt(np.maximum(implied_variance, 0)).astype(
                np.float32,
            ),
            "geo_radial_extrap_score": radial.astype(np.float32),
            "geo_nbr_distance": neighbor_distance.astype(np.float32),
            "geo_nbr_path_alignment": alignment.astype(np.float32),
            "geo_prefix_weight_ratio": (weights * support.is_prefix[indices])
            .sum(axis=1)
            .astype(np.float32),
        }


def _squared_distances(
    centers: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    squared_norms: NDArray[np.float64],
) -> NDArray[np.float64]:
    distance = -2 * (centers @ points.T)
    norms = cast("NDArray[np.float64]", np.einsum("ij,ij->i", centers, centers))
    distance += norms[:, None]
    distance += squared_norms[None, :]
    return np.maximum(distance, 0)


def _labels(
    points: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> NDArray[np.int64]:
    norms = cast("NDArray[np.float64]", np.einsum("ij,ij->i", centers, centers))
    return np.argmin(
        -2 * (points @ centers.T) + norms[None, :],
        axis=1,
    ).astype(np.int64)


def _nearest(
    points: NDArray[np.float64],
    *,
    queries: NDArray[np.float64],
    count: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Find each query's nearest points by distance, ties to the lower row."""
    if np.count_nonzero(~np.isfinite(np.concatenate((points, queries)))):
        raise ValueError("Geographic neighbor coordinates must be finite.")
    dx = queries[:, None, 0] - points[None, :, 0]
    dy = queries[:, None, 1] - points[None, :, 1]
    squared = dx * dx + dy * dy
    # Equal distances must keep row order; argpartition and the default sort do not.
    indices = np.argsort(squared, axis=1, kind="stable")[:, :count]
    return np.sqrt(np.take_along_axis(squared, indices, axis=1)), indices
