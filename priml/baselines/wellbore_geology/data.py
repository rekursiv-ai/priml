"""Well observations, transformations and runtime data loading."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import (
    Final,
    Protocol,
    cast,
    overload,
    override,
    runtime_checkable,
)

import csv
import math
import random
import re
import statistics

from configgle import Fig, InlineConfig, Makeable
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.worker import WorkerInfo, get_worker_info
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.wellbore_geology.folds import (
    GeographicPrior,
    RelativeSurface,
    WellTrajectory,
)
from priml.data.na_values import PANDAS_NA_VALUES
from priml.math.fused import fma64


PARTICLE_PLANES: Final = (
    "pf_prob",
    "pf_prob_ffbsi",
    "pf_particle_density_prob",
    "pf_tvt",
    "pf_tvt_ffbsi",
)
"""Every image plane the particle priors produce, in no declared order."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WellRecord:
    """Complete loaded numeric paths, paired curve and optional particle prior."""

    well_id: str
    horizontal: Mapping[str, NDArray[np.float32] | NDArray[np.float64]]
    reference_depth: NDArray[np.float32]
    reference_gamma: NDArray[np.float32]
    particle: Mapping[str, NDArray[np.generic]] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureItem:
    """Every complete model array, auxiliary target, and original-row observation."""

    arrays: dict[str, NDArray[np.generic]]
    auxiliary: dict[str, NDArray[np.float32]]
    metadata: dict[str, object]


class ParticleFeatures:
    """Declare the particle-prior planes that enter the posterior image."""

    class Config(Fig["ParticleFeatures"]):
        path: str = "pf_tvt_rel_pred"
        """Particle cache path published as the PF_TVT column of every prediction."""


class ImageFeatures:
    """Build the complete fixed-window posterior image without mutating inputs."""

    class Config(Fig["ImageFeatures"]):
        prefix_rows: int = 1024
        """Visible rows right-aligned before the prediction boundary."""

        target_rows: int = 10_000
        """Maximum original suffix rows occupying the prediction canvas."""

        downsample: int = 32
        """Original rows aggregated into each horizontal image bin."""

        reference_columns: int = 400
        """Posterior depth candidates in the local paired-curve window."""

        reference_window: float = 100.0
        """Physical depth radius on either side of the final visible anchor."""

        channels: list[str] = field(
            default_factory=lambda: [
                "tw_gr",
                "gr",
                "gr_abs_diff",
                "gr_isnan_rate",
                "gr_std",
                "gr_first_last_delta",
                "gr_slope",
                "gr_quadratic_a",
                "gr_quadratic_b",
                "gr_quadratic_c",
                "gr_quadratic_rmse",
                "tw_gr_is_nan",
                "tw_tvt_rel",
                "seen_tvt_rel",
                "tw_seen_tvt_abs_diff",
                "z_diff",
            ],
        )
        """Ordered active feature planes; each family declares its complete list."""

        particle: ParticleFeatures.Config | None = None
        """Particle-prior planes in the image; None when no channel comes from them."""

        gamma_stats: tuple[float, float] = (87.43, 21.09)
        """Horizontal gamma mean and scale."""

        reference_gamma_stats: tuple[float, float] = (78.62, 28.81)
        """Paired-curve gamma mean and scale."""

        seen_depth_stats: tuple[float, float] = (-18.08, 48.21)
        """Visible anchor-relative depth mean and scale."""

        target_stats: tuple[float, float] = (1.591691, 15.853182)
        """Regression target mean and scale."""

        vertical_step_stats: tuple[float, float] = (-0.016892, 0.1065)
        """Observed vertical-coordinate first-difference mean and scale."""

        surface_step_stats: tuple[float, float] = (0.005535, 0.037389)
        """Geographic surface first-difference mean and scale."""

        depth_axis_scale: float = 57.73
        """Scale for paired-curve depth and absolute candidate differences."""

        alignment_sigma: float = 1.25
        """Physical standard deviation of dense Gaussian alignment targets."""

        gamma_penalty_clip: float = 25.0
        """Largest normalized squared candidate gamma mismatch."""

        feature_clip: float = 8.0
        """Absolute normalized slope and quadratic feature bound."""

        depth_difference_clip: float = 100.0
        """Largest physical absolute candidate depth mismatch."""

        quadratic_stats: tuple[tuple[float, float], ...] = (
            (0.000585828, 0.38249254),
            (0.001914565, 0.338615701),
            (-0.006968186, 1.025529919),
            (0.212819023, 0.131140903),
        )
        """Production downsample-32 coefficient and residual mean/scale pairs."""

    def __init__(self, config: Config) -> None:
        if config.particle is None and set(config.channels).intersection(
            PARTICLE_PLANES,
        ):
            raise ValueError("Particle image channels require the particle features.")
        self.config = config

    def __call__(self, well: WellRecord, *, training: bool) -> FeatureItem:
        """Construct complete image, objective arrays and original-row metadata.

        Args:
          well: Complete row-preserving observations and paired curve.
          training: Normalize targets and retain all suffix truth observations.

        Returns:
          item: Complete image planes, objective arrays, masks and row metadata.

        """
        window = self._window(well)
        cfg = self.config
        bins = len(window.valid) // cfg.downsample + 1
        raw = window.raw
        suffix_mask = (
            _finish(
                _bin_mean(
                    window.suffix.astype(np.float32),
                    np.ones_like(window.valid),
                    cfg.downsample,
                    bins,
                ),
            )
            > 0
        )
        target_depth = _bin_mean(raw["truth"], window.suffix, cfg.downsample, bins)
        target = _bin_mean(raw["truth_rel"], window.suffix, cfg.downsample, bins)
        target_mask = (
            suffix_mask & np.isfinite(target) if training else suffix_mask.copy()
        )
        if training:
            target = (target - cfg.target_stats[0]) / cfg.target_stats[1]
        row_mask = (
            _finish(
                _bin_mean(
                    window.valid.astype(np.float32),
                    np.ones_like(window.valid),
                    cfg.downsample,
                    bins,
                ),
            )
            > 0
        )
        z_relative = _finish(_bin_mean(raw["z"], window.suffix, cfg.downsample, bins))
        z_step = _finish(
            _bin_mean(raw["suffix_z_step"], window.suffix, cfg.downsample, bins),
        )
        surface = _finish(_bin_mean(raw["geo"], window.suffix, cfg.downsample, bins))
        gamma = _bin_mean(raw["gamma"], window.valid, cfg.downsample, bins)
        nan_rate = _bin_mean(
            (~np.isfinite(raw["gamma"])).astype(np.float32),
            window.valid,
            cfg.downsample,
            bins,
        )
        nan_rate[~np.isfinite(nan_rate)] = 1
        seen = _bin_mean(raw["seen"], window.valid, cfg.downsample, bins)
        axis = np.linspace(
            -cfg.reference_window + cfg.reference_window / cfg.reference_columns,
            cfg.reference_window - cfg.reference_window / cfg.reference_columns,
            cfg.reference_columns,
            dtype=np.float32,
        )
        reference = np.full(cfg.reference_columns, np.nan, dtype=np.float32)
        candidate_depth = window.depth_anchor + axis
        lower = cast("np.float32", well.reference_depth[0])
        upper = cast("np.float32", well.reference_depth[-1])
        supported = (candidate_depth >= lower) & (candidate_depth <= upper)
        reference[supported] = np.interp(
            candidate_depth[supported],
            well.reference_depth,
            well.reference_gamma,
        ).astype(np.float32)
        planes = self._planes(
            window,
            gamma=gamma,
            nan_rate=nan_rate,
            reference=reference,
            axis=axis,
            seen=seen,
        )
        geo_depth = surface - z_relative
        geo_step = np.zeros(bins, dtype=np.float32)
        active = np.flatnonzero(suffix_mask & np.isfinite(surface))
        if len(active) >= 2:
            geo_step[active[1:]] = np.diff(surface[active]) / (
                np.maximum(np.diff(active), 1).astype(np.float32)
                * np.float32(cfg.downsample)
            )
        geo_difference = (
            geo_step - z_step - cfg.surface_step_stats[0]
        ) / cfg.surface_step_stats[1]
        geo_difference[~suffix_mask] = 0
        planes.update(
            geo_tvt_diff=np.broadcast_to(
                _finish(geo_difference)[:, None],
                (bins, len(axis)),
            ),
            geo_tvt_rel=np.broadcast_to(
                _finish((geo_depth - cfg.target_stats[0]) / cfg.target_stats[1])[
                    :,
                    None,
                ],
                (bins, len(axis)),
            ),
            geo_tvt_abs_diff=_finish(
                np.minimum(
                    np.abs(axis[None, :] - geo_depth[:, None]),
                    cfg.depth_difference_clip,
                )
                / cfg.depth_axis_scale,
            ),
        )
        metadata: dict[str, object] = {
            "well_id": well.well_id,
            "submit_indices": np.arange(
                window.start,
                len(well.horizontal["MD"]),
                dtype=np.int64,
            ),
            "last_seen_idx": window.start - 1,
            "tvt0": window.depth_anchor,
            "z0": window.z_anchor,
            "suffix_z": well.horizontal["Z"][window.start :].copy(),
            "suffix_len": len(well.horizontal["MD"]) - window.start,
            "kept_suffix_len": window.kept,
            "geo_prior_TVT": (
                well.horizontal["geo_s_rel_prior_abs"][window.start :]
                - well.horizontal["Z"][window.start :]
            ).copy(),
        }
        if training:
            metadata["TVT"] = well.horizontal["TVT"][window.start :]
        if cfg.particle is not None:
            particle = self._particle(
                well,
                window=window,
                seen=seen,
                target_depth=target_depth,
                suffix=suffix_mask,
                training=training,
                axis=axis,
            )
            for name in ("pf_prob", "pf_prob_ffbsi", "pf_particle_density_prob"):
                planes[name] = probability_image(particle[name], columns=len(axis))
            for name, key in (
                ("pf_tvt", "pf_tvt_rel_pred"),
                ("pf_tvt_ffbsi", "pf_tvt_rel_pred_ffbsi"),
            ):
                values = (particle[key] - cfg.target_stats[0]) / cfg.target_stats[1]
                valid = (
                    particle[
                        "pf_prob_ffbsi" if name.endswith("ffbsi") else "pf_prob"
                    ].sum(axis=1)
                    > 0
                )
                values[~valid] = 0
                planes[name] = np.broadcast_to(values[:, None], (bins, len(axis)))
            metadata["pf_tvt_rel_pred"] = particle[cfg.particle.path].copy()
        probabilities = (
            np.zeros((bins, len(axis)), dtype=np.float32)
            if training
            else np.zeros(0, dtype=np.float32)
        )
        if training:
            relative = (
                target_depth[target_mask].astype(np.float64) - window.depth_anchor
            )
            relative = np.clip(
                relative,
                cast("np.float32", axis[0]),
                cast("np.float32", axis[-1]),
            )
            delta = axis.astype(np.float64)[None, :] - relative[:, None]
            masses = np.exp(-0.5 * np.square(delta / cfg.alignment_sigma))
            masses /= masses.sum(axis=1, keepdims=True)
            probabilities[target_mask] = masses.astype(np.float32)
        penalty = np.square(
            ((reference[None, :] - gamma[:, None]) / cfg.gamma_stats[1]).astype(
                np.float32,
            ),
        )
        penalty = np.minimum(penalty, np.float32(cfg.gamma_penalty_clip))
        valid_gamma = np.isfinite(gamma) & (nan_rate < 1)
        penalty[~valid_gamma] = 0
        penalty[valid_gamma[:, None] & ~np.isfinite(reference)[None, :]] = np.float32(
            cfg.gamma_penalty_clip,
        )
        return FeatureItem(
            arrays={
                "geo_s_rel_prior": surface,
                "unet_static": np.stack([planes[name] for name in cfg.channels]).astype(
                    np.float32,
                ),
                "typewell_aux": np.stack(
                    (_finish(reference), (~np.isfinite(reference)).astype(np.float32)),
                ),
                "z_rel": z_relative,
                "bin_count": np.bincount(
                    np.flatnonzero(window.suffix) // cfg.downsample,
                    minlength=bins,
                ).astype(np.float32),
                "z_diff": z_step,
                "target": _finish(target),
                "typewell_target_probs": probabilities,
                "target_mask": target_mask,
                "row_mask": row_mask,
                "suffix_mask": suffix_mask,
            },
            auxiliary={"GR_penalty_error": _finish(penalty)},
            metadata=metadata,
        )

    def _window(self, well: WellRecord) -> _Window:
        cfg = self.config
        horizontal = well.horizontal
        known = np.flatnonzero(np.isfinite(horizontal["TVT_input"]))
        if len(known) == 0:
            raise ValueError("A well requires a visible depth anchor.")
        start = int(known.max()) + 1
        depth_anchor = float(horizontal["TVT_input"][start - 1 : start].sum())
        z_anchor = float(horizontal["Z"][start - 1 : start].sum())
        md_anchor = float(horizontal["MD"][start - 1 : start].sum())
        truth = horizontal.get(
            "TVT",
            np.full(len(horizontal["MD"]), np.nan, dtype=np.float32),
        )
        z_step = np.full_like(truth, np.nan)
        z_step[start:] = np.diff(horizontal["Z"])[start - 1 :]
        sources = {
            "seen": horizontal["TVT_input"] - depth_anchor,
            "gamma": horizontal["GR"],
            "z": horizontal["Z"] - z_anchor,
            "md": horizontal["MD"] - md_anchor,
            "index": horizontal.get(
                "orig_index",
                np.arange(len(truth), dtype=np.float32),
            ),
            "truth": truth,
            "truth_rel": truth - depth_anchor,
            "geo": horizontal["geo_s_rel_prior_abs"]
            - np.float32(depth_anchor + z_anchor),
            "suffix_z_step": z_step,
        }
        left = max(0, start - cfg.prefix_rows)
        kept = min(len(truth) - start, cfg.target_rows)
        destination = cfg.prefix_rows - (start - left)
        size = start - left + kept
        raw = {
            name: np.full(cfg.prefix_rows + cfg.target_rows, np.nan, dtype=np.float32)
            for name in sources
        }
        for name, values in sources.items():
            raw[name][destination : destination + size] = values[left : start + kept]
        valid = np.zeros(cfg.prefix_rows + cfg.target_rows, dtype=np.bool_)
        valid[destination : destination + size] = True
        suffix = np.zeros_like(valid)
        suffix[cfg.prefix_rows : cfg.prefix_rows + kept] = True
        return _Window(
            raw=raw,
            valid=valid,
            suffix=suffix,
            start=start,
            kept=kept,
            depth_anchor=depth_anchor,
            z_anchor=z_anchor,
        )

    def _planes(
        self,
        window: _Window,
        *,
        gamma: NDArray[np.float32],
        nan_rate: NDArray[np.float32],
        reference: NDArray[np.float32],
        axis: NDArray[np.float32],
        seen: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        cfg = self.config
        bins = len(gamma)
        shape = (bins, len(axis))
        raw_gamma = window.raw["gamma"]
        finite = window.valid & np.isfinite(raw_gamma)
        index = np.arange(len(finite), dtype=np.int64) // cfg.downsample
        values = raw_gamma[finite].astype(np.float64)
        selected = index[finite]
        counts = np.bincount(selected, minlength=bins)
        sums = np.bincount(selected, weights=values, minlength=bins)
        square_sums = np.bincount(selected, weights=values * values, minlength=bins)
        mean = np.zeros(bins, dtype=np.float64)
        mean_square = np.zeros_like(mean)
        active = counts > 0
        mean[active], mean_square[active] = (
            sums[active] / counts[active],
            square_sums[active] / counts[active],
        )
        deviation = np.full(bins, np.nan, dtype=np.float32)
        deviation[active] = np.sqrt(
            np.maximum(mean_square[active] - mean[active] * mean[active], 0),
        ).astype(np.float32)
        unique, first, count = np.unique(
            selected,
            return_index=True,
            return_counts=True,
        )
        observed = raw_gamma[finite]
        difference = np.full(bins, np.nan, dtype=np.float32)
        difference[unique] = observed[first + count - 1] - observed[first]
        x = window.raw["md"][finite].astype(np.float64)
        sx = np.bincount(selected, weights=x, minlength=bins)
        sxx = np.bincount(selected, weights=x * x, minlength=bins)
        sxy = np.bincount(selected, weights=x * values, minlength=bins)
        denominator = counts * sxx - sx * sx
        eligible = (counts >= 2) & (np.abs(denominator) > 1e-12)
        slope = np.full(bins, np.nan, dtype=np.float32)
        slope[eligible] = (
            (counts[eligible] * sxy[eligible] - sx[eligible] * sums[eligible])
            / denominator[eligible]
        ).astype(np.float32)
        vertical_step = np.full_like(window.raw["z"], np.nan)
        vertical_step[1:] = np.diff(window.raw["z"])
        scalars = {
            "gr": _finish((gamma - cfg.gamma_stats[0]) / cfg.gamma_stats[1]),
            "gr_isnan_rate": nan_rate,
            "gr_std": _finish(deviation / cfg.gamma_stats[1]),
            "gr_first_last_delta": _finish(difference / cfg.gamma_stats[1]),
            "gr_slope": _finish(
                np.clip(
                    slope * float(cfg.downsample) / cfg.gamma_stats[1],
                    -cfg.feature_clip,
                    cfg.feature_clip,
                ),
            ),
            "seen_tvt_rel": _finish(
                (_finish(seen) - cfg.seen_depth_stats[0]) / cfg.seen_depth_stats[1],
            ),
            "z_diff": _finish(
                (
                    _bin_mean(vertical_step, window.valid, cfg.downsample, bins)
                    - cfg.vertical_step_stats[0]
                )
                / cfg.vertical_step_stats[1],
            ),
        }
        quadratic = _quadratic(raw_gamma, finite, downsample=cfg.downsample, bins=bins)
        for index, name in enumerate(
            ("gr_quadratic_a", "gr_quadratic_b", "gr_quadratic_c", "gr_quadratic_rmse"),
        ):
            values = quadratic[:, index]
            if index == 2:
                values = values - np.float32(cfg.gamma_stats[0])
            values = values / np.float32(cfg.gamma_stats[1])
            mean_stat, scale = cfg.quadratic_stats[index]
            scalars[name] = _finish(
                np.clip(
                    (values - mean_stat) / scale,
                    -cfg.feature_clip,
                    cfg.feature_clip,
                ),
            )
        planes = {
            name: np.broadcast_to(values[:, None], shape)
            for name, values in scalars.items()
        }
        planes["tw_gr"] = np.broadcast_to(
            _finish(
                (reference - cfg.reference_gamma_stats[0])
                / cfg.reference_gamma_stats[1],
            )[None, :],
            shape,
        )
        planes["tw_gr_is_nan"] = np.broadcast_to(
            (~np.isfinite(reference)).astype(np.float32)[None, :],
            shape,
        )
        planes["tw_tvt_rel"] = np.broadcast_to(
            (axis / cfg.depth_axis_scale)[None, :],
            shape,
        )
        absolute_gamma = np.abs(
            (gamma[:, None] - reference[None, :]) / np.float32(cfg.gamma_stats[1]),
        )
        absolute_gamma[nan_rate >= 1] = 0
        absolute_gamma[:, ~np.isfinite(reference)] = 0
        planes["gr_abs_diff"] = _finish(absolute_gamma)
        planes["tw_seen_tvt_abs_diff"] = _finish(
            np.minimum(
                np.abs(axis[None, :] - _finish(seen)[:, None]),
                cfg.depth_difference_clip,
            )
            / cfg.depth_axis_scale,
        )
        return planes

    def _particle(
        self,
        well: WellRecord,
        *,
        window: _Window,
        seen: NDArray[np.float32],
        target_depth: NDArray[np.float32],
        suffix: NDArray[np.bool_],
        training: bool,
        axis: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        if well.particle is None:
            raise ValueError(
                "Particle image channels require a complete particle prior.",
            )
        cache = {
            name: values.astype(np.float32, copy=False)
            for name, values in well.particle.items()
        }
        bins = len(seen)
        positions = _bin_mean(
            window.raw["index"],
            window.valid,
            self.config.downsample,
            bins,
        )
        cached_rows = cache["window_orig_index"]
        valid = np.isfinite(cached_rows) & (cached_rows >= 0)
        source = np.full(bins, np.nan, dtype=np.float32)
        if np.count_nonzero(valid) >= 2:
            source[:] = np.interp(
                positions,
                cached_rows[valid],
                np.arange(len(cached_rows), dtype=np.float32)[valid],
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)
        original_depth = _vector_at(cache["window_tvt_rel"], source, default=0)
        current = np.zeros(bins, dtype=np.float32)
        has_depth = np.isfinite(seen)
        current[has_depth] = seen[has_depth]
        if training:
            selected = suffix & np.isfinite(target_depth)
            current[selected] = (target_depth - np.float32(window.depth_anchor))[
                selected
            ]
            has_depth |= selected
        cached_depth = _vector_at(cache["window_has_tvt"], source, default=0) > 0.5
        delta = np.zeros(bins, dtype=np.float32)
        selected = has_depth & cached_depth
        delta[selected] = current[selected] - original_depth[selected]
        result: dict[str, NDArray[np.float32]] = {}
        for name in ("pf_prob", "pf_prob_ffbsi", "pf_particle_density_prob"):
            probability = _image_at(cache[name], source)
            probability = shift_probability_axis(probability, delta=delta, axis=axis)
            result[name] = probability
        for probability_name, depth_name in (
            ("pf_prob", "pf_tvt_rel_pred"),
            ("pf_prob_ffbsi", "pf_tvt_rel_pred_ffbsi"),
        ):
            values = _vector_at(cache[depth_name], source, default=np.nan) + delta
            invalid = (
                ~np.isfinite(source)
                | ~np.isfinite(values)
                | (result[probability_name].sum(axis=1) <= 0)
            )
            result[probability_name][invalid] = 0
            values[invalid] = 0
            result[depth_name] = values
        return result


def image_probability(
    values: NDArray[np.float32],
    *,
    columns: int,
) -> NDArray[np.float32]:
    """Decode complete binary32 posterior-image values with binary64 expm1 rounded once.

    Args:
      values: Complete nonnegative encoded image in its original logical shape.
      columns: Candidate count used to encode the probability image.

    Returns:
      probability: Binary32 decoded mass in the original logical shape.

    """
    return np.expm1(values.astype(np.float64)).astype(np.float32) / np.float32(
        columns,
    )


def probability_image(
    values: NDArray[np.float32],
    *,
    columns: int,
) -> NDArray[np.float32]:
    """Encode complete posterior mass with binary64 log1p rounded once.

    Args:
      values: Complete posterior probabilities in their original logical shape.
      columns: Candidate count scaling each probability before its logarithm.

    Returns:
      image: Binary32 log1p of each scaled probability, preserving shape.

    """
    return np.log1p((values * np.float32(columns)).astype(np.float64)).astype(
        np.float32,
    )


def normalize_probability(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize the complete float32 posterior after dropping invalid mass."""
    result = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    result[result < 0] = 0
    total = result.sum(axis=1, keepdims=True)
    np.divide(result, total, out=result, where=total > 0)
    return result


def shift_probability_axis(
    values: NDArray[np.float32],
    *,
    delta: NDArray[np.float32],
    axis: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Translate complete posterior rows on a physical axis and normalize mass.

    Args:
      values: Complete float32 row-by-candidate posterior probabilities.
      delta: Physical depth translation for every row.
      axis: Complete equally spaced candidate depths.

    Returns:
      probability: Complete translated, nonnegative and normalized row masses.

    """
    origin = cast("np.float32", axis[0])
    step = cast("np.float32", axis[1]) - origin
    positions = cast(
        "NDArray[np.float32]",
        (axis[None, :] - delta[:, None] - origin) / step,
    )
    left = np.floor(positions).astype(np.int64)
    right = left + 1
    fraction = positions - left.astype(np.float32)
    row = np.arange(len(values), dtype=np.int64)[:, None]
    result = np.zeros_like(values)
    contributions: tuple[tuple[NDArray[np.int64], NDArray[np.float32]], ...] = (
        (left, 1 - fraction),
        (right, fraction),
    )
    for index, weight in contributions:
        valid = np.greater_equal(index, 0) & np.less(index, len(axis))
        result += np.where(
            valid,
            values[row, np.clip(index, 0, len(axis) - 1)] * weight,
            0,
        )
    result = np.maximum(_finish(result), 0)
    totals = result.sum(axis=1, keepdims=True)
    np.divide(result, totals, out=result, where=totals > 0)
    return cast("NDArray[np.float32]", result)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Window:
    raw: dict[str, NDArray[np.float32]]
    valid: NDArray[np.bool_]
    suffix: NDArray[np.bool_]
    start: int
    kept: int
    depth_anchor: float
    z_anchor: float


def _finish(values: NDArray[np.generic]) -> NDArray[np.float32]:
    return np.nan_to_num(values, posinf=0, neginf=0).astype(np.float32)


def _bin_mean(
    values: NDArray[np.float32],
    valid: NDArray[np.bool_],
    downsample: int,
    bins: int,
) -> NDArray[np.float32]:
    selected = valid & np.isfinite(values)
    rows = np.flatnonzero(selected) // downsample
    counts = np.bincount(rows, minlength=bins)
    sums = np.bincount(rows, weights=values[selected], minlength=bins)
    result = np.full(bins, np.nan, dtype=np.float32)
    active = counts > 0
    result[active] = (sums[active] / counts[active]).astype(np.float32)
    return result


def _quadratic(
    values: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    downsample: int,
    bins: int,
) -> NDArray[np.float32]:
    result = np.full((bins, 4), np.nan, dtype=np.float32)
    positions = np.flatnonzero(valid & np.isfinite(values))
    groups, first, counts = np.unique(
        positions // downsample,
        return_index=True,
        return_counts=True,
    )
    for position in range(len(groups)):
        group = cast("np.int64", groups[position])
        offset = cast("np.int64", first[position])
        count = cast("np.int64", counts[position])
        if count < 6:
            continue
        rows = positions[offset : offset + count]
        within = (rows % downsample).astype(np.float64)
        center = float((within[:1] + within[-1:]).sum()) / 2
        half_span = float((within[-1:] - within[:1]).sum()) / 2
        x = (within - center) / half_span
        y = values[rows].astype(np.float64)
        square = x * x
        moments = np.asarray(
            [
                np.bincount(np.zeros(len(x), dtype=np.int64), weights=power)[0]
                for power in (np.ones_like(x), x, square, square * x, square * square)
            ],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [
                np.bincount(np.zeros(len(x), dtype=np.int64), weights=product)[0]
                for product in (square * y, x * y, y)
            ],
            dtype=np.float64,
        )
        normal = moments[np.asarray([[4, 3, 2], [3, 2, 1], [2, 1, 0]], dtype=np.int64)]
        coefficient = _quadratic_solve(normal, rhs)
        residual = max(
            float(
                np.bincount(np.zeros(len(y), dtype=np.int64), weights=y * y).sum()
                - np.sum(coefficient * rhs),
            ),
            0,
        )
        result[group, :3] = coefficient.astype(np.float32)
        root = cast("np.float64", np.sqrt(np.float64(residual / int(count))))
        result[group, 3] = np.float32(root)
    return result


def _quadratic_solve(
    normal: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the full-rank three-moment system in the reference rounding order."""
    matrix = cast("list[list[float]]", normal.tolist())
    result = cast("list[float]", rhs.tolist())
    for column in range(3):
        # Left-looking LU: upper entries use dot products; lower updates fuse.
        for row in range(1, column):
            matrix[row][column] -= sum(
                matrix[row][index] * matrix[index][column] for index in range(row)
            )
        if column:
            for row in range(column, 3):
                total = 0.0
                for index in range(column):
                    total = fma64(
                        matrix[row][index],
                        y=matrix[index][column],
                        addend=total,
                    )
                matrix[row][column] -= total
        pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        result[column], result[pivot] = result[pivot], result[column]
        reciprocal = 1 / matrix[column][column]
        for row in range(column + 1, 3):
            matrix[row][column] *= reciprocal
    for column in range(2):
        for row in range(column + 1, 3):
            result[row] = fma64(
                -result[column],
                y=matrix[row][column],
                addend=result[row],
            )
    for column in range(2, -1, -1):
        result[column] /= matrix[column][column]
        for row in range(column):
            result[row] = fma64(
                -result[column],
                y=matrix[row][column],
                addend=result[row],
            )
    return np.asarray(result, dtype=np.float64)


def _vector_at(
    values: NDArray[np.float32],
    positions: NDArray[np.float32],
    *,
    default: float,
) -> NDArray[np.float32]:
    result = np.full(len(positions), default, dtype=np.float32)
    valid = np.isfinite(positions) & (positions >= 0) & (positions <= len(values) - 1)
    result[valid] = np.interp(
        positions[valid],
        np.arange(len(values), dtype=np.float32),
        values,
    ).astype(np.float32)
    return result


def _image_at(
    values: NDArray[np.float32],
    positions: NDArray[np.float32],
) -> NDArray[np.float32]:
    result = np.zeros((len(positions), values.shape[1]), dtype=np.float32)
    valid = (positions >= 0) & (positions <= len(values) - 1)
    left = np.floor(positions[valid]).astype(np.int64)
    right = np.minimum(left + 1, len(values) - 1)
    fraction = (positions[valid] - left.astype(np.float32))[:, None]
    result[valid] = values[left] * (1 - fraction) + values[right] * fraction
    return result


class DerivativeBlock:
    """One candidate empirical derivative block length and its sampling mass."""

    class Config(Fig["DerivativeBlock"]):
        length: int = 1024
        """Rows of the sampled derivative block."""

        weight: float = 1.0
        """Relative sampling probability of this block length."""


class BootstrapTrend:
    """Integrate sampled empirical derivative blocks with short blended joins."""

    class Config(Fig["BootstrapTrend"]):
        minimum_suffix_rows: int = 1000
        """Minimum original target rows for a training donor."""

        maximum_step: float = 0.15
        """Exclusive absolute derivative bound for the empirical pool."""

        blocks: list[DerivativeBlock.Config] = field(
            default_factory=lambda: [
                DerivativeBlock.Config(length=1024, weight=0.45),
                DerivativeBlock.Config(length=2048, weight=0.4),
                DerivativeBlock.Config(length=4096, weight=0.15),
            ],
        )
        """Candidate derivative block lengths and their relative probabilities."""

        join_rows: int = 16
        """Rows linearly joining the preceding derivative to the next block."""

        reverse_probability: float = 0.0
        """Probability of reversing and negating an empirical block."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lengths = [block.length for block in config.blocks]
        if any(length <= 0 for length in self.lengths):
            raise ValueError("Bootstrap block lengths must be positive.")
        weights = np.asarray(
            [block.weight for block in config.blocks],
            dtype=np.float64,
        )
        _validate_distribution(
            weights,
            count=len(self.lengths),
            name="Bootstrap block",
        )
        self.weights = weights / weights.sum()
        self.pool: list[NDArray[np.float32]] = []

    def prepare(self, wells: Sequence[WellRecord]) -> None:
        """Snapshot complete eligible derivative arrays before any simulation.

        Args:
          wells: Complete training observations before augmentation.

        """
        self.pool = []
        for well in wells:
            selected = np.isnan(well.horizontal["TVT_input"])
            if np.count_nonzero(selected) < self.config.minimum_suffix_rows:
                continue
            values = well.horizontal["TVT"][selected].astype(np.float32, copy=True)
            finite = np.isfinite(values)
            if np.count_nonzero(finite) < 2:
                continue
            if not finite.all():
                axis = np.arange(len(values), dtype=np.float32)
                values = np.interp(axis, axis[finite], values[finite]).astype(
                    np.float32,
                )
            differences = np.diff(values)
            differences = differences[
                np.isfinite(differences)
                & (np.abs(differences) < self.config.maximum_step)
            ]
            if len(differences):
                self.pool.append(differences)
        if not self.pool:
            raise ValueError("Bootstrap trend pool is empty.")

    def __call__(
        self,
        length: int,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> NDArray[np.float32]:
        """Sample a complete anchor-zero path from the prepared training pool.

        Args:
          length: Number of output rows; nonpositive requests consume no draws.
          numpy: Caller-owned legacy stream for derivative-block choices.
          python: Caller-owned stream for donor, offset, and reversal draws.

        Returns:
          path: Independently owned float32 depths relative to the first row.

        Raises:
          ValueError: A positive request precedes preparation of eligible donors.

        """
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        if not self.pool:
            raise ValueError("Bootstrap trend pool is empty.")
        # The donor selection precedes derivative-block selection in this recipe.
        python.randint(0, len(self.pool) - 1)
        derivatives = np.empty(length - 1, dtype=np.float32)
        cursor = 0
        while cursor < len(derivatives):
            requested = min(
                int(
                    cast(
                        "np.int64",
                        numpy.choice(self.lengths, p=self.weights),
                    ),
                ),
                len(derivatives) - cursor,
            )
            donor = self.pool[python.randint(0, len(self.pool) - 1)]
            count = min(requested, len(donor), len(derivatives) - cursor)
            start = python.randint(0, len(donor) - count) if len(donor) > count else 0
            block = donor[start : start + count].copy()
            if len(block) > 1 and python.random() < self.config.reverse_probability:
                block = -block[::-1]
            if cursor and self.config.join_rows:
                overlap = min(self.config.join_rows, count)
                blend = (np.arange(overlap, dtype=np.float32) + 1) / np.float32(
                    overlap + 1,
                )
                block[:overlap] = (1 - blend) * derivatives[cursor - 1] + blend * block[
                    :overlap
                ]
            derivatives[cursor : cursor + count] = block
            cursor += count
        result = np.empty(length, dtype=np.float32)
        result[0] = 0
        result[1:] = np.cumsum(derivatives).astype(np.float32)
        return (result - result.flat[0]).astype(np.float32)


class BoxcarComponent:
    """One boxcar width of the depth corruption and its variance mass."""

    class Config(Fig["BoxcarComponent"]):
        window: int = 1
        """Rows of the boxcar average; one is pointwise noise."""

        weight: float = 1.0
        """Variance mixture weight of the standardized component."""


class MismatchRange:
    """One interval of gamma mismatch standard deviations and its sampling mass."""

    class Config(Fig["MismatchRange"]):
        lower: float = 0.0
        """Smallest sampled gamma mismatch standard deviation."""

        upper: float = 0.0
        """Largest sampled gamma mismatch standard deviation."""

        weight: float = 1.0
        """Relative sampling probability of this interval."""


class GammaBias:
    """Synthesize correlated TVT-bias mismatch at the unchanged physical scales."""

    class Config(Fig["GammaBias"]):
        depth_std: float = 0.95
        """Standard deviation of the physical depth corruption."""

        components: list[BoxcarComponent.Config] = field(
            default_factory=lambda: [
                BoxcarComponent.Config(window=1536, weight=0.68),
                BoxcarComponent.Config(window=256, weight=0.2),
                BoxcarComponent.Config(window=2, weight=0.12),
            ],
        )
        """Long, medium and pointwise boxcar widths with their variance weights."""

        ranges: list[MismatchRange.Config] = field(
            default_factory=lambda: [
                MismatchRange.Config(lower=5.6, upper=8.4, weight=0.24),
                MismatchRange.Config(lower=8.4, upper=12.2, weight=0.51),
                MismatchRange.Config(lower=12.2, upper=17.5, weight=0.25),
            ],
        )
        """Gamma mismatch standard deviation intervals and their probabilities."""

        offset_std: float = 2.0
        """Standard deviation of the sample-wide gamma offset."""

        clip: float = 36.0
        """Largest absolute gamma mismatch."""

    def __init__(self, config: Config) -> None:
        self.config = config
        with np.errstate(over="ignore"):
            components = np.asarray(
                [c.weight for c in config.components],
                dtype=np.float32,
            )
        _validate_distribution(
            components,
            count=len(config.components),
            name="Gamma component",
        )
        _validate_distribution(
            np.asarray([r.weight for r in config.ranges]),
            count=len(config.ranges),
            name="Gamma range",
        )

    def __call__(
        self,
        depth: NDArray[np.float32],
        matched_gamma: NDArray[np.float64],
        *,
        reference_depth: NDArray[np.float32],
        reference_gamma: NDArray[np.float32],
        numpy: np.random.RandomState,
    ) -> NDArray[np.float32]:
        """Return the complete clipped gamma mismatch for a simulated path.

        Args:
          depth: Complete simulated float32 depth path.
          matched_gamma: Reference gamma already interpolated at those depths.
          reference_depth: Ordered physical depth axis of the paired curve.
          reference_gamma: Paired float32 gamma observations on that axis.
          numpy: Caller-owned legacy stream for all corruption draws.

        Returns:
          mismatch: Independently owned float32 residuals, clipped to the
            configured physical gamma bound in the recipe's arithmetic order.

        """
        config = self.config
        weights = np.asarray([c.weight for c in config.components], dtype=np.float32)
        weights /= weights.sum()
        components: list[NDArray[np.float64]] = []
        for component, weight in zip(config.components, weights, strict=True):
            window = component.window
            raw = numpy.normal(0, 1, size=len(depth) + window - 1).astype(np.float32)
            if window > 1:
                raw = _boxcar_mean(raw, width=window)
            components.append(np.float64(weight) ** np.float64(0.5) * _standardize(raw))
        noise = (
            np.float32(config.depth_std) * _standardize(np.sum(components, axis=0))
        ).astype(np.float32)
        corrupted = np.clip(
            depth + noise,
            reference_depth.flat[0],
            reference_depth.flat[-1],
        ).astype(np.float32)
        gamma = np.interp(
            corrupted,
            reference_depth,
            reference_gamma,
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
        result = (gamma - matched_gamma.astype(np.float32)).astype(np.float32)
        finite = np.isfinite(result)
        if finite.any():
            probabilities = np.asarray(
                [r.weight for r in config.ranges],
                dtype=np.float64,
            )
            probabilities /= probabilities.sum()
            chosen = config.ranges[numpy.choice(len(probabilities), p=probabilities)]
            scale = float(numpy.uniform(chosen.lower, chosen.upper))
            current = float(np.std(result[finite]))
            if current > 1e-6:
                result = (result * np.float32(scale / current)).astype(np.float32)
            result = (result + np.float32(numpy.normal(0, config.offset_std))).astype(
                np.float32,
            )
        return np.clip(result, -config.clip, config.clip).astype(np.float32)


class ReferenceJitter:
    """Corrupt the paired gamma curve while preserving horizontal residuals."""

    class Config(Fig["ReferenceJitter"]):
        probability: float = 0.2
        """Probability of coherent paired-curve corruption."""

        gain_std: float = 0.04
        """Standard deviation of gain around one."""

        gain_bounds: tuple[float, float] = (0.9, 1.1)
        """Inclusive gain clipping interval."""

        offset_std: float = 2.0
        """Sample-wide gamma offset standard deviation."""

        drift_std: float = 1.5
        """Standard deviation of smooth gamma drift."""

        drift_windows: tuple[float, ...] = (15, 30, 60)
        """Physical widths sampled for reference drift."""

        smooth_probability: float = 0.1
        """Conditional probability of smoothing the corrupted curve."""

        smooth_windows: tuple[int, ...] = (5, 11, 21)
        """Candidate smoothing widths."""

        smooth_blend: tuple[float, float] = (0.1, 0.35)
        """Uniform interval for smoothing mixture mass."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def apply(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        """Corrupt the well with this component's own probability.

        Args:
          well: Complete horizontal, paired-curve, and particle observations.
          numpy: Caller-owned legacy stream for continuous curve corruption.
          python: Caller-owned stream for the gate, mode, window, and texture draws.

        Returns:
          simulated: The corrupted well, or the input when the gate stays closed.

        """
        if python.random() >= self.config.probability:
            return well
        return self(well, numpy=numpy, python=python)

    def __call__(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        """Return independently owned coherent curve and horizontal observations.

        Args:
          well: Complete horizontal, paired-curve, and particle observations.
          numpy: Caller-owned legacy stream for continuous curve corruption.
          python: Caller-owned stream for mode, window, and texture draws.

        Returns:
          simulated: Complete well owning every array and mapping independently
            of the input, including when no finite reference gamma exists.

        """
        well = deepcopy(well)
        python.random()  # The single enabled residual-copy mode still takes a draw.
        if not np.isfinite(well.reference_gamma).any():
            return well
        after = self._curve(well, numpy=numpy, python=python)
        well = self._copy_residual(well, after=after)
        return replace(well, reference_gamma=after)

    def _curve(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> NDArray[np.float32]:
        """Sample the complete paired curve from the two owned streams."""
        config = self.config
        before = well.reference_gamma
        finite = np.isfinite(before)
        center = np.float32(np.mean(before[finite]))
        gain = np.float32(1 + numpy.normal(0, config.gain_std))
        gain = np.float32(np.clip(gain, *config.gain_bounds))
        offset = np.float32(numpy.normal(0, config.offset_std))
        after = (center + gain * (before - center) + offset).astype(np.float32)
        if len(after) > 1:
            widths = np.diff(well.reference_depth)
            widths = widths[np.isfinite(widths) & (widths > 0)]
            width = (
                max(
                    1,
                    round(
                        python.choice(config.drift_windows) / float(np.median(widths)),
                    ),
                )
                if len(widths)
                else 1
            )
            noise = numpy.normal(0, 1, size=len(after)).astype(np.float32)
            width = python.choice((width,))
            noise = _smooth(noise, width)
            deviation = float(np.std(noise))
            if deviation > 1e-6:
                noise *= np.float32(config.drift_std / deviation)
            else:
                noise[:] = 0
            after = (after + noise).astype(np.float32)
        if len(after) > 2 and python.random() < config.smooth_probability:
            smooth = _smooth(after, python.choice(config.smooth_windows))
            blend = np.float32(numpy.uniform(*config.smooth_blend))
            after = ((np.float32(1) - blend) * after + blend * smooth).astype(
                np.float32,
            )
        return after

    def _copy_residual(
        self,
        well: WellRecord,
        *,
        after: NDArray[np.float32],
    ) -> WellRecord:
        """Transport complete float32 residuals while retaining raw missing rows."""
        before = well.reference_gamma
        depth = well.horizontal["TVT"].astype(np.float32, copy=False)
        gamma = well.horizontal["GR"].astype(np.float32, copy=False)
        old_match = np.interp(
            depth,
            well.reference_depth,
            before,
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
        new_match = np.interp(
            depth,
            well.reference_depth,
            after,
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
        residual = (gamma - old_match).astype(np.float32)
        if np.isfinite(residual).any():
            output = (new_match + _fill(residual)).astype(np.float32)
            output[~np.isfinite(new_match)] = gamma[~np.isfinite(new_match)]
            output[~np.isfinite(gamma)] = np.nan
            well = replace(well, horizontal={**well.horizontal, "GR": output})
        return well


class ReferenceTransform:
    """Apply independent affine calibration, smoothing and sharpening."""

    class Config(Fig["ReferenceTransform"]):
        probability: float = 0.6
        """Probability of transforming the paired gamma curve."""

        gain_std: float = 0.05
        """Affine gain standard deviation."""

        offset_std: float = 2.0
        """Affine gamma offset standard deviation."""

        smooth_probability: float = 0.1
        """Conditional probability of smoothing the affine curve."""

        sharpen_probability: float = 0.1
        """Conditional probability of sharpening instead of smoothing."""

        smooth_window: int = 5
        """Moving average width for both texture branches."""

        sharpen_alpha: tuple[float, float] = (0, 0.3)
        """Uniform interval for the sharpening update scale."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def apply(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        """Transform the paired curve with this component's own probability.

        Args:
          well: Complete observations with the paired curve to transform.
          numpy: Caller-owned legacy stream for affine and sharpening draws.
          python: Caller-owned stream for the gate and the texture branch.

        Returns:
          transformed: The transformed well, or the input when the gate stays closed.

        """
        if python.random() >= self.config.probability:
            return well
        return self(well, numpy=numpy, python=python)

    def __call__(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        """Transform the full reference curve in the recipe's arithmetic order.

        Args:
          well: Complete observations with the paired curve to transform.
          numpy: Caller-owned legacy stream for affine and sharpening draws.
          python: Caller-owned stream selecting the texture branch.

        Returns:
          transformed: Well with an independently allocated gamma curve; other
            input arrays and mappings are retained without mutation.

        """
        config = self.config
        gain = np.float32(numpy.normal(1, config.gain_std))
        offset = np.float32(numpy.normal(0, config.offset_std))
        result = (gain * well.reference_gamma + offset).astype(np.float32)
        texture = python.random()
        if texture < config.smooth_probability:
            result = _smooth(result, config.smooth_window)
        elif texture < config.smooth_probability + config.sharpen_probability:
            smooth = _smooth(result, config.smooth_window)
            alpha = np.float32(numpy.uniform(*config.sharpen_alpha))
            result = (result + alpha * (result - smooth)).astype(np.float32)
        return replace(well, reference_gamma=result)


class FixedNanPattern:
    """Shift or keep the missing-gamma mask of every simulated target region."""

    class Config(Fig["FixedNanPattern"]):
        shift: bool = True
        """Whether the target region's missing-gamma mask is cyclically shifted."""

    def __init__(self, config: Config) -> None:
        self.shift = config.shift

    def __call__(self, python: Random) -> bool:
        """Return the fixed policy without consuming a draw."""
        del python
        return self.shift


class SampledNanPattern:
    """Sample the missing-gamma mask policy once per simulated well."""

    class Config(Fig["SampledNanPattern"]):
        weights: dict[bool, float] = field(default_factory=lambda: {True: 1.0})
        """Relative probability of shifting (True) or keeping (False) the mask."""

    def __init__(self, config: Config) -> None:
        self.weights = config.weights

    def __call__(self, python: Random) -> bool:
        """Draw the policy from the owned Python stream."""
        return python.choices(
            tuple(self.weights),
            weights=tuple(self.weights.values()),
        )[0]


class PathSimulation:
    """Generate complete native path and reference observations before binning."""

    class Config(Fig["PathSimulation"]):
        trend: BootstrapTrend.Config = field(default_factory=BootstrapTrend.Config)
        """Empirical derivative-block donor distribution."""

        gamma_bias: GammaBias.Config = field(default_factory=GammaBias.Config)
        """Correlated depth-bias gamma residual generator."""

        reference_jitter: ReferenceJitter.Config = field(
            default_factory=ReferenceJitter.Config,
        )
        """Coherent curve and horizontal residual corruption."""

        reference_transform: ReferenceTransform.Config = field(
            default_factory=ReferenceTransform.Config,
        )
        """Independent paired-curve affine and texture transformation."""

        shift_probability: float = 0.85
        """Probability of replacing the raw target trend."""

        prefix_sample_rows: int = 0
        """Additional visible-prefix rows included in trend simulation."""

        shift_nan_pattern: Makeable[Callable[[Random], bool]] = field(
            default_factory=SampledNanPattern.Config,
        )
        """Policy for cyclic target-region missing gamma masks, fixed or sampled."""

        shift_range: float = 40.0
        """Uniform depth-anchor shift bound."""

        jump_threshold: float = 0.15
        """Smallest original depth discontinuity retained during simulation."""

        fault_probability: float = 0.05
        """Probability of adding sparse cumulative Laplace faults."""

        maximum_faults: int = 3
        """Largest sampled number of fault points."""

        fault_std: float = 1.75
        """Laplace fault scale."""

        reverse_probability: float = 0.25
        """Probability of reversing the observed path."""

        reverse_head_bound: float = 10.0
        """Reversal cutoff threshold; no crossing retains the complete path."""

        reverse_visible_rows: int = 256
        """Visible rows retained at the new reversed prefix."""

        gamma_noise_shift_probability: float = 0.0
        """Probability of cyclically shifting horizontal gamma residuals."""

        gamma_noise_shift_ratio: float = 1.0
        """Largest cyclic offset relative to suffix length."""

        reference_drift_probability: float = 0.3
        """Probability of shifting the entire reference depth axis."""

        reference_drift_std: float = 1.0
        """Reference depth-axis shift standard deviation."""

        stretch_probability: float = 0.4
        """Probability of resampling the complete MD path."""

        stretch_range: tuple[float, float] = (0.7, 1.3)
        """Uniform ratio of output rows to original rows."""

        tail_cut_probability: float = 0.2
        """Probability of shortening the raw masked suffix."""

        maximum_tail_cut: float = 0.5
        """Largest uniformly sampled removed suffix fraction."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.trend = config.trend.make()
        self.gamma_bias = config.gamma_bias.make()
        self.reference_jitter = config.reference_jitter.make()
        self.reference_transform = config.reference_transform.make()
        self.shift_nan_pattern = config.shift_nan_pattern.make()

    def prepare(self, wells: Sequence[WellRecord]) -> None:
        """Bind immutable training donors before any worker consumes a sample.

        Args:
          wells: Complete training observations before any simulation draws.

        """
        self.trend.prepare(wells)

    def __call__(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        """Return one complete independently owned simulated well.

        Prepare eligible trend donors before enabling path replacement.
        Reversal may leave no hidden target rows, as in the public recipe.

        Args:
          well: Complete input paths, paired curve, and optional particle cache.
          numpy: Caller-owned legacy stream for continuous simulation draws.
          python: Caller-owned stream for ordered event and discrete draws,
            including the recipe's disabled-event draws.

        Returns:
          simulated: Complete well with every array and mapping independently
            owned, preserving inputs and consuming caller streams in source order.

        """
        config = self.config
        well = deepcopy(well)
        python.random()  # Disabled keep-original event.
        if python.random() < config.shift_probability:
            python.random()  # Single enabled TVT-bias residual mode.
            well = self._shift(well, numpy=numpy, python=python)
        python.random()  # Disabled XY shift.
        python.random()  # Disabled reference replacement.
        well = self.reference_jitter.apply(well, numpy=numpy, python=python)
        if python.random() < config.reverse_probability:
            horizontal = well.horizontal
            visible = horizontal["TVT_input"]
            anchor = visible.flat[np.flatnonzero(np.isfinite(visible)).flat[-1]]
            qualifying = np.flatnonzero(
                np.abs(visible - anchor) > config.reverse_head_bound,
            )
            start = int(qualifying.flat[-1]) + 1 if len(qualifying) else 0
            reversed_horizontal = {
                name: values[start:][::-1] for name, values in horizontal.items()
            }
            visible = reversed_horizontal["TVT"].copy()
            visible[config.reverse_visible_rows :] = np.nan
            reversed_horizontal["TVT_input"] = visible
            reversed_horizontal["MD"] = np.arange(len(visible), dtype=np.float32)
            well = replace(well, horizontal=reversed_horizontal)
        python.random()  # Disabled prefix-boundary shift.
        if python.random() < config.gamma_noise_shift_probability:
            horizontal = well.horizontal
            suffix = np.isnan(horizontal["TVT_input"])
            depth = horizontal["TVT"][suffix]
            matched = np.interp(
                depth,
                well.reference_depth,
                well.reference_gamma,
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)
            noise = horizontal["GR"][suffix] - matched
            if len(noise):
                shift = int(
                    numpy.uniform(
                        -len(noise) * config.gamma_noise_shift_ratio,
                        len(noise) * config.gamma_noise_shift_ratio,
                    ),
                )
                horizontal["GR"][suffix] = (
                    matched + noise[(np.arange(len(noise)) + shift) % len(noise)]
                )
        well = self.reference_transform.apply(well, numpy=numpy, python=python)
        if python.random() < config.reference_drift_probability:
            well = replace(
                well,
                reference_depth=(
                    well.reference_depth
                    + np.float32(numpy.normal(0, config.reference_drift_std))
                ).astype(np.float32),
            )
        if python.random() < config.stretch_probability:
            md = well.horizontal["MD"]
            axis = np.linspace(
                md.flat[0],
                md.flat[-1],
                int(len(md) * numpy.uniform(*config.stretch_range)),
                dtype=np.float64,
            )
            resampled: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
                name: np.interp(axis, md, values, left=np.nan, right=np.nan).astype(
                    np.float32,
                )
                for name, values in well.horizontal.items()
            }
            resampled["MD"] = axis
            well = replace(well, horizontal=resampled)
        if python.random() < config.tail_cut_probability:
            horizontal = well.horizontal
            visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"]))
            suffix_rows = len(horizontal["MD"]) - int(visible.flat[-1]) - 1
            if suffix_rows > 1:
                count = min(
                    int(suffix_rows * numpy.uniform(0, config.maximum_tail_cut)),
                    suffix_rows - 1,
                )
                if count:
                    well = replace(
                        well,
                        horizontal={
                            name: values[:-count] for name, values in horizontal.items()
                        },
                    )
        return well

    def _shift(
        self,
        well: WellRecord,
        *,
        numpy: np.random.RandomState,
        python: Random,
    ) -> WellRecord:
        config = self.config
        horizontal = well.horizontal
        depth = horizontal["TVT"].astype(np.float32, copy=False)
        known = horizontal["TVT_input"].astype(np.float32, copy=False)
        suffix = np.isnan(known)
        selected = suffix.copy()
        last = int(np.flatnonzero(np.isfinite(known)).flat[-1])
        if config.prefix_sample_rows > 0:
            selected[max(0, last + 1 - config.prefix_sample_rows) : last + 1] = True
        if np.count_nonzero(selected) == 0:
            return well
        shift_nan_pattern = self.shift_nan_pattern(python)
        python.random()  # Single enabled residual-noise mode.
        original = depth[selected].copy()
        if python.random() < config.fault_probability and len(original) > 1:
            faults = numpy.randint(0, min(config.maximum_faults, len(original) - 1) + 1)
            if faults:
                positions = numpy.choice(
                    np.arange(1, len(original)),
                    size=faults,
                    replace=False,
                )
                jumps = np.zeros(len(original), dtype=np.float32)
                jumps[positions] = numpy.laplace(0, config.fault_std, size=faults)
                original += np.cumsum(jumps)
        trend = self.trend(len(original), numpy=numpy, python=python)
        python.random()  # The disabled large-drift event still takes a draw.
        simulated = original.flat[0] + trend
        jumps = np.diff(original)
        jumps[np.abs(jumps) < config.jump_threshold] = 0
        simulated[1:] += np.cumsum(jumps)
        changed = depth.copy()
        changed[selected] = simulated
        z = horizontal["Z"].astype(np.float32, copy=True)
        z[selected] = original + z[selected] - simulated
        anchor = float(
            changed.flat[last] if config.prefix_sample_rows > 0 else known.flat[last],
        )
        lower = max(
            -config.shift_range,
            anchor - (float(well.reference_depth.flat[-1]) - 100),
        )
        upper = min(
            config.shift_range,
            anchor - (float(well.reference_depth.flat[0]) + 100),
        )
        shift = float(numpy.uniform(lower, upper)) if lower <= upper else 0
        new_known = known
        if shift:
            changed -= shift
            z += shift
            new_known = known.copy()
            finite = np.isfinite(new_known)
            new_known[finite] -= shift
        if config.prefix_sample_rows > 0:
            new_known = known.copy()
            new_known[~suffix] = changed[~suffix]
        gamma = horizontal["GR"]
        simulated_match = np.interp(
            changed,
            well.reference_depth,
            well.reference_gamma,
            left=np.nan,
            right=np.nan,
        )
        match = np.interp(
            depth,
            well.reference_depth,
            well.reference_gamma,
            left=np.nan,
            right=np.nan,
        )
        residual = gamma - match
        residual[selected] = self.gamma_bias(
            changed[selected],
            simulated_match[selected],
            reference_depth=well.reference_depth,
            reference_gamma=well.reference_gamma,
            numpy=numpy,
        )
        output = simulated_match + residual
        if shift_nan_pattern:
            region = output[selected].copy()
            if np.isfinite(region).any():
                region = _fill(region)
            else:
                region = np.zeros_like(region, dtype=np.float32)
            mask = np.isnan(gamma[selected])
            offset = python.randrange(len(mask)) if len(mask) > 1 else 0
            region[mask[(np.arange(len(mask)) + offset) % len(mask)]] = np.nan
            output[selected] = region
        else:
            output[np.isnan(gamma)] = np.nan
        if shift == 0:
            output[~selected] = gamma[~selected]
        return replace(
            well,
            horizontal={
                **horizontal,
                "Z": z,
                "TVT": changed,
                "TVT_input": new_known,
                "GR": output,
            },
        )


def _validate_distribution(
    weights: NDArray[np.float32] | NDArray[np.float64],
    *,
    count: int,
    name: str,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        total = weights.sum()
    if (
        len(weights) != count
        or np.count_nonzero(~np.isfinite(weights))
        or np.count_nonzero(weights < 0)
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError(f"{name} weights must form a positive distribution.")


def _boxcar_mean(values: NDArray[np.float32], *, width: int) -> NDArray[np.float32]:
    """Average every complete window in float64, rounded once to float32."""
    # np.convolve sends each window to BLAS dot, whose summation order varies by host.
    windows = sliding_window_view(values.astype(np.float64), width)
    return (windows.sum(axis=1) / width).astype(np.float32)


def _standardize(values: NDArray[np.generic]) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    deviation = float(np.std(values))
    centered = values - float(np.mean(values))
    return centered if deviation <= 1e-6 else (centered / deviation).astype(np.float32)


def _smooth(values: NDArray[np.float32], width: int) -> NDArray[np.float32]:
    width = min(width, len(values))
    if width <= 2:
        return values.copy()
    if width % 2 == 0:
        width -= 1
    return _boxcar_mean(np.pad(values, width // 2, mode="edge"), width=width)


def _fill(values: NDArray[np.generic]) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    axis = np.arange(len(values), dtype=np.float32)
    return np.interp(axis, axis[finite], values[finite]).astype(np.float32)


class ImageAugmentation:
    """Corrupt complete PF feature rows and selected horizontal intervals."""

    class Config(Fig["ImageAugmentation"]):
        rotate_probability: float = 0.1
        """Probability of shifting target-shaped PF priors."""

        ratio_jitter: float = 0.25
        """Uniform multiplicative jitter around the fitted target slope."""

        maximum_shift: float = 20.0
        """Physical shift bound; infinity leaves the shift unbounded."""

        channel_probability: float = 0.7
        """Probability of attempting feature-channel masking."""

        channel_mask_probability: float = 0.1
        """Independent masking probability for each non-PF channel."""

        particle_mask_probability: float = 0.05
        """Joint masking probability for all PF feature planes."""

        sequence_probability: float = 0.1
        """Probability of masking one contiguous suffix interval."""

        sequence_mask_ratio: float = 0.3
        """Largest sampled fraction of masked suffix bins."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        item: FeatureItem,
        *,
        features: ImageFeatures.Config,
        numpy: np.random.RandomState,
        python: Random,
    ) -> FeatureItem:
        """Apply the complete active image corruption sequence in place.

        Args:
          item: Complete mutable image planes, targets and original-row metadata.
          features: Candidate geometry, target normalization and ordered channels.
          numpy: Caller-owned legacy stream for continuous corruption draws.
          python: Caller-owned stream for discrete augmentation decisions.

        Returns:
          item: The same complete item with selected image intervals corrupted.

        """
        if python.random() < self.config.rotate_probability:
            self._rotate(item, features=features, numpy=numpy)
        python.random()  # Disabled candidate-prior corruption.
        if python.random() < self.config.channel_probability:
            particle = python.random() < self.config.particle_mask_probability
            image = item.arrays["unet_static"]
            for index, name in enumerate(features.channels):
                if name in PARTICLE_PLANES:
                    if particle:
                        image[index] = 0
                elif python.random() < self.config.channel_mask_probability:
                    image[index] = 0
        if python.random() < self.config.sequence_probability:
            selected = np.flatnonzero(item.arrays["suffix_mask"])
            if len(selected):
                ratio = numpy.uniform(0, self.config.sequence_mask_ratio)
                count = min(len(selected), max(1, math.ceil(ratio * len(selected))))
                offset = python.randint(0, len(selected) - count)
                begin, end = (
                    int(selected.flat[offset]),
                    int(selected.flat[offset + count - 1]) + 1,
                )
                item.arrays["geo_s_rel_prior"][begin:end] = 0
                item.arrays["unet_static"][:, begin:end] = 0
        python.random()  # Disabled horizontal flip.
        return item

    def _rotate(
        self,
        item: FeatureItem,
        *,
        features: ImageFeatures.Config,
        numpy: np.random.RandomState,
    ) -> None:
        channels = {name: index for index, name in enumerate(features.channels)}
        if not set(channels).intersection(
            ("pf_prob", "pf_prob_ffbsi", "pf_tvt", "pf_tvt_ffbsi"),
        ):
            return
        target = item.arrays["target"].astype(np.float32)
        valid = (
            item.arrays["target_mask"].astype(bool)
            & item.arrays["suffix_mask"].astype(bool)
            & np.isfinite(target)
        )
        if np.count_nonzero(valid) < 2:
            return
        mean, std = features.target_stats
        relative = np.nan_to_num(
            target * np.float32(std) + np.float32(mean),
            posinf=0,
            neginf=0,
        )
        center = float(relative.flat[np.flatnonzero(valid).flat[0]])
        centered = relative[valid] - np.float32(center)
        centered64 = np.asarray(centered, dtype=np.float64)
        denominator = float(
            cast(
                "np.float64",
                np.dot(centered64, centered64),
            ),
        )
        if denominator <= 1e-6:
            return
        slope = float(
            cast(
                "np.float64",
                np.dot(centered64, relative[valid]),
            )
            / denominator,
        )
        ratio = float(
            numpy.uniform(1 - self.config.ratio_jitter, 1 + self.config.ratio_jitter),
        )
        base = np.zeros_like(relative)
        base[valid] = relative[valid] - np.float32(center)
        delta = np.float32(slope * ratio) * base
        delta = np.clip(delta, -self.config.maximum_shift, self.config.maximum_shift)
        delta[~valid] = 0
        if not np.isfinite(delta).all() or float(
            cast("np.float32", np.abs(delta).max()),
        ) <= float(np.float32(1e-6)):
            return
        columns = features.reference_columns
        axis = np.linspace(
            -features.reference_window + features.reference_window / columns,
            features.reference_window - features.reference_window / columns,
            columns,
            dtype=np.float32,
        )
        image = item.arrays["unet_static"]
        for probability_name, path_name in (
            ("pf_prob", "pf_tvt"),
            ("pf_prob_ffbsi", "pf_tvt_ffbsi"),
        ):
            active = None
            if probability_name in channels:
                channel = cast(
                    "NDArray[np.float32]",
                    image[channels[probability_name]],
                )
                probability = normalize_probability(
                    image_probability(np.maximum(channel, 0), columns=columns),
                )
                shifted = shift_probability_axis(probability, delta=delta, axis=axis)
                active = shifted.sum(axis=1) > 0
                image[channels[probability_name]] = probability_image(
                    shifted,
                    columns=columns,
                )
            if path_name in channels:
                channel = cast(
                    "NDArray[np.generic]",
                    image[channels[path_name]],
                ).astype(np.float32)
                if active is None:
                    active = np.any(np.abs(channel) > 1e-6, axis=1)
                depth = channel[:, :1] * np.float32(std) + np.float32(mean)
                depth[active] += delta[active, None]
                shifted_path = (depth - mean) / std
                image[channels[path_name]] = np.broadcast_to(
                    shifted_path,
                    channel.shape,
                )
                image[channels[path_name], ~active] = 0


class AugmentedFeatures:
    """Preserve complete observed paths while sampling native training inputs."""

    class Config(Fig["AugmentedFeatures"]):
        features: ImageFeatures.Config = field(default_factory=ImageFeatures.Config)
        """Complete model image and objective target builder."""

        paths: PathSimulation.Config | None = field(
            default_factory=PathSimulation.Config,
        )
        """Complete raw path and paired-reference simulation; None keeps the
        observed paths."""

        images: ImageAugmentation.Config | None = field(
            default_factory=ImageAugmentation.Config,
        )
        """PF, channel and horizontal-interval corruption of training images; None
        leaves every image intact."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.paths = config.paths.make() if config.paths is not None else None
        self.features = config.features.make()
        self.images = config.images.make() if config.images is not None else None

    def prepare(self, donors: Sequence[WellRecord]) -> None:
        """Snapshot the complete training donor pool before drawing samples."""
        if self.paths is not None:
            self.paths.prepare(donors)

    def __call__(
        self,
        well: WellRecord,
        *,
        training: bool,
        numpy: np.random.RandomState,
        python: Random,
    ) -> FeatureItem:
        """Build the complete next model item from owned RNG streams.

        Args:
          well: Complete observed paths and their calibrated paired curve.
          training: Retain normalized labels and corrupt the image when declared.
          numpy: Caller-owned legacy stream for continuous simulation draws.
          python: Caller-owned stream for discrete simulation decisions.

        Returns:
          item: Complete model arrays, auxiliary targets and row metadata.

        """
        if self.paths is not None:
            well = self.paths(well, numpy=numpy, python=python)
        item = self.features(well, training=training)
        if self.images is None or not training:
            return item
        return self.images(
            item,
            features=self.config.features,
            numpy=numpy,
            python=python,
        )


def read_numeric_csv(
    path: Path,
    *,
    columns: Sequence[str],
) -> dict[str, NDArray[np.float64]]:
    """Read complete numeric columns in requested order without table runtimes.

    Args:
      path: Complete competition numeric CSV, possibly containing unused text.
      columns: Required numeric column names in returned dictionary order.

    Returns:
      arrays: Full float64 columns with empty observations represented by NaN.

    """
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}.")
        values: dict[str, list[float]] = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                value = row[name]
                values[name].append(_csv_float(value))
    return {
        name: np.asarray(column, dtype=np.float64) for name, column in values.items()
    }


def discover_wells(directory: Path) -> tuple[str, ...]:
    """Return sorted well identifiers, requiring both competition CSVs.

    Args:
      directory: One extracted train or test directory.

    Returns:
      wells: Lexicographically ordered well identifiers.

    """
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in directory.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv")
        for path in directory.glob("*__typewell.csv")
    }
    if horizontal != typewell:
        raise FileNotFoundError(
            f"Missing paired well CSVs: {sorted(horizontal ^ typewell)}.",
        )
    if not horizontal:
        raise FileNotFoundError(f"No paired well CSVs found in {directory}.")
    return tuple(sorted(horizontal))


def read_centroids(path: Path, *, well_ids: Sequence[str]) -> NDArray[np.float64]:
    """Align the complete published geographic map to exact training-well order.

    Args:
      path: Full centroid CSV with six-decimal coordinates.
      well_ids: Complete ordered training identifiers.

    Returns:
      coordinates: Unscaled float64 coordinates in the requested well order.

    """
    values: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["well_id"]
            if name in values:
                raise ValueError(f"Duplicate centroid identifier: {name}.")
            values[name] = (
                float(row["horizontal_avg_X"]),
                float(row["horizontal_avg_Y"]),
            )
    missing = set(well_ids).difference(values)
    if missing:
        raise ValueError(f"Missing centroid identifiers: {sorted(missing)}.")
    coordinates = np.asarray([values[name] for name in well_ids], dtype=np.float64)
    if np.count_nonzero(~np.isfinite(coordinates)):
        raise ValueError("Geographic centroid coordinates must be finite.")
    return coordinates


class ReferenceCalibration:
    """Blend rounded prefix means into the raw reference curve inside its support."""

    class Config(Fig["ReferenceCalibration"]):
        blend: float = 0.7
        """Prefix pseudo-curve mass inside the visible support range."""

        raw_weight: float = 0.3
        """Raw paired-curve mass there; the public recipe fixes it independently
        because one minus the blend changes the float64 result bits."""

        window: float = 100.0
        """Depth radius around the final visible anchor admitted to the prefix means."""

        resolution: float = 4.0
        """Depth cells per physical unit when rounding the prefix means."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        depth: NDArray[np.float64],
        *,
        gamma: NDArray[np.float64],
        reference_depth: NDArray[np.float64],
        reference_gamma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Blend rounded prefix means into the raw reference inside support.

        Args:
          depth: Complete visible depth input, with hidden suffix observations NaN.
          gamma: Complete measured gamma path in the original row order.
          reference_depth: Complete paired curve coordinates.
          reference_gamma: Complete uncalibrated paired curve observations.

        Returns:
          calibrated: Owned full float64 paired curve with unchanged exterior values.

        """
        cfg = self.config
        visible = np.flatnonzero(np.isfinite(depth))
        if len(visible) == 0:
            raise ValueError("Reference calibration requires a visible depth anchor.")
        anchor = cast("np.float64", depth[visible[-1]])
        valid = (
            np.isfinite(depth)
            & np.isfinite(gamma)
            & (depth >= anchor - cfg.window)
            & (depth <= anchor + cfg.window)
        )
        result = reference_gamma.copy()
        if np.count_nonzero(valid) == 0:
            return result
        rounded = np.round(depth[valid] * cfg.resolution) / cfg.resolution
        groups, inverse, counts = np.unique(
            rounded,
            return_inverse=True,
            return_counts=True,
        )
        members = np.split(
            gamma[valid][np.argsort(inverse, kind="stable")],
            np.cumsum(counts)[:-1],
        )
        # A float64 sum rounds before its division and misrounds a quarter of means.
        means = np.asarray(
            [statistics.mean(cast("list[float]", group.tolist())) for group in members],
            dtype=np.float64,
        )
        left = cast("np.float64", groups[0])
        right = cast("np.float64", groups[-1])
        supported = (reference_depth >= left) & (reference_depth <= right)
        pseudo = np.interp(reference_depth[supported], groups, means)
        blended = cfg.blend * pseudo + cfg.raw_weight * reference_gamma[supported]
        result[supported] = np.where(
            np.isfinite(blended),
            blended,
            reference_gamma[supported],
        )
        return result


class WorkerStreams:
    """Keep independent legacy NumPy and Python streams for one data consumer."""

    class Config(Fig["WorkerStreams"]):
        seed: int = -1
        """Initial fold seed, replaced by PyTorch's derived seed inside workers; -1
        leaves it to the fold loop."""

    def __init__(self, config: Config) -> None:
        self.numpy = np.random.RandomState(config.seed)
        self.python = InlineConfig(random.Random, config.seed).make()

    def reset(self, seed: int) -> None:
        """Seed both owned streams at consumer initialization."""
        self.numpy.seed(seed)
        self.python.seed(seed)


@runtime_checkable
class WorkerSeeded(Protocol):
    """A dataset that can seed its owned random streams at worker startup."""

    def seed_worker(self, seed: int) -> None:
        """Initialize both owned streams with the derived 32-bit worker seed."""
        ...


def initialize_worker(
    worker_id: int,
    *,
    information: Callable[[], WorkerInfo | None] = get_worker_info,
) -> None:
    """Seed dataset-owned streams once from the complete PyTorch worker seed.

    Args:
      worker_id: PyTorch worker identifier; its derived Torch seed owns seeding.
      information: PyTorch's worker-context reader at the process boundary.

    """
    del worker_id
    worker = information()
    if worker is None:
        raise ValueError("Expected worker is not None.")
    dataset = cast("Dataset[object]", worker.dataset)
    assert isinstance(dataset, WorkerSeeded)
    dataset.seed_worker(torch.initial_seed() % 2**32)


class WellDataset(Dataset[FeatureItem]):
    """Bind complete observations and prepare every donor before iteration."""

    class Config(Fig["WellDataset"]):
        observations: Makeable[Sequence[WellRecord]] = field(
            default_factory=lambda: InlineConfig(tuple),
        )
        """Deferred complete prepared observations in declared well order."""

        transform: AugmentedFeatures.Config = field(
            default_factory=AugmentedFeatures.Config,
        )
        """Raw simulation, complete feature construction and image corruption."""

        streams: WorkerStreams.Config = field(default_factory=WorkerStreams.Config)
        """The dataset's independently owned fold or worker streams."""

        training: bool = True
        """Retain normalized complete labels for training or validation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.observations = config.observations.make()
        self.wells = tuple(self.observations)
        self.transform = config.transform.make()
        self.streams = config.streams.make()
        self.transform.prepare(self.wells)

    def seed_worker(self, seed: int) -> None:
        """Initialize the two streams once for this persistent worker's lifetime."""
        self.streams.reset(seed)

    def __len__(self) -> int:
        """Return the complete declared observation count."""
        return len(self.wells)

    @override
    def __getitem__(self, index: int) -> FeatureItem:
        """Build one complete native item from the next owned stream state.

        Args:
          index: Position in the complete declared well sequence.

        Returns:
          item: Complete model arrays, auxiliary targets and original-row metadata.

        """
        return self.transform(
            self.wells[index],
            training=self.config.training,
            numpy=self.streams.numpy,
            python=self.streams.python,
        )


def collate_features(samples: Sequence[FeatureItem]) -> dict[str, object]:
    """Stack all eleven fields, sorted auxiliary targets and ordered metadata.

    Args:
      samples: Complete model items in sample iteration order.

    Returns:
      batch: Every tensor, all sorted auxiliary targets and complete metadata.

    """
    batch: dict[str, object] = {}
    keys = {"unet_static": "media", "target": "label"}
    for name in (
        "geo_s_rel_prior",
        "unet_static",
        "typewell_aux",
        "z_rel",
        "bin_count",
        "z_diff",
        "target",
        "typewell_target_probs",
        "target_mask",
        "row_mask",
        "suffix_mask",
    ):
        values = np.stack([item.arrays[name] for item in samples])
        batch[keys.get(name, name)] = torch.from_numpy(
            values if values.dtype == np.bool_ else values.astype(np.float32),
        )
    batch["aux_targets"] = {
        name: torch.from_numpy(
            np.stack([item.auxiliary[name] for item in samples]).astype(np.float32),
        )
        for name in sorted(samples[0].auxiliary)
    }
    batch["meta"] = [item.metadata for item in samples]
    return batch


class WellLoader:
    """Construct an actual PyTorch DataLoader with an independent generator."""

    class Config(Fig["WellLoader"]):
        dataset: WellDataset.Config = field(default_factory=WellDataset.Config)
        """Complete eager observations and per-consumer native transformations."""

        seed: int = -1
        """Loader generator seed, separate from the dataset's fold seed; -1 leaves
        it to the fold loop."""

        batch_size: int = 16
        """Complete wells in each model batch."""

        shuffle: bool = True
        """Shuffle samples with the independent owned Torch generator."""

        drop_last: bool = True
        """Discard incomplete training batches."""

        num_workers: int = 4
        """Persistent parallel data consumers in the complete public recipe."""

        prefetch_factor: int = 1
        """Queued batches per data worker."""

        persistent_workers: bool = True
        """Retain worker-owned streams across epochs."""

        pin_memory: bool = False
        """Pin host batches for asynchronous CUDA copies."""

    def __init__(self, config: Config) -> None:
        if config.seed < 0:
            raise ValueError("The loader requires a resolved generator seed.")
        self.dataset = config.dataset.make()
        self.generator = torch.Generator().manual_seed(config.seed)
        self.loader = cast(
            "DataLoader[dict[str, object]]",
            DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                collate_fn=collate_features,
                drop_last=config.drop_last,
                worker_init_fn=initialize_worker,
                generator=self.generator,
                prefetch_factor=config.prefetch_factor if config.num_workers else None,
                persistent_workers=config.persistent_workers
                if config.num_workers
                else False,
            ),
        )


FORMATIONS: Final = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",  # codespell:ignore buda -- Required ROGII horizontal CSV formation key.
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RawWell:
    """Owned complete float64 CSV columns before any precision conversion."""

    well_id: str
    horizontal: dict[str, NDArray[np.float64]]
    reference_depth: NDArray[np.float64]
    reference_gamma: NDArray[np.float64]


def load_raw(directory: Path, well_id: str) -> RawWell:
    """Read every active numeric field from both complete competition CSVs.

    Args:
      directory: Complete competition train or test split.
      well_id: Paired horizontal-well and typewell identifier.

    Returns:
      well: Owned complete original float64 columns.

    """
    horizontal_path = directory / f"{well_id}__horizontal_well.csv"
    with horizontal_path.open(newline="") as stream:
        fields = csv.DictReader(stream).fieldnames or ()
    columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if "TVT" in fields:
        columns.append("TVT")
    if "ANCC" in fields:
        columns.extend(FORMATIONS)
    horizontal = read_numeric_csv(horizontal_path, columns=columns)
    reference = read_numeric_csv(
        directory / f"{well_id}__typewell.csv",
        columns=("TVT", "GR"),
    )
    return RawWell(
        well_id=well_id,
        horizontal=horizontal,
        reference_depth=reference["TVT"],
        reference_gamma=reference["GR"],
    )


def load_split(directory: Path) -> dict[str, RawWell]:
    """Load every complete raw well in stable competition filename order."""
    return {name: load_raw(directory, name) for name in discover_wells(directory)}


def trajectory(well: RawWell) -> WellTrajectory:
    """Expose the complete float32 coordinate path to the fold-safe prior.

    Args:
      well: Complete original numeric CSV observations.

    Returns:
      path: Complete coordinates, optional full truth and the visible boundary.

    """
    values = well.horizontal
    known = values["TVT_input"].astype(np.float32)
    visible = np.flatnonzero(np.isfinite(known))
    if len(visible) == 0:
        raise ValueError("A well requires a visible depth anchor.")
    return WellTrajectory(
        well_id=well.well_id,
        md=values["MD"].astype(np.float32),
        x=values["X"].astype(np.float32),
        y=values["Y"].astype(np.float32),
        z=values["Z"].astype(np.float32),
        known_depth=known,
        depth=values["TVT"].astype(np.float32) if "TVT" in values else None,
        suffix_start=int(visible.flat[-1]) + 1,
    )


def prepare_well(
    well: RawWell,
    *,
    surface: RelativeSurface,
    calibration: ReferenceCalibration,
    particle: Mapping[str, NDArray[np.generic]] | None = None,
) -> WellRecord:
    """Calibrate the complete curve and attach full fold-safe/cache observations.

    Args:
      well: Full float64 input observations before calibration.
      surface: Full fold-safe relative surface and diagnostic observations.
      calibration: Visible-prefix calibration of the paired gamma-ray curve.
      particle: Optional complete sanitized particle-cache arrays.

    Returns:
      prepared: Complete native model/simulation input with owned float32 paths.

    """
    path = trajectory(well)
    if len(surface.values) != len(path.md):
        raise ValueError(
            f"{well.well_id}: geographic prior length differs from horizontal rows.",
        )
    horizontal = {
        name: values.astype(np.float32)
        for name, values in well.horizontal.items()
        if name not in FORMATIONS
    }
    horizontal["orig_index"] = np.arange(len(path.md), dtype=np.float32)
    anchor = (
        path.known_depth.flat[path.suffix_start - 1]
        + path.z.flat[path.suffix_start - 1]
    )
    horizontal["geo_s_rel_prior_abs"] = (anchor + surface.values).astype(np.float32)
    for name in ("geo_nbr_std", "geo_nbr_dS_std", "geo_radial_extrap_score"):
        if name in surface.diagnostics:
            values = surface.diagnostics[name]
            if len(values) != len(path.md):
                raise ValueError(
                    f"{well.well_id}: geographic diagnostic length differs from horizontal rows.",
                )
            horizontal[name] = values.copy()
    if "ANCC" in well.horizontal:
        formations = np.column_stack([well.horizontal[name] for name in FORMATIONS])
        for column in range(len(FORMATIONS)):
            valid = np.flatnonzero(~np.isnan(formations[:, column]))
            if len(valid):
                # Forward fill followed by backward fill keeps each formation's
                # first observed value through its leading missing rows.
                previous = np.maximum.accumulate(
                    np.where(
                        ~np.isnan(formations[:, column]),
                        np.arange(len(formations)),
                        -1,
                    ),
                )
                previous[previous < 0] = valid.flat[0]
                formations[:, column] = formations[previous, column]
        count = cast(
            "NDArray[np.intp]",
            np.count_nonzero(~np.isnan(formations), axis=1),
        )
        mean = np.full(len(formations), np.nan, dtype=np.float64)
        np.divide(np.nansum(formations, axis=1), count, out=mean, where=count > 0)
        horizontal["S_avg"] = mean.astype(np.float32)
    calibrated = calibration(
        well.horizontal["TVT_input"],
        gamma=well.horizontal["GR"],
        reference_depth=well.reference_depth,
        reference_gamma=well.reference_gamma,
    )
    reference_depth = well.reference_depth.astype(np.float32)
    order = np.argsort(reference_depth, kind="stable")
    return WellRecord(
        well_id=well.well_id,
        horizontal=horizontal,
        reference_depth=reference_depth[order],
        reference_gamma=calibrated.astype(np.float32)[order],
        particle={name: values.copy() for name, values in particle.items()}
        if particle is not None
        else None,
    )


class ParticleSource(Protocol):
    """Prepared complete particle records keyed by well."""

    def read(self, well_id: str) -> Mapping[str, NDArray[np.generic]]:
        """Return one well's complete sanitized particle record."""
        ...


class FoldObservations(Sequence[WellRecord]):
    """Rebuild complete fold observations from declared CSV/cache inputs."""

    class Config(Fig["FoldObservations"]):
        directory: Path | None = None
        """Complete staged horizontal/typewell CSV split holding the query wells; None
        leaves it to the family."""

        support_directory: Path | None = None
        """CSV split holding the support wells; None reads them beside the queries."""

        well_ids: tuple[str, ...] = ()
        """Complete query wells in loader order."""

        support_well_ids: tuple[str, ...] = ()
        """Training-fold donors in their original order."""

        geography: GeographicPrior.Config | None = None
        """Fold-safe relative-surface construction; None leaves it to the family."""

        leave_one_out: bool = True
        """Exclude each query from its own geographic donors."""

        calibration: ReferenceCalibration.Config = field(
            default_factory=ReferenceCalibration.Config,
        )
        """Visible-prefix calibration of each paired gamma-ray curve."""

        particle: Makeable[ParticleSource] | None = None
        """Prepared complete particle records, or None for plain image families."""

    def __init__(self, config: Config) -> None:
        if config.directory is None or config.geography is None:
            raise ValueError("Observations require their directory and geography.")
        support_directory = (
            config.directory
            if config.support_directory is None
            else config.support_directory
        )
        support = {
            name: load_raw(support_directory, name) for name in config.support_well_ids
        }
        queries = [
            support[name]
            if name in support and support_directory == config.directory
            else load_raw(config.directory, name)
            for name in config.well_ids
        ]
        prior = config.geography.make().build(
            [trajectory(support[name]) for name in config.support_well_ids],
            queries=[trajectory(well) for well in queries],
            leave_one_out=config.leave_one_out,
        )
        self.surfaces = prior
        source = config.particle.make() if config.particle is not None else None
        calibration = config.calibration.make()
        self.wells = tuple(
            prepare_well(
                well,
                surface=prior[well.well_id],
                calibration=calibration,
                particle=source.read(well.well_id) if source is not None else None,
            )
            for well in queries
        )

    @override
    def __len__(self) -> int:
        return len(self.wells)

    @overload
    def __getitem__(self, index: int) -> WellRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[WellRecord, ...]: ...

    @override
    def __getitem__(self, index: int | slice) -> WellRecord | tuple[WellRecord, ...]:
        return self.wells[index]


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionRows:
    """Complete original-row predictions in well and row iteration order."""

    well_ids: tuple[str, ...]
    submit_indices: NDArray[np.int64]
    tvt: NDArray[np.float32]
    z: NDArray[np.float32]
    target: NDArray[np.float32] | None


class ExpandPrediction:
    """Invert the public recipe's prefix bins and suffix truncation."""

    class Config(Fig["ExpandPrediction"]):
        prefix_rows: int = -1
        """Original rows reserved for the known prefix; -1 inherits the features."""

        target_rows: int = -1
        """Maximum original suffix rows represented by distinct model bins; -1
        inherits the features."""

        downsample: int = -1
        """Original rows represented by one horizontal model bin; -1 inherits the
        features."""

        target_stats: tuple[float, float] | None = None
        """Normalization center and scale of relative depth labels; None inherits
        the features."""

    def __init__(self, config: Config) -> None:
        if min(config.prefix_rows, config.target_rows, config.downsample) < 0:
            raise ValueError("Prediction expansion requires its window geometry.")
        if config.target_stats is None:
            raise ValueError("Prediction expansion requires its target statistics.")
        self.config = config
        self.target_mean, self.target_std = config.target_stats

    def expand(self, values: NDArray[np.generic], *, count: int) -> NDArray[np.float32]:
        """Repeat each horizontal bin over its original suffix rows.

        Args:
          values: One well's complete horizontal-bin values.
          count: Original suffix rows; rows beyond the target length repeat the last.

        Returns:
          expanded: One float32 value per original suffix row.

        """
        cfg = self.config
        kept = min(count, cfg.target_rows)
        bins = (cfg.prefix_rows + np.arange(kept)) // cfg.downsample
        expanded = np.empty(count, dtype=np.float32)
        expanded[:kept] = values[bins]
        if count > kept:
            expanded[kept:] = expanded[kept - 1]
        return expanded

    def __call__(
        self,
        prediction: NDArray[np.float32],
        *,
        metadata: Sequence[dict[str, object]],
        include_target: bool,
    ) -> PredictionRows:
        """Expand every original suffix row and retain official identifiers.

        Args:
          prediction: Complete normalized horizontal-bin predictions per well.
          metadata: Original suffix lengths, row identifiers, Z, and depth anchors.
          include_target: Retain original target depths for local scoring.

        Returns:
          rows: Complete suffix observations in the input well order.

        """
        wells: list[str] = []
        indices: list[NDArray[np.int64]] = []
        predictions: list[NDArray[np.float32]] = []
        heights: list[NDArray[np.float32]] = []
        targets: list[NDArray[np.float32]] = []
        for index, meta in zip(range(len(prediction)), metadata, strict=True):
            count = from_plain(meta["suffix_len"], int)
            relative = self.expand(prediction[index, :], count=count) * self.target_std
            relative += self.target_mean
            predictions.append(np.float32(from_plain(meta["tvt0"], float)) + relative)
            wells.extend([from_plain(meta["well_id"], str)] * count)
            indices.append(np.asarray(meta["submit_indices"], dtype=np.int64))
            heights.append(np.asarray(meta["suffix_z"], dtype=np.float32))
            if include_target:
                targets.append(np.asarray(meta["TVT"], dtype=np.float32))
        return PredictionRows(
            well_ids=tuple(wells),
            submit_indices=np.concatenate(indices),
            tvt=np.concatenate(predictions),
            z=np.concatenate(heights),
            target=np.concatenate(targets) if include_target else None,
        )


def _csv_float(value: str) -> float:
    """Parse one competition number to the nearest binary64 value."""
    if value in PANDAS_NA_VALUES:
        return float("nan")
    token = value.strip()
    if token.lower().lstrip("+-") in {"inf", "infinity", "nan"}:
        return float(token)
    match = re.fullmatch(
        r"([+-]?)([0-9]*)(?:\.([0-9]*))?(?:[eE]([+-]?[0-9]+))?",
        token,
    )
    if match is None or not (match[2] or match[3]):
        raise ValueError(f"Invalid numeric CSV observation: {value!r}.")
    number = float(token)
    if math.isinf(number):
        raise ValueError(f"Numeric CSV observation exceeds float64 range: {value!r}.")
    return number
