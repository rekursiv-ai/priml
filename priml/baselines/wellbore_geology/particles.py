"""Particle filtering priors, tuning and rebuildable particle caches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import json
import logging
import math

from configgle import Fig
from numpy.typing import NDArray
from treekle import to_plain
from wrapt import lazy_import

import numpy as np

from priml.baselines.wellbore_geology.data import (
    ReferenceCalibration,
    normalize_probability,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.lib.npyio import NpzFile

    import numba

    from priml.baselines.wellbore_geology.data import RawWell
    from priml.train.budget import WallBudget
else:
    numba = lazy_import("numba")

type Result = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]
type Kernel = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
    ],
    Result,
]


class JitableRegistration(Protocol):
    """The decorator-preserving public Numba registration boundary."""

    def register_jitable[F: Callable[..., object]](self, function: F, /) -> F:
        """Register a typed Python helper for calls from compiled kernels."""
        ...


@cache
def engine() -> Kernel:
    """Compile once, reusing the provisioned host cache across worker runs."""
    registration = cast("JitableRegistration", lazy_import("numba.extending"))
    registration.register_jitable(_uniform_numba)
    registration.register_jitable(_normal_numba)
    return numba.njit(cache=True)(_track_numba)


def _uniform_numba(
    state_array: NDArray[np.int64],
    cursor_array: NDArray[np.int64],
) -> float:
    """Advance the local MT19937 stream to its next 53-bit uniform."""
    state = state_array.flat
    cursor = cursor_array.flat
    words = np.empty(2, dtype=np.int64).flat
    for part in range(2):
        if cursor[0] == 624:
            for index in range(624):
                joined = (state[index] & 0x80000000) | (
                    state[(index + 1) % 624] & 0x7FFFFFFF
                )
                state[index] = (
                    state[(index + 397) % 624]
                    ^ (joined >> 1)
                    ^ (0x9908B0DF if joined & 1 else 0)
                )
            cursor[0] = 0
        value = state[cursor[0]]
        cursor[0] += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        words[part] = value
    return ((words[0] >> 5) * 67_108_864.0 + (words[1] >> 6)) * (
        1.0 / 9_007_199_254_740_992.0
    )


def _normal_numba(
    state: NDArray[np.int64],
    cursor_array: NDArray[np.int64],
    spare_array: NDArray[np.float64],
) -> float:
    """Preserve the cached polar-normal transform and its conditional draws."""
    cursor = cursor_array.flat
    spare = spare_array.flat
    if cursor[1]:
        cursor[1] = 0
        return float(spare[0])
    radius = 2.0
    first = 0.0
    second = 0.0
    while radius >= 1.0 or radius == 0.0:
        first = 2.0 * _uniform_numba(state, cursor_array) - 1.0
        second = 2.0 * _uniform_numba(state, cursor_array) - 1.0
        radius = first * first + second * second
    scale = np.float64(-2.0 * math.log(radius) / radius) ** np.float64(0.5)
    spare[0] = first * scale
    cursor[1] = 1
    return second * scale


# `observations` stores gamma, finite-GR flag, MD and Z. references contains calibrated
# and raw gamma on the shared regular depth grid. initial stores anchor, position, rate,
# previous MD, sigma and the grid/bounding coordinates. parameters stores the named
# values packed by ParticleDynamics (12), ParticleEmission (8) and ParticleTracking (7).
def _track_numba(
    observations: NDArray[np.float64],
    references: NDArray[np.float64],
    setup: NDArray[np.float64],
    parameters: NDArray[np.float64],
    dimensions: NDArray[np.int64],
) -> Result:
    """Track all merged observations and sample complete backward paths."""
    rows = int(dimensions.flat[0])
    steps = int(dimensions.flat[1])
    columns = int(dimensions.flat[2])
    particles = int(dimensions.flat[3])
    paths = int(dimensions.flat[4])
    seed = int(dimensions.flat[5])
    merge = int(dimensions.flat[6])
    smoothing_steps = int(dimensions.flat[7])
    initial = setup.flat
    motion = parameters[:12].flat
    emit = parameters[12:20].flat
    tuning = parameters[20:27].flat
    window = initial[9]
    observed = observations.flat
    reference_values = references.flat
    # MT19937 plus the cached polar normal transform define the public stream.
    # Keep it local to each run; NumPy's Generator uses a different normal map.
    state = np.empty(624, dtype=np.int64)
    state_cursor = np.asarray([624, 0], dtype=np.int64)
    spare = np.zeros(1)
    state.flat[0] = seed
    for word in range(1, 624):
        previous = state.flat[word - 1]
        state.flat[word] = (
            1_812_433_253 * (previous ^ (previous >> 30)) + word
        ) & 0xFFFFFFFF

    histogram = np.zeros((steps, columns))
    density = np.zeros((steps, columns))
    smoothed = np.zeros((steps, columns))
    path_array = np.zeros((steps, 2))
    hist = histogram.flat
    cloud = density.flat
    smooth = smoothed.flat
    trajectory = path_array.flat
    position = (np.empty(particles)).flat
    velocity = (np.empty(particles)).flat
    weights = (np.ones(particles) / particles).flat
    snapshots_pos = np.empty((steps, particles)).flat
    snapshots_rate = np.empty((steps, particles)).flat
    snapshots_weight = np.empty((steps, particles)).flat
    powers = np.empty(64).flat
    cumulative = (np.empty(particles)).flat
    replacement_pos = (np.empty(particles)).flat
    replacement_rate = (np.empty(particles)).flat
    categorical = (np.empty(particles)).flat
    anchor = initial[0]
    axis_start = -window + window / columns
    axis_step = 2.0 * window / columns
    for particle in range(particles):
        quantile = (particle + _uniform_numba(state, state_cursor)) / particles
        position[particle] = (
            initial[1] + motion[6] * (2.0 * quantile - 1.0) * 1.7320508075688772
        )
        quantile = (
            (particle * 37) % particles + _uniform_numba(state, state_cursor)
        ) / particles
        velocity[particle] = (
            initial[2] + motion[7] * (2.0 * quantile - 1.0) * 1.7320508075688772
        )
    previous_md = initial[3]
    shared_noise = 0.0
    finite_run = 0.0
    log_evidence = 0.0
    for step in range(steps):
        begin = step * merge
        end = min(begin + merge, rows)
        width = end - begin
        first_md = observed[(begin) * 4 + (2)]
        first_z = observed[(begin) * 4 + (3)]
        distance = max(first_md - previous_md, 1.0)
        shared_noise = tuning[0] * shared_noise + tuning[1] * _normal_numba(
            state,
            state_cursor,
            spare,
        )
        jump_probability = min(motion[9] * distance, 1.0)
        for particle in range(particles):
            velocity[particle] = motion[0] * velocity[particle] + motion[
                1
            ] * _normal_numba(state, state_cursor, spare)
            velocity[particle] += motion[8] * (initial[2] - velocity[particle])
            perturbation = (
                0.5 * _normal_numba(state, state_cursor, spare)
                + 0.8660254037844386 * shared_noise
            )
            position[particle] = (
                position[particle]
                + velocity[particle] * distance
                + motion[2] * perturbation
            )
            if _uniform_numba(state, state_cursor) < jump_probability:
                position[particle] += motion[10] * _normal_numba(
                    state,
                    state_cursor,
                    spare,
                )
                velocity[particle] += motion[11] * _normal_numba(
                    state,
                    state_cursor,
                    spare,
                )
            depth = min(max(position[particle] - first_z, initial[6]), initial[7])
            position[particle] = depth + first_z
        merged_power = np.float64(width) ** emit[0]
        base_power = merged_power / width
        original_power = 0.0
        adjusted_power = 0.0
        for offset in range(width):
            original_power += base_power
            if observed[(begin + offset) * 4 + (1)] > 0.0:
                finite_run += 1.0
            else:
                finite_run = 0.0
            power = base_power
            if finite_run > 1.0:
                power *= 1.0 + emit[1] * math.log(1.0 + finite_run)
                power = min(power, 1.0)
                power *= max(
                    1.0 / np.float64(1.0 + (finite_run - 1.0)) ** np.float64(0.5),
                    emit[2],
                )
            powers[offset] = power
            adjusted_power += power
        power_scale = merged_power / original_power
        effective_power = adjusted_power * power_scale
        for offset in range(width):
            powers[offset] *= power_scale
        evidence = 0.0
        weight_sum = 0.0
        for particle in range(particles):
            first_depth = position[particle] - first_z
            likelihood = 1.0
            for offset in range(width):
                row = begin + offset
                depth = (
                    first_depth
                    + velocity[particle] * (observed[(row) * 4 + (2)] - first_md)
                    + (first_z - observed[(row) * 4 + (3)])
                )
                emission = 0.0
                for reference in range(2):
                    coordinate = (depth - initial[5]) / emit[7]
                    left = int(coordinate)
                    last = int(initial[8]) - 1
                    if left < 0:
                        expected = reference_values[(reference) * int(initial[8]) + (0)]
                    elif left >= last:
                        expected = reference_values[
                            (reference) * int(initial[8]) + (last)
                        ]
                    else:
                        fraction = coordinate - left
                        expected = (
                            reference_values[(reference) * int(initial[8]) + (left)]
                            * (1.0 - fraction)
                            + reference_values[
                                (reference) * int(initial[8]) + (left + 1)
                            ]
                            * fraction
                        )
                    residual = (observed[(row) * 4 + (0)] - expected) / initial[4]
                    probability = math.exp(
                        -0.5 * min(residual * residual, 600.0) * powers[offset],
                    )
                    emission += (emit[3] if reference == 0 else emit[4]) * probability
                likelihood *= max(emission, 1e-300)
            for offset in range(width):
                row = begin + offset
                depth = (
                    first_depth
                    + velocity[particle] * (observed[(row) * 4 + (2)] - first_md)
                    + (first_z - observed[(row) * 4 + (3)])
                )
                residual = (depth - anchor) / emit[5]
                likelihood *= math.exp(
                    -0.5 * min(residual * residual, 600.0) * emit[6],
                )
            likelihood = max(likelihood, 1e-300)
            evidence += weights[particle] * likelihood
            weights[particle] *= likelihood
            weight_sum += weights[particle]
        log_evidence += math.log(max(evidence, 1e-300))
        output_md_delta = observed[(end - 1) * 4 + (2)] - first_md
        output_z = observed[(end - 1) * 4 + (3)]
        inverse_ess = 0.0
        for particle in range(particles):
            weights[particle] /= weight_sum
            depth = position[particle] - first_z
            depth = depth + velocity[particle] * output_md_delta + (first_z - output_z)
            depth = min(max(depth, initial[6]), initial[7])
            trajectory[(step) * 2 + (0)] += weights[particle] * depth
            coordinate = (depth - anchor - axis_start) / axis_step
            left = math.floor(coordinate)
            fraction = coordinate - left
            if 0 <= left < columns:
                hist[step * columns + left] += weights[particle] * (1.0 - fraction)
                cloud[step * columns + left] += 1.0 - fraction
            if 0 <= left + 1 < columns:
                hist[step * columns + left + 1] += weights[particle] * fraction
                cloud[step * columns + left + 1] += fraction
            snapshots_pos[step * particles + particle] = position[particle]
            snapshots_rate[step * particles + particle] = velocity[particle]
            snapshots_weight[step * particles + particle] = weights[particle]
            inverse_ess += weights[particle] * weights[particle]
        threshold = max(
            motion[3] * (1.0 / (1.0 + tuning[2] * effective_power)),
            tuning[3],
        )
        if 1.0 / inverse_ess < threshold * particles:
            total = 0.0
            for particle in range(particles):
                total += weights[particle]
                cumulative[particle] = total
            start = _uniform_numba(state, state_cursor) * (1.0 / particles)
            cursor = 0
            for particle in range(particles):
                quantile = start + particle / particles
                while cursor < particles - 1 and cumulative[cursor] < quantile:
                    cursor += 1
                replacement_pos[particle] = position[cursor] + motion[
                    4
                ] * _normal_numba(state, state_cursor, spare)
                replacement_rate[particle] = velocity[cursor] + motion[
                    5
                ] * _normal_numba(state, state_cursor, spare)
            for particle in range(particles):
                depth = min(
                    max(replacement_pos[particle] - first_z, initial[6]),
                    initial[7],
                )
                position[particle] = depth + first_z
                velocity[particle] = replacement_rate[particle]
                weights[particle] = 1.0 / particles
        previous_md = first_md
    if 1 < steps <= smoothing_steps:
        for _ in range(paths):
            next_position = 0.0
            next_velocity = 0.0
            for step in range(steps - 1, -1, -1):
                begin = step * merge
                end = min(begin + merge, rows)
                total = 0.0
                if step == steps - 1:
                    for particle in range(particles):
                        categorical[particle] = snapshots_weight[
                            step * particles + particle
                        ]
                        total += categorical[particle]
                else:
                    distance = max(
                        observed[(end) * 4 + (2)] - observed[(begin) * 4 + (2)],
                        1.0,
                    )
                    jump = min(motion[9] * distance, 1.0)
                    rate_variance = (
                        motion[1] * motion[1] + jump * motion[11] * motion[11]
                    )
                    pos_variance = (
                        motion[2] * motion[2]
                        + distance * distance * motion[1] * motion[1]
                        + jump * motion[10] * motion[10]
                    )
                    position_scale = max(
                        np.float64(max(pos_variance, 1e-18)) ** np.float64(0.5)
                        * tuning[4],
                        tuning[5],
                    )
                    velocity_scale = max(
                        np.float64(max(rate_variance, 1e-18)) ** np.float64(0.5)
                        * tuning[4],
                        tuning[6],
                    )
                    best = -1e300
                    for particle in range(particles):
                        predicted_rate = (
                            motion[0] * snapshots_rate[step * particles + particle]
                        )
                        predicted_rate += motion[8] * (initial[2] - predicted_rate)
                        predicted_position = (
                            snapshots_pos[step * particles + particle]
                            + predicted_rate * distance
                        )
                        rate_error = (next_velocity - predicted_rate) / velocity_scale
                        pos_error = (
                            next_position - predicted_position
                        ) / position_scale
                        score = (
                            math.log(
                                max(
                                    snapshots_weight[step * particles + particle],
                                    1e-300,
                                ),
                            )
                            + max(-0.5 * rate_error * rate_error, -300.0)
                            + max(-0.5 * pos_error * pos_error, -300.0)
                        )
                        categorical[particle] = score
                        best = max(best, score)
                    for particle in range(particles):
                        categorical[particle] = math.exp(categorical[particle] - best)
                        total += categorical[particle]
                quantile = _uniform_numba(state, state_cursor) * total
                cumulative_probability = 0.0
                selected = particles - 1
                for particle in range(particles):
                    cumulative_probability += categorical[particle]
                    if cumulative_probability >= quantile:
                        selected = particle
                        break
                next_position = snapshots_pos[step * particles + selected]
                next_velocity = snapshots_rate[step * particles + selected]
                first_z = observed[(begin) * 4 + (3)]
                depth = next_position - first_z
                depth = (
                    depth
                    + next_velocity
                    * (observed[(end - 1) * 4 + (2)] - observed[(begin) * 4 + (2)])
                    + (first_z - observed[(end - 1) * 4 + (3)])
                )
                depth = min(max(depth, initial[6]), initial[7])
                trajectory[(step) * 2 + (1)] += depth
                coordinate = (depth - anchor - axis_start) / axis_step
                left = math.floor(coordinate)
                fraction = coordinate - left
                if 0 <= left < columns:
                    smooth[step * columns + left] += 1.0 - fraction
                if 0 <= left + 1 < columns:
                    smooth[step * columns + left + 1] += fraction

        for step in range(steps):
            trajectory[(step) * 2 + (1)] /= paths
    else:
        smoothed[:] = histogram
        path_array[:, 1] = path_array[:, 0]
    return histogram, density, smoothed, path_array, log_evidence


class ParticleEmission:
    """Gamma-ray emission and anchor-prior constants of one merged block."""

    class Config(Fig["ParticleEmission"]):
        power_exponent: float = 1.05
        """Exponent of a merged block's observation power in its row count."""

        run_gain: float = 0.06
        """Logarithmic power gain per consecutive finite gamma-ray row."""

        run_floor: float = 0.6
        """Smallest inverse-square-root discount of a long finite run."""

        primary_weight: float = 0.925
        """Emission mass on the calibrated paired curve."""

        raw_weight: float = 0.075
        """Emission mass on the raw paired curve."""

        anchor_scale: float = 80.0
        """Physical depth scale of the anchor prior."""

        anchor_strength: float = 0.002
        """Weight of the anchor prior in the log likelihood."""

        grid_step: float = 0.05
        """Spacing of the regular paired-curve depth grid."""

    def __init__(self, config: Config) -> None:
        self.grid_step = config.grid_step
        self.values = np.asarray(
            [
                config.power_exponent,
                config.run_gain,
                config.run_floor,
                config.primary_weight,
                config.raw_weight,
                config.anchor_scale,
                config.anchor_strength,
                config.grid_step,
            ],
            dtype=np.float64,
        )


class ParticleTracking:
    """Merge, resampling, backward-simulation and calibration constants."""

    class Config(Fig["ParticleTracking"]):
        merge_rows: int = 64
        """Original suffix rows merged into one filtering step."""

        smoothing_max_steps: int = 512
        """Largest step count that still receives backward trajectories."""

        noise_correlation: float = 0.2
        """Autoregression of the noise shared by every particle."""

        resample_decay: float = 0.45
        """Effective-observation discount of the resampling threshold."""

        resample_floor: float = 0.3
        """Smallest effective cloud fraction that triggers resampling."""

        backward_scale: float = 1.35
        """Inflation of the backward transition widths."""

        position_floor: float = 0.15
        """Smallest backward position width."""

        velocity_floor: float = 0.0025
        """Smallest backward velocity width."""

        calibration: ReferenceCalibration.Config = field(
            default_factory=lambda: ReferenceCalibration.Config(
                blend=0.75,
                raw_weight=0.25,
            ),
        )
        """Visible-prefix calibration of the curve the particles are scored against."""

        sigma_bounds: tuple[float, float] = (10.0, 60.0)
        """Lower and upper clip of the visible-prefix gamma mismatch scale."""

        rate_rows: int = 30
        """Trailing visible rows estimating the initial vertical rate."""

        rate_min_moving: int = 3
        """Fewest advancing rows from which an initial rate is estimated."""

        reference_pad: float = 100.0
        """Depth margin around the paired curve bounding the particles."""

        evidence_temperature: float = 6.0
        """Log-evidence temperature weighting seeds within a profile."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.calibration = config.calibration.make()
        correlation = config.noise_correlation
        self.values = np.asarray(
            [
                correlation,
                (max(0.0, 1.0 - correlation * correlation)) ** 0.5,
                config.resample_decay,
                config.resample_floor,
                config.backward_scale,
                config.position_floor,
                config.velocity_floor,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ParticleWell:
    """Complete raw float64 observations before any model-data calibration."""

    md: NDArray[np.float64]
    z: NDArray[np.float64]
    gamma: NDArray[np.float64]
    known_depth: NDArray[np.float64]
    depth: NDArray[np.float64] | None
    reference_depth: NDArray[np.float64]
    reference_gamma: NDArray[np.float64]


class ParticleDynamics:
    """One independently seeded transition profile and its prior mixture mass."""

    class Config(Fig["ParticleDynamics"]):
        weight: float = 0.6514002115702939
        """Prior profile mass, normalized before seed allocation."""

        momentum: float = 0.9995
        """Velocity autoregression coefficient."""

        rate_noise: float = 0.0009
        """Independent velocity innovation standard deviation."""

        position_noise: float = 0.0025
        """Position innovation standard deviation."""

        resample_threshold: float = 0.3
        """Effective cloud fraction triggering systematic resampling."""

        rough_position: float = 0.08
        """Resampling position jitter standard deviation."""

        rough_rate: float = 0.006
        """Resampling velocity jitter standard deviation."""

        initial_position: float = 2.0
        """Stratified initial-position standard deviation."""

        initial_rate: float = 0.02
        """Stratified initial-velocity standard deviation."""

        mean_reversion: float = 0.001
        """Velocity shrinkage toward the visible-prefix rate."""

        jump_probability: float = 7.5e-5
        """Rare joint position/velocity jump probability per MD unit."""

        jump_position: float = 7.0
        """Rare position jump standard deviation."""

        jump_rate: float = 0.0003
        """Rare velocity jump standard deviation."""

    def __init__(self, config: Config) -> None:
        self.weight = config.weight
        self.values = np.asarray(
            [
                config.momentum,
                config.rate_noise,
                config.position_noise,
                config.resample_threshold,
                config.rough_position,
                config.rough_rate,
                config.initial_position,
                config.initial_rate,
                config.mean_reversion,
                config.jump_probability,
                config.jump_position,
                config.jump_rate,
            ],
            dtype=np.float64,
        )


def profile_configs() -> list[ParticleDynamics.Config]:
    """Declare the five public transition distributions in their original order.

    Returns:
      profiles: Complete ordered dynamics with only profile deltas overridden.

    """
    baseline = ParticleDynamics.Config()
    wide = ParticleDynamics.Config()
    wide.weight = 0.19545809861466074
    wide.initial_position = 2.5
    wide.initial_rate = 0.025
    wide.resample_threshold = 0.35
    wide.rough_position = 0.1
    smooth = ParticleDynamics.Config()
    smooth.weight = 0.053120440286232196
    smooth.jump_position = 9.0
    smooth.position_noise = 0.003
    smooth.mean_reversion = 0.0015
    smooth.rate_noise = 0.0008
    smooth.rough_position = 0.06
    smooth.rough_rate = 0.0045
    rare = ParticleDynamics.Config()
    rare.weight = 0.07370294173329474
    rare.jump_rate = 0.0006
    rare.jump_position = 10.0
    rare.resample_threshold = 0.35
    tight = ParticleDynamics.Config()
    tight.weight = 0.02631830779551842
    tight.initial_position = 1.0
    tight.initial_rate = 0.015
    tight.position_noise = 0.0035
    tight.rough_position = 0.06
    return [baseline, wide, smooth, rare, tight]


class ParticlePrior:
    """Retain every active public filtering, mixture and FFBSi stage."""

    class Config(Fig["ParticlePrior"]):
        prefix_rows: int = -1
        """Visible rows aligned before the cached prediction boundary; -1 inherits
        the family's features."""

        target_rows: int = -1
        """Maximum suffix rows traced by the complete particle producer; -1
        inherits the family's features."""

        downsample: int = -1
        """Original rows represented by one cached feature row; -1 inherits the
        family's features."""

        reference_columns: int = -1
        """Complete posterior depth-axis width; -1 inherits the family's features."""

        reference_window: float = -1.0
        """Physical posterior half width around the last visible depth; negative
        inherits the family's features."""

        particles: int = 500
        """Complete particle cloud size for each seeded profile run."""

        seeds: int = 128
        """Complete seed budget allocated among all five profiles."""

        paths: int = 64
        """Complete sampled backward trajectories for each profile run."""

        base_seed: int = 20_265_667
        """Independent generator seed for seeded runs and profile allocation."""

        profiles: list[ParticleDynamics.Config] = field(default_factory=profile_configs)
        """Ordered transition profiles, including their allocation weights."""

        emission: ParticleEmission.Config = field(
            default_factory=ParticleEmission.Config,
        )
        """Gamma-ray emission, anchor prior and reference-grid constants."""

        tracking: ParticleTracking.Config = field(
            default_factory=ParticleTracking.Config,
        )
        """Block merge, resampling, backward-simulation and calibration constants."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.profiles = [profile.make() for profile in config.profiles]
        self.emission = config.emission.make()
        self.tracking = config.tracking.make()
        if (
            min(
                config.prefix_rows,
                config.target_rows,
                config.downsample,
                config.reference_columns,
                config.reference_window,
            )
            < 0
        ):
            raise ValueError("Particle tracking requires its window geometry.")
        if config.tracking.merge_rows < 1:
            raise ValueError("Particle filtering requires a positive row merge.")
        if min(config.particles, config.seeds, config.paths) < 1:
            raise ValueError(
                "Particle, seed and backward-path budgets must be positive.",
            )
        self.weights = np.asarray(
            [
                profile.weight / sum(p.weight for p in self.profiles)
                for profile in self.profiles
            ],
            dtype=np.float64,
        )
        self.weights /= self.weights.sum()

    def __call__(
        self,
        well: ParticleWell,
        *,
        has_target: bool,
    ) -> dict[str, NDArray[np.generic]]:
        """Build complete filtering/smoothing arrays and every cache diagnostic."""
        config = self.config
        visible = np.flatnonzero(np.isfinite(well.known_depth))
        if len(visible) == 0:
            raise ValueError("Particle tracking requires a visible depth anchor.")
        last = int(visible.flat[-1])
        anchor = float(well.known_depth.flat[last])
        suffix_rows = len(well.md) - last - 1
        kept = min(suffix_rows, config.target_rows)
        bins = (config.prefix_rows + config.target_rows) // config.downsample + 1
        record = self._window(well, last=last, anchor=anchor, has_target=has_target)
        posteriors = [
            np.zeros((bins, config.reference_columns), dtype=np.float32)
            for _ in range(3)
        ]
        predictions = [np.zeros(bins, dtype=np.float32) for _ in range(2)]
        if kept > 0:
            probabilities, paths = self._suffix(
                well,
                last=last,
                anchor=anchor,
                rows=kept,
            )
            for index, probability in enumerate(probabilities):
                posteriors[index] = self._expand_probability(
                    probability,
                    rows=kept,
                    bins=bins,
                )
            for index, path in enumerate(paths):
                predictions[index] = self._expand_path(
                    path,
                    anchor=anchor,
                    rows=kept,
                    bins=bins,
                )
        relative = [
            (prediction - np.float32(anchor)).astype(np.float32)
            for prediction in predictions
        ]
        if kept == 0:
            relative = [np.zeros(bins, dtype=np.float32) for _ in predictions]
        for arrays in (predictions, relative):
            for index, values in enumerate(arrays):
                arrays[index] = np.nan_to_num(values, nan=0.0).astype(np.float32)
        fixed = record["window_has_tvt"].astype(np.bool_) & ~record[
            "target_mask"
        ].astype(
            np.bool_,
        )
        fixed_relative = record["window_tvt_rel"].astype(np.float32)
        for row in np.flatnonzero(fixed).flat:
            delta = float(fixed_relative.flat[row])
            coordinate = (
                delta
                + config.reference_window
                - config.reference_window / config.reference_columns
            ) / (2 * config.reference_window / config.reference_columns)
            left = math.floor(coordinate)
            fraction = coordinate - left
            mass = np.zeros(config.reference_columns, dtype=np.float32)
            if 0 <= left < config.reference_columns:
                mass[left] += np.float32(1 - fraction)
            if 0 <= left + 1 < config.reference_columns:
                mass[left + 1] += np.float32(fraction)
            total = float(mass.sum())
            if total > 0:
                mass /= np.float32(total)
                for posterior in posteriors:
                    posterior[row] = mass
                for prediction, rel in zip(predictions, relative, strict=True):
                    prediction[row] = np.float32(anchor + delta)
                    rel[row] = np.float32(delta)
        record.update(
            last_seen_idx=np.asarray(last),
            tvt0=np.asarray(anchor),
            prefix_start=np.asarray(max(0, last + 1 - config.prefix_rows)),
            suffix_len=np.asarray(suffix_rows),
            pf_prob=posteriors[0],
            pf_particle_density_prob=posteriors[1],
            pf_prob_ffbsi=posteriors[2],
            pf_tvt_pred=predictions[0],
            pf_tvt_rel_pred=relative[0],
            pf_tvt_pred_ffbsi=predictions[1],
            pf_tvt_rel_pred_ffbsi=relative[1],
        )
        return record

    def _window(
        self,
        well: ParticleWell,
        *,
        last: int,
        anchor: float,
        has_target: bool,
    ) -> dict[str, NDArray[np.generic]]:
        config = self.config
        length = config.prefix_rows + config.target_rows
        bins = length // config.downsample + 1
        raw = np.full(length, np.nan, dtype=np.float32)
        original = np.full(length, np.nan, dtype=np.float32)
        target = np.zeros(length, dtype=bool)
        start = max(0, last + 1 - config.prefix_rows)
        prefix_size = last + 1 - start
        destination = slice(config.prefix_rows - prefix_size, config.prefix_rows)
        raw[destination] = well.known_depth[start : last + 1].astype(np.float32)
        original[destination] = np.arange(start, last + 1, dtype=np.float32)
        kept = min(len(well.md) - last - 1, config.target_rows)
        destination = slice(config.prefix_rows, config.prefix_rows + kept)
        original[destination] = np.arange(last + 1, last + 1 + kept, dtype=np.float32)
        if has_target and well.depth is not None:
            raw[destination] = well.depth[last + 1 : last + 1 + kept].astype(np.float32)
            target[destination] = True
        indices = np.arange(length, dtype=np.int64) // config.downsample
        means: list[NDArray[np.float32]] = []
        for values in (raw, original):
            valid = np.isfinite(values)
            sums = np.bincount(indices[valid], weights=values[valid], minlength=bins)
            counts = np.bincount(indices[valid], minlength=bins)
            mean = np.full(bins, np.nan, dtype=np.float32)
            filled = counts > 0
            mean[filled] = (sums[filled] / counts[filled]).astype(np.float32)
            means.append(mean)
        return {
            "window_tvt": np.nan_to_num(means[0], nan=0.0),
            "window_tvt_rel": np.nan_to_num(means[0] - np.float32(anchor), nan=0.0),
            "window_has_tvt": np.isfinite(means[0]),
            "window_orig_index": np.nan_to_num(means[1], nan=-1.0),
            "target_mask": np.bincount(indices[target], minlength=bins) > 0,
        }

    def _allocation(self) -> NDArray[np.int64]:
        budget = self.config.seeds
        counts = np.zeros(len(self.profiles), dtype=np.int64)
        if budget >= len(counts):
            counts[:] = 1
        expected = self.weights * (budget - int(counts.sum()))
        counts += np.floor(expected).astype(np.int64)
        remaining = budget - int(counts.sum())
        priority = np.argsort(expected - np.floor(expected), kind="stable")[::-1]
        counts[priority[:remaining]] += 1
        result = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        np.random.default_rng(self.config.base_seed + 100_003 * budget).shuffle(result)
        return result

    def _suffix(
        self,
        well: ParticleWell,
        *,
        last: int,
        anchor: float,
        rows: int,
    ) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        config = self.config
        merge = config.tracking.merge_rows
        steps = (rows + merge - 1) // merge
        known = (
            np.isfinite(well.md[: last + 1])
            & np.isfinite(well.z[: last + 1])
            & np.isfinite(well.known_depth[: last + 1])
        )
        if np.count_nonzero(known) < 2 or len(well.reference_depth) < 3:
            return (
                [
                    np.zeros((steps, config.reference_columns), dtype=np.float32)
                    for _ in range(3)
                ],
                [np.full(steps, np.nan, dtype=np.float32) for _ in range(2)],
            )
        order = np.argsort(well.reference_depth, kind="stable")
        depth = well.reference_depth[order]
        raw_curve = well.reference_gamma[order].copy()
        primary = self.tracking.calibration(
            well.known_depth,
            gamma=well.gamma,
            reference_depth=depth,
            reference_gamma=raw_curve,
        )
        for curve in (raw_curve, primary):
            curve[np.isnan(curve)] = np.nanmean(curve)
        reference_min = float(depth.flat[0])
        reference_max = float(depth.flat[-1])
        step = self.emission.grid_step
        grid = np.arange(reference_min, reference_max + step, step, dtype=np.float64)
        references = np.stack(
            [np.interp(grid, depth, primary), np.interp(grid, depth, raw_curve)],
        )
        finite_gamma = np.flatnonzero(np.isfinite(well.gamma))
        query_gamma = (
            np.interp(
                np.arange(len(well.gamma)),
                finite_gamma,
                well.gamma[finite_gamma],
            )
            if len(finite_gamma)
            else np.full(len(well.gamma), float(np.nanmean(primary)))
        )
        destination = slice(last + 1, last + 1 + rows)
        observations = np.column_stack(
            [
                query_gamma[destination],
                np.isfinite(well.gamma[destination]).astype(np.float64),
                well.md[destination],
                well.z[destination],
            ],
        )
        md = well.md[: last + 1][known]
        z = well.z[: last + 1][known]
        visible = well.known_depth[: last + 1][known]
        observed = np.nan_to_num(well.gamma[: last + 1], nan=0.0)[known]
        sigma = float(
            np.clip(
                np.nanstd(observed - np.interp(visible, depth, primary)),
                *config.tracking.sigma_bounds,
            ),
        )
        recent = -config.tracking.rate_rows
        delta_md = np.diff(md[recent:])
        moving = delta_md > 0
        rate = (
            float(
                np.median(
                    (np.diff(visible[recent:]) + np.diff(z[recent:]))[moving]
                    / delta_md[moving],
                ),
            )
            if np.count_nonzero(moving) >= config.tracking.rate_min_moving
            else 0.0
        )
        initial = np.asarray(
            [
                anchor,
                float(visible.flat[-1] + z.flat[-1]),
                rate,
                float(md.flat[-1]),
                sigma,
                reference_min,
                reference_min - config.tracking.reference_pad,
                reference_max + config.tracking.reference_pad,
                float(len(grid)),
                config.reference_window,
            ],
            dtype=np.float64,
        )
        allocation = self._allocation()
        seeds = np.random.default_rng(config.base_seed).integers(
            0,
            np.iinfo(np.int32).max,
            size=config.seeds,
            dtype=np.int64,
        )
        probabilities: list[list[NDArray[np.float32]]] = [[], [], []]
        paths: list[list[NDArray[np.float32]]] = [[], []]
        evidence = np.empty(config.seeds, dtype=np.float64)
        for index, seed in enumerate(seeds.flat):
            profile = self.profiles[int(allocation.flat[index])]
            filtered, density, smooth, path, score = engine()(
                observations,
                references,
                initial,
                np.concatenate(
                    (profile.values, self.emission.values, self.tracking.values),
                ),
                np.asarray(
                    [
                        rows,
                        steps,
                        config.reference_columns,
                        config.particles,
                        config.paths,
                        int(seed),
                        merge,
                        config.tracking.smoothing_max_steps,
                    ],
                    dtype=np.int64,
                ),
            )
            evidence[index] = score
            smoothed = 1 < steps <= config.tracking.smoothing_max_steps
            for target, values in zip(
                probabilities,
                (filtered, density, smooth / config.paths if smoothed else smooth),
                strict=True,
            ):
                target.append(normalize_probability(values.astype(np.float32)))
            for target, values in zip(paths, (path[:, 0], path[:, 1]), strict=True):
                target.append(values.astype(np.float32))
        weights = np.zeros(config.seeds, dtype=np.float64)
        for profile_index, prior in enumerate(self.weights.flat):
            mask = cast("NDArray[np.bool_]", allocation == profile_index)
            if np.count_nonzero(mask) == 0:
                continue
            local = np.exp(
                (evidence[mask] - evidence[mask].max())
                / config.tracking.evidence_temperature,
            )
            weights[mask] = prior * (local / local.sum())
        weights /= weights.sum()
        mixed: list[NDArray[np.float32]] = []
        for index, records in enumerate(probabilities):
            stack = np.stack(records)
            local_weights = (
                np.full(config.seeds, 1 / config.seeds, dtype=np.float32)
                if index == 1
                else weights
            )
            mixed.append(
                normalize_probability(
                    (stack * local_weights[:, None, None])
                    .sum(axis=0)
                    .astype(np.float32),
                ),
            )
        mixed_paths: list[NDArray[np.float32]] = []
        for records in paths:
            stack = np.stack(records).astype(np.float64)
            finite = np.isfinite(stack)
            weighted = np.where(finite, stack, 0.0) * weights[:, None]
            denominator = finite.astype(np.float64).T @ weights
            result = np.full(steps, np.nan, dtype=np.float32)
            np.divide(
                weighted.sum(axis=0),
                denominator,
                out=result,
                where=denominator > 0,
            )
            mixed_paths.append(result)
        return mixed, mixed_paths

    def _expand_probability(
        self,
        values: NDArray[np.float32],
        *,
        rows: int,
        bins: int,
    ) -> NDArray[np.float32]:
        merge = self.config.tracking.merge_rows
        result = np.zeros((bins, self.config.reference_columns), dtype=np.float32)
        counts = np.zeros(bins, dtype=np.float32)
        for step in range(len(values)):
            value = cast("NDArray[np.float32]", values[step])
            indices = (
                self.config.prefix_rows
                + np.arange(step * merge, min(step * merge + merge, rows))
            ) // self.config.downsample
            unique, sizes = np.unique(indices, return_counts=True)
            for index, size in zip(unique.flat, sizes.flat, strict=True):
                result[index] += np.float32(size) * value
                counts[index] += np.float32(size)
        valid = counts > 0
        result[valid] /= counts[valid, None]
        return normalize_probability(result)

    def _expand_path(
        self,
        values: NDArray[np.float32],
        *,
        anchor: float,
        rows: int,
        bins: int,
    ) -> NDArray[np.float32]:
        merge = self.config.tracking.merge_rows
        raw = np.full(rows, np.nan, dtype=np.float32)
        previous = anchor
        for step, value in enumerate(values.flat):
            begin = step * merge
            end = min(begin + merge, rows)
            current = float(value) if np.isfinite(value) else previous
            delta = (current - previous) / (end - begin)
            for offset in range(end - begin):
                raw[begin + offset] = np.float32(previous + delta * (offset + 1))
            previous = current
        indices = (self.config.prefix_rows + np.arange(rows)) // self.config.downsample
        sums = np.bincount(indices, weights=raw, minlength=bins)
        counts = np.bincount(indices, minlength=bins)
        result = np.full(bins, np.nan, dtype=np.float32)
        filled = counts > 0
        result[filled] = (sums[filled] / counts[filled]).astype(np.float32)
        return result


logger = logging.getLogger(__name__)

PROBABILITIES: Final = ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi")
"""All complete posterior matrices quantized by the released cache writer."""

PATHS: Final = (
    "pf_tvt_pred",
    "pf_tvt_rel_pred",
    "pf_tvt_pred_ffbsi",
    "pf_tvt_rel_pred_ffbsi",
)
"""All complete absolute and relative filtered/backward depth paths."""

WINDOWS: Final = ("window_tvt", "window_tvt_rel", "window_orig_index")
"""Complete original-row alignment observations retained in each particle cache."""

FLAGS: Final = ("window_has_tvt", "target_mask")
"""Complete original visibility and target masks."""

COUNTERS: Final = ("last_seen_idx", "prefix_start", "suffix_len")
"""Whole raw-row cache boundary counters."""


class ParticleCache:
    """Read prepared per-well particle records from one family and split directory."""

    class Config(Fig["ParticleCache"]):
        directory: Path | None = None
        """Family and split specific rebuildable cache holding one file per well;
        None leaves it to the family."""

    def __init__(self, config: Config) -> None:
        if config.directory is None:
            raise ValueError("A particle cache requires its directory.")
        self.directory = config.directory

    def read(self, well_id: str) -> dict[str, NDArray[np.generic]]:
        """Return one well's complete sanitized particle record."""
        return read_particle_cache(self.directory / f"{well_id}.npz")


def prepare_particles(
    wells: Mapping[str, RawWell],
    *,
    config: ParticlePrior.Config,
    directory: Path,
    has_target: bool,
    budget: WallBudget | None = None,
) -> dict[str, dict[str, NDArray[np.generic]]]:
    """Generate or validate every complete per-well filtering and FFBSi record.

    Args:
      wells: Complete raw observations in declared well order.
      config: Every physical and Monte Carlo parameter for this family.
      directory: External family and split specific rebuildable cache.
      has_target: Retain original target observations in the complete record.
      budget: Optional separate experiment's remaining cold wall allowance.

    Returns:
      records: Every complete cache after its public float16 IO boundary.

    """
    directory = validated_output_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "config.json"
    encoded = _manifest_text(config)
    if manifest.exists():
        if manifest.read_text() != encoded:
            raise ValueError("Particle cache configuration differs from this recipe.")
    else:
        if next(directory.iterdir(), None) is not None:
            raise FileExistsError(
                "Particle cache requires its native configuration manifest.",
            )
        manifest.write_text(encoded)
    producer = config.make()
    records: dict[str, dict[str, NDArray[np.generic]]] = {}
    for name, raw in wells.items():
        if budget is not None:
            budget.training_seconds()
        path = directory / f"{name}.npz"
        if not path.exists():
            write_particle_cache(
                path,
                producer(particle_well(raw), has_target=has_target),
            )
        records[name] = read_particle_cache(path)
    return records


def validate_particles(
    names: Iterable[str],
    *,
    config: ParticlePrior.Config,
    directory: Path,
) -> None:
    """Require a complete prepared cache that matches this recipe exactly.

    Args:
      names: Every well whose record the run will read.
      config: Every physical and Monte Carlo parameter for this family.
      directory: External family and split specific rebuildable cache.

    Raises:
      FileNotFoundError: The manifest or a well record is missing.
      ValueError: The manifest was written by a different recipe.

    """
    manifest = directory / "config.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Particle cache is not prepared: {directory}.")
    if manifest.read_text() != _manifest_text(config):
        raise ValueError("Particle cache configuration differs from this recipe.")
    missing = [name for name in names if not (directory / f"{name}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Particle cache lacks {len(missing)} wells, such as {missing[0]}.",
        )


def check_cold_cache(directory: Path, *, required: bool) -> None:
    """Reject preprocessing inherited from an earlier timed candidate.

    Args:
      directory: Candidate's family-isolated particle cache root.
      required: Whether this run must measure cold preprocessing.

    """
    if required and directory.exists() and next(directory.iterdir(), None) is not None:
        raise FileExistsError("Cold preprocessing requires an empty PF cache.")


def particle_well(well: RawWell) -> ParticleWell:
    """Expose every original float64 observation to native particle tracking.

    Args:
      well: Complete original numeric CSV observations.

    Returns:
      path: Every tracking observation in its original precision and row order.

    """
    values = well.horizontal
    return ParticleWell(
        md=values["MD"],
        z=values["Z"],
        gamma=values["GR"],
        known_depth=values["TVT_input"],
        depth=values.get("TVT"),
        reference_depth=well.reference_depth,
        reference_gamma=well.reference_gamma,
    )


def write_particle_cache(path: Path, values: Mapping[str, NDArray[np.generic]]) -> None:
    """Apply the released cache's posterior quantization at the IO boundary.

    Args:
      path: External runtime cache filename.
      values: Complete native producer record before any IO quantization.

    """
    payload: dict[str, NDArray[np.generic]] = {
        name: values[name].astype(np.float16) for name in PROBABILITIES
    }
    payload.update(
        {name: values[name].astype(np.float32) for name in (*PATHS, *WINDOWS, "tvt0")},
    )
    payload.update({name: values[name].astype(np.bool_) for name in FLAGS})
    payload.update({name: values[name].astype(np.int64) for name in COUNTERS})
    np.savez_compressed(path, allow_pickle=False, **payload)


def read_particle_cache(path: Path) -> dict[str, NDArray[np.generic]]:
    """Restore complete cache fields and sanitize every original visible row.

    Args:
      path: Complete external runtime particle cache.

    Returns:
      cache: Every complete numeric loader field with sanitized original prefix.

    """
    with cast("NpzFile[np.generic]", np.load(path)) as archive:
        result: dict[str, NDArray[np.generic]] = {
            name: archive[name].astype(np.float32)
            for name in (*PROBABILITIES, *PATHS, *WINDOWS)
        }
        result.update({name: archive[name].astype(np.bool_) for name in FLAGS})
        result.update({name: archive[name].astype(np.int64) for name in COUNTERS})
        result["tvt0"] = np.asarray(archive["tvt0"], dtype=np.float64)
    indices = cast("NDArray[np.float32]", result["window_orig_index"])
    visible = (
        np.isfinite(indices)
        & (indices >= 0)
        & (indices <= result["last_seen_idx"].astype(np.float32))
    )
    for name in PROBABILITIES:
        result[name][visible] = 0
    for name in PATHS:
        result[name] = np.where(visible, np.nan, result[name])
    return result


def _manifest_text(config: ParticlePrior.Config) -> str:
    return json.dumps(to_plain(config), sort_keys=True)
