"""Check particle filtering, tuning and caches."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import math

import numpy as np
import pytest

from priml.baselines.wellbore_geology.data import load_split
from priml.baselines.wellbore_geology.particles import (
    COUNTERS,
    PROBABILITIES,
    ParticleCache,
    ParticleDynamics,
    ParticleEmission,
    ParticlePrior,
    ParticleTracking,
    ParticleWell,
    _normal_numba,
    _track_numba,
    _uniform_numba,
    check_cold_cache,
    engine,
    particle_well,
    prepare_particles,
    read_particle_cache,
    validate_particles,
    write_particle_cache,
)
from priml.baselines.wellbore_geology.testing import (
    numerical_tensors,
    write_training_wells,
)
from priml.testing.golden import mismatches
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_complete_native_particle_cache_matches_full_producer_and_io(
    tmp_path: Path,
) -> None:
    write_training_wells(tmp_path / "raw", count=2)
    wells = load_split(tmp_path / "raw")
    config = ParticlePrior.Config()
    config.prefix_rows = 4
    config.target_rows = 6
    config.downsample = 2
    config.reference_columns = 7
    config.reference_window = 100.0
    config.particles = 3
    config.seeds = 5
    config.paths = 2
    actual = prepare_particles(
        wells,
        config=config,
        directory=tmp_path / "cache",
        has_target=True,
    )
    expected: dict[str, dict[str, NDArray[np.generic]]] = {}
    for name, well in wells.items():
        filename = tmp_path / f"expected-{name}.npz"
        write_particle_cache(
            filename,
            config.make()(particle_well(well), has_target=True),
        )
        expected[name] = read_particle_cache(filename)
    assert (
        mismatches(numerical_tensors(expected), numerical_tensors(actual), ulps=True)
        == []
    )
    reused = prepare_particles(
        wells,
        config=config,
        directory=tmp_path / "cache",
        has_target=True,
    )
    assert (
        mismatches(numerical_tensors(actual), numerical_tensors(reused), ulps=True)
        == []
    )
    config.seeds = 6
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_particles(
            wells,
            config=config,
            directory=tmp_path / "cache",
            has_target=True,
        )


def test_nonempty_unidentified_cache_fails_before_generating(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "unidentified.npz").write_bytes(b"unknown")
    with pytest.raises(FileExistsError, match="configuration manifest"):
        prepare_particles(
            {},
            config=ParticlePrior.Config(),
            directory=directory,
            has_target=False,
        )


def test_validation_requires_a_complete_cache_matching_the_recipe(
    tmp_path: Path,
) -> None:
    write_training_wells(tmp_path / "raw", count=2)
    wells = load_split(tmp_path / "raw")
    config = ParticlePrior.Config()
    config.prefix_rows = 4
    config.target_rows = 6
    config.downsample = 2
    config.reference_columns = 7
    config.reference_window = 100.0
    config.particles = 3
    config.seeds = 5
    config.paths = 2
    directory = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="not prepared"):
        validate_particles(wells, config=config, directory=directory)
    prepare_particles(wells, config=config, directory=directory, has_target=True)
    validate_particles(wells, config=config, directory=directory)
    other = config.copy_tree()
    other.particles = 4
    with pytest.raises(ValueError, match="differs from this recipe"):
        validate_particles(wells, config=other, directory=directory)
    (directory / "well1.npz").unlink()
    with pytest.raises(FileNotFoundError, match="lacks 1 wells, such as well1"):
        validate_particles(wells, config=config, directory=directory)


def test_generation_stops_before_a_well_once_the_allowance_is_spent(
    tmp_path: Path,
) -> None:
    write_training_wells(tmp_path / "raw", count=2)
    config = ParticlePrior.Config(
        prefix_rows=4,
        target_rows=6,
        downsample=2,
        reference_columns=7,
        reference_window=100.0,
    )
    # The clock reads 0 at construction and 11 at the first well's charge.
    budget = WallBudget.Config(seconds=10, clock=iter([0.0, 11.0]).__next__).make()
    with pytest.raises(TimeoutError, match="No training time remains"):
        prepare_particles(
            load_split(tmp_path / "raw"),
            config=config,
            directory=tmp_path / "cache",
            has_target=True,
            budget=budget,
        )
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == [
        "config.json",
    ]


def test_cold_cache_is_part_of_candidate_preprocessing(tmp_path: Path) -> None:
    check_cold_cache(tmp_path / "absent", required=True)
    check_cold_cache(tmp_path, required=True)
    (tmp_path / "cache_entry").write_text("previous candidate")
    with pytest.raises(
        FileExistsError,
        match="Cold preprocessing requires an empty PF cache",
    ):
        check_cold_cache(tmp_path, required=True)
    check_cold_cache(tmp_path, required=False)


def test_particle_cache_requires_its_directory() -> None:
    with pytest.raises(ValueError, match="requires its directory"):
        ParticleCache.Config().make()


def _seeded_stream(seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Seed the kernel's packed MT19937 words exactly as its tracker does."""
    state = np.empty(624, dtype=np.int64)
    state[0] = seed
    for word in range(1, 624):
        previous = int(state[word - 1])
        state[word] = (1_812_433_253 * (previous ^ (previous >> 30)) + word) & (
            2**32 - 1
        )
    return state, np.asarray([624, 0], dtype=np.int64)


def test_uniform_stream_takes_two_words_per_draw_across_state_regeneration() -> None:
    state, cursor = _seeded_stream(5)
    draws = [_uniform_numba(state, cursor) for _ in range(700)]
    # Draws 312 and 624 are the first after each regeneration of the 624 words.
    assert [draws[index] for index in (0, 1, 2, 311, 312, 623, 624, 699)] == [
        0.22199317108973948,
        0.8707323061773764,
        0.20671915533942642,
        0.9985157672628341,
        0.03834193016002463,
        0.6146949375583336,
        0.21281212452735265,
        0.7464423502866897,
    ]
    assert cursor.tolist() == [2 * 700 - 2 * 624, 0]
    assert all(draw * 2**53 == int(draw * 2**53) for draw in draws)


@pytest.mark.parametrize(
    ("count", "words", "cached"),
    [(1, 8, -0.28607299681629417), (6, 16, None), (7, 24, 0.3154026684256969)],
)
def test_normal_stream_caches_the_second_polar_draw(
    count: int,
    words: int,
    cached: float | None,
) -> None:
    state, cursor = _seeded_stream(11)
    spare = np.zeros(1)
    actual = [_normal_numba(state, cursor, spare) for _ in range(count)]
    assert (
        actual
        == [
            1.7494547413051793,
            -0.28607299681629417,
            -0.4845651322211443,
            -2.653318559261479,
            -0.00828462937293584,
            -0.31963136376429824,
            -0.53662936223473,
        ][:count]
    )
    assert cursor.tolist() == [words, int(cached is not None)]
    if cached is not None:
        assert spare[0] == cached


def test_engine_compiles_the_tracker_once() -> None:
    assert engine() is engine()


def test_one_merged_step_scores_both_curves_and_the_anchor_prior() -> None:
    still = ParticleDynamics.Config(
        momentum=1,
        rate_noise=0,
        position_noise=0,
        resample_threshold=0,
        rough_position=0,
        rough_rate=0,
        initial_position=0,
        initial_rate=0,
        mean_reversion=0,
        jump_probability=0,
        jump_position=0,
        jump_rate=0,
    ).make()
    emission = ParticleEmission.Config(
        power_exponent=0,
        run_gain=0.06,
        run_floor=0.75,
        primary_weight=0.75,
        raw_weight=0.25,
        anchor_scale=4,
        anchor_strength=8,
        grid_step=8,
    ).make()
    tracking = ParticleTracking.Config(noise_correlation=1, resample_floor=0).make()
    # Four motionless particles sit at depth 12, 2 deeper than the anchor, halfway
    # between grid depths 8 and 16: the calibrated curve reads 85 and the raw 80.
    # Two rows observe gamma 85 and 75 with sigma 5.
    histogram, density, smoothed, path, evidence = _track_numba(
        np.asarray([[85, 1, 1, 0], [75, 1, 2, 0]], dtype=np.float64),
        np.asarray([[70, 80, 90], [60, 60, 100]], dtype=np.float64),
        np.asarray([10, 12, 0, 0, 5, 0, -100, 100, 3, 2.5], dtype=np.float64),
        np.concatenate((still.values, emission.values, tracking.values)),
        np.asarray([2, 1, 5, 4, 6, 7, 2, 512], dtype=np.int64),
    )
    # A zero exponent gives the block power 1, half per row; the second finite row
    # gains its run bonus and takes the 0.75 floor. The anchor prior is e^-1 per row.
    power = 0.5 * (1 + 0.06 * math.log(3)) * 0.75
    likelihood = (
        (0.75 + 0.25 * math.exp(-0.25))
        * (0.75 * math.exp(-2 * power) + 0.25 * math.exp(-0.5 * power))
        * math.exp(-1)
        * math.exp(-1)
    )
    assert evidence == math.log(sum([0.25 * likelihood] * 4))
    # The relative depth 2 is the last of the candidates -2, -1, 0, 1 and 2.
    np.testing.assert_array_equal(histogram, [[0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(density, [[0, 0, 0, 0, 4]])
    np.testing.assert_array_equal(smoothed, histogram)
    np.testing.assert_array_equal(path, [[12, 12]])


def test_integer_transition_parameters_retain_binary64_kernel_operands() -> None:
    dynamics = ParticleDynamics.Config(
        momentum=1,
        rate_noise=0,
        position_noise=0,
        resample_threshold=1,
        rough_position=0,
        rough_rate=0,
        initial_position=2,
        initial_rate=0,
        mean_reversion=0,
        jump_probability=0,
        jump_position=7,
        jump_rate=0,
    ).make()
    np.testing.assert_array_equal(
        dynamics.values,
        np.asarray([1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 7, 0], dtype=np.float64),
        strict=True,
    )


def test_a_well_without_hidden_rows_places_each_visible_bin_on_the_axis() -> None:
    # Seven candidates one unit apart, centered on the last visible depth.
    config = ParticlePrior.Config(
        prefix_rows=4,
        target_rows=4,
        downsample=2,
        reference_columns=7,
        reference_window=3.5,
        particles=3,
        seeds=5,
        paths=2,
    )
    known = np.asarray([10, 10.25, 10.5, 10.5])
    well = ParticleWell(
        md=np.arange(4.0),
        z=np.zeros(4),
        gamma=np.full(4, 80.0),
        known_depth=known,
        depth=known.copy(),
        reference_depth=np.arange(5.0),
        reference_gamma=np.full(5, 80.0),
    )
    record = config.make()(well, has_target=True)
    # Bin 0 averages 10 and 10.25, 0.375 below the anchor: 5/8 of its mass on the
    # anchor's candidate and 3/8 on the next one down.
    mass = np.zeros((5, 7), dtype=np.float32)
    mass[0, 2:4] = [0.375, 0.625]
    mass[1, 3] = 1
    for name in ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi"):
        np.testing.assert_array_equal(record[name], mass, strict=True)
    for name in ("pf_tvt_pred", "pf_tvt_pred_ffbsi"):
        np.testing.assert_array_equal(
            record[name],
            np.asarray([10.125, 10.5, 0, 0, 0], dtype=np.float32),
            strict=True,
        )
    for name in ("pf_tvt_rel_pred", "pf_tvt_rel_pred_ffbsi", "window_tvt_rel"):
        np.testing.assert_array_equal(
            record[name],
            np.asarray([-0.375, 0, 0, 0, 0], dtype=np.float32),
            strict=True,
        )
    np.testing.assert_array_equal(
        record["window_orig_index"],
        np.asarray([0.5, 2.5, -1, -1, -1], dtype=np.float32),
        strict=True,
    )
    np.testing.assert_array_equal(record["window_has_tvt"], [1, 1, 0, 0, 0])
    assert not record["target_mask"].any()
    assert [int(record[name]) for name in COUNTERS] == [3, 0, 0]
    assert record["tvt0"] == 10.5


@pytest.mark.parametrize("has_target", [False, True])
def test_window_bins_average_visible_and_retained_target_depths(
    has_target: bool,
) -> None:
    well = suffix_particle_well(64)
    assert well.depth is not None
    record = _small_prior().make()(well, has_target=has_target)
    depth = well.depth.astype(np.float32).astype(np.float64)
    # (32 + 192) // 32 + 1 bins: the prefix, two of hidden rows and five empty.
    target = [has_target] * 2
    means = [depth[start : start + 32].mean() for start in (0, 32, 64)]
    expected = np.asarray(
        [means[0], *(means[1:] if has_target else [np.nan] * 2), *[np.nan] * 5],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        record["window_tvt"],
        np.nan_to_num(expected),
        strict=True,
    )
    np.testing.assert_array_equal(
        record["window_tvt_rel"],
        np.nan_to_num(expected - np.float32(depth[31])),
        strict=True,
    )
    np.testing.assert_array_equal(
        record["window_has_tvt"],
        [True, *target, *[False] * 5],
    )
    np.testing.assert_array_equal(record["target_mask"], [False, *target, *[False] * 5])
    np.testing.assert_array_equal(
        record["window_orig_index"],
        np.asarray([15.5, 47.5, 79.5, *[-1] * 5], dtype=np.float32),
        strict=True,
    )
    assert [int(record[name]) for name in COUNTERS] == [31, 0, 64]


@pytest.mark.parametrize(("suffix_rows", "steps"), [(64, 1), (192, 3)])
def test_tracked_posteriors_are_normalized_and_paths_are_anchor_relative(
    suffix_rows: int,
    steps: int,
) -> None:
    record = _small_prior().make()(
        suffix_particle_well(suffix_rows),
        has_target=False,
    )
    # Thirty-two rows per bin after the 32-row prefix bin, eight bins in all.
    tracked = np.arange(1, 1 + suffix_rows // 32)
    untracked = np.arange(1 + suffix_rows // 32, 8)
    for name in PROBABILITIES:
        values = record[name].astype(np.float32)
        assert values.shape == (8, 7)
        assert values.min() >= 0
        np.testing.assert_allclose(values[tracked].sum(axis=1), 1, rtol=0, atol=1e-6)
        assert not values[untracked].any()
    anchor = np.float32(record["tvt0"])
    for path, relative in (
        ("pf_tvt_pred", "pf_tvt_rel_pred"),
        ("pf_tvt_pred_ffbsi", "pf_tvt_rel_pred_ffbsi"),
    ):
        values = record[path].astype(np.float32)
        assert np.isfinite(values[tracked]).all()
        np.testing.assert_array_equal(
            record[relative][tracked],
            values[tracked] - anchor,
            strict=True,
        )
        assert not values[untracked].any()
    # A single merged step has no backward pass, so FFBSi repeats the filter.
    assert np.array_equal(record["pf_prob_ffbsi"], record["pf_prob"]) == (steps == 1)
    assert np.array_equal(record["pf_tvt_pred_ffbsi"], record["pf_tvt_pred"]) == (
        steps == 1
    )


def test_merged_steps_expand_row_by_row_into_their_bins() -> None:
    config = ParticlePrior.Config(
        prefix_rows=1,
        target_rows=8,
        downsample=4,
        reference_columns=2,
        reference_window=1.0,
    )
    config.tracking.merge_rows = 3
    prior = config.make()
    # One prefix row puts rows 0-2 in bin 0, rows 3-6 in bin 1 and row 7 in bin 2.
    path = prior._expand_path(
        np.asarray([13, np.nan, 19], dtype=np.float32),
        anchor=10.0,
        rows=8,
        bins=4,
    )
    # Rows climb 11, 12, 13 from the anchor, hold 13 over the gap, then reach 16, 19.
    np.testing.assert_array_equal(
        path,
        np.asarray([12, 13.75, 19, np.nan], dtype=np.float32),
        strict=True,
    )
    probability = prior._expand_probability(
        np.asarray([[0.5, 0.5], [0, 1], [1, 0]], dtype=np.float32),
        rows=8,
        bins=4,
    )
    # Bin 1 weighs three rows of the second step against one of the third.
    np.testing.assert_array_equal(
        probability,
        np.asarray([[0.5, 0.5], [0.25, 0.75], [1, 0], [0, 0]], dtype=np.float32),
        strict=True,
    )


def test_seed_runs_mix_by_profile_prior_and_skip_missing_path_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_prior()
    config.reference_columns = 3
    config.seeds = 3
    config.tracking.merge_rows = 32
    config.profiles = [
        ParticleDynamics.Config(weight=0.75, initial_position=1),
        ParticleDynamics.Config(weight=0.25, initial_position=2),
    ]
    # The first profile runs two of the three seeds; equal evidence splits its 3/4.
    kernel = _ScriptedKernel(
        {
            (1.0, 0): (0, [1, math.nan]),
            (1.0, 1): (2, [3, 2]),
            (2.0, 0): (1, [5, 7]),
        },
    )
    monkeypatch.setattr(
        "priml.baselines.wellbore_geology.particles.engine",
        lambda: kernel,
    )
    well = suffix_particle_well(64)
    probabilities, paths = config.make()._suffix(
        well,
        last=31,
        anchor=float(well.known_depth[31]),
        rows=64,
    )
    mixed = np.asarray([[0.375, 0.25, 0.375]] * 2, dtype=np.float32)
    # The density map weighs every seed run equally.
    uniform = np.full((2, 3), 1 / 3, dtype=np.float32)
    for actual, expected in zip(probabilities, (mixed, uniform, mixed), strict=True):
        np.testing.assert_array_equal(actual, expected, strict=True)
    # Step 0 averages 1, 3 and 5 at 3/8, 3/8 and 1/4; step 1 drops the missing run.
    for path in paths:
        np.testing.assert_array_equal(
            path,
            np.asarray([2.75, 4], dtype=np.float32),
            strict=True,
        )


class _ScriptedKernel:
    """Return one scripted tracker run per call, keyed by profile and call order."""

    def __init__(self, runs: dict[tuple[float, int], tuple[int, list[float]]]) -> None:
        self.runs = runs
        self.calls: dict[float, int] = {}

    def __call__(
        self,
        observations: NDArray[np.float64],
        references: NDArray[np.float64],
        initial: NDArray[np.float64],
        parameters: NDArray[np.float64],
        sizes: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        del observations, references, initial
        # Each profile is told apart by its initial position.
        profile = float(parameters[6])
        call = self.calls.get(profile, 0)
        self.calls[profile] = call + 1
        column, path = self.runs[profile, call]
        mass = np.zeros((int(sizes[1]), int(sizes[2])))
        mass[:, column] = 1
        track = np.repeat(np.asarray(path, dtype=np.float64)[:, None], 2, axis=1)
        return mass, mass.copy(), mass.copy(), track, 0.0


def _small_prior() -> ParticlePrior.Config:
    return ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=3,
        seeds=5,
        paths=2,
    )


def _public_geometry() -> ParticlePrior.Config:
    config = ParticlePrior.Config()
    config.prefix_rows = 1024
    config.target_rows = 10_000
    config.downsample = 32
    config.reference_columns = 400
    config.reference_window = 100.0
    return config


def suffix_particle_well(suffix_rows: int) -> ParticleWell:
    """Build the deterministic raw observations of the particle tests."""
    row = np.arange(32 + suffix_rows, dtype=np.float64)
    target = 20 + row / 128 + row**2 / 65_536
    visible = target.copy()
    visible[32:] = np.nan
    gamma = 75 + (row % 17) / 4 + row / 32
    gamma[2::19] = np.nan
    reference_axis = np.arange(9, dtype=np.float64)
    return ParticleWell(
        md=row,
        z=-10 + row / 64,
        gamma=gamma,
        known_depth=visible,
        depth=target,
        reference_depth=-140 + reference_axis * 40,
        reference_gamma=70 + reference_axis**2 / 2,
    )


@pytest.mark.parametrize("disabled", ["rough", "jump"])
def test_resampling_jitter_and_rare_joint_jumps_reach_the_record(disabled: str) -> None:
    config = _small_prior()
    config.particles = 4
    well = _sharp_particle_well()
    changed = config.copy_tree()
    for profile in changed.profiles:
        if disabled == "rough":
            profile.rough_position = profile.rough_rate = 0
        else:
            profile.jump_position = profile.jump_rate = 0
    wanted = config.make()(well, has_target=True)
    actual = changed.make()(well, has_target=True)
    assert any(not np.array_equal(wanted[key], actual[key]) for key in wanted)


def test_the_backward_transition_width_reaches_only_the_backward_paths() -> None:
    config = _small_prior()
    config.particles = 8
    well = _sharp_particle_well()
    changed = config.copy_tree()
    changed.tracking.backward_scale = 3.0
    wanted = config.make()(well, has_target=True)
    actual = changed.make()(well, has_target=True)
    assert {key for key in wanted if not np.array_equal(wanted[key], actual[key])} == {
        "pf_prob_ffbsi",
        "pf_tvt_pred_ffbsi",
        "pf_tvt_rel_pred_ffbsi",
    }


def _sharp_particle_well() -> ParticleWell:
    """Rows 64 MD apart make jumps likely; the sharp curve forces resampling."""
    well = suffix_particle_well(192)
    assert well.depth is not None
    gamma = 70 + 2 * well.depth + (np.arange(224) % 3) / 64
    gamma[2::19] = np.nan
    return replace(
        well,
        md=64 * well.md,
        gamma=gamma,
        reference_gamma=70 + 2 * well.reference_depth,
    )


def test_complete_public_dynamics_and_small_seed_allocation() -> None:
    prior = _public_geometry().make()
    np.testing.assert_array_equal(
        np.stack([profile.values for profile in prior.profiles]),
        [
            [
                0.9995,
                0.0009,
                0.0025,
                0.3,
                0.08,
                0.006,
                2,
                0.02,
                0.001,
                0.000075,
                7,
                0.0003,
            ],
            [
                0.9995,
                0.0009,
                0.0025,
                0.35,
                0.1,
                0.006,
                2.5,
                0.025,
                0.001,
                0.000075,
                7,
                0.0003,
            ],
            [
                0.9995,
                0.0008,
                0.003,
                0.3,
                0.06,
                0.0045,
                2,
                0.02,
                0.0015,
                0.000075,
                9,
                0.0003,
            ],
            [
                0.9995,
                0.0009,
                0.0025,
                0.35,
                0.08,
                0.006,
                2,
                0.02,
                0.001,
                0.000075,
                10,
                0.0006,
            ],
            [
                0.9995,
                0.0009,
                0.0035,
                0.3,
                0.06,
                0.006,
                1,
                0.015,
                0.001,
                0.000075,
                7,
                0.0003,
            ],
        ],
    )
    assert [profile.weight for profile in prior.profiles] == [
        0.6514002115702939,
        0.19545809861466074,
        0.053120440286232196,
        0.07370294173329474,
        0.02631830779551842,
    ]
    for budget in (1, 3, 5, 128, 256):
        config = _public_geometry()
        config.seeds = budget
        allocation = config.make()._allocation()
        assert len(allocation) == budget
        assert int(allocation.min()) >= 0
        assert int(allocation.max()) < 5
        if budget >= 5:
            assert set(allocation.flat) == set(range(5))


def test_invalid_particle_budgets_and_missing_anchor_fail_before_tracking() -> None:
    config = _public_geometry()
    config.seeds = 0
    with pytest.raises(ValueError, match="budgets must be positive"):
        config.make()
    well = suffix_particle_well(0)
    with pytest.raises(ValueError, match="visible depth anchor"):
        _public_geometry().make()(
            replace(well, known_depth=np.full_like(well.known_depth, np.nan)),
            has_target=False,
        )


@pytest.mark.parametrize("missing", ["prefix", "reference"])
def test_insufficient_tracking_support_retains_the_anchor_path(missing: str) -> None:
    config = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=3,
        seeds=5,
        paths=2,
    )
    well = suffix_particle_well(64)
    if missing == "prefix":
        visible = well.known_depth.copy()
        visible[1:] = np.nan
        well = replace(well, known_depth=visible)
    else:
        well = replace(
            well,
            reference_depth=well.reference_depth[:2],
            reference_gamma=well.reference_gamma[:2],
        )
    record = config.make()(well, has_target=False)
    for name in ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi"):
        assert np.count_nonzero(record[name][1:]) == 0
    anchor = np.float32(
        well.known_depth.flat[np.flatnonzero(np.isfinite(well.known_depth)).flat[-1]],
    )
    for name in ("pf_tvt_pred", "pf_tvt_pred_ffbsi"):
        active = record["window_orig_index"][1:].astype(np.float32) >= 0
        np.testing.assert_array_equal(
            record[name][1:][active],
            np.full(np.count_nonzero(active), anchor),
        )


def test_tracking_constants_are_printed_tunables() -> None:
    config = _public_geometry()
    plain = {
        "emission": config.emission.make().values.tolist(),
        "tracking": config.tracking.make().values.tolist(),
    }
    assert plain["emission"] == [1.05, 0.06, 0.6, 0.925, 0.075, 80.0, 0.002, 0.05]
    assert plain["tracking"][2:] == [0.45, 0.3, 1.35, 0.15, 0.0025]
    assert plain["tracking"][:2] == [0.2, (1 - 0.2 * 0.2) ** 0.5]
    config.tracking.merge_rows = 0
    with pytest.raises(ValueError, match="positive row merge"):
        config.make()


@pytest.mark.parametrize(
    "name",
    [
        "merge_rows",
        "blend",
        "sigma_bounds",
        "evidence_temperature",
        "power_exponent",
        "primary_weight",
    ],
)
def test_each_tracking_and_emission_constant_reaches_the_record(name: str) -> None:
    base = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=3,
        # Seven seeds give two profiles a second run for the evidence to weigh.
        seeds=7,
        paths=2,
    )
    changed = base.copy_tree()
    if name == "merge_rows":
        changed.tracking.merge_rows = 32
    elif name == "blend":
        changed.tracking.calibration.blend = 0.5
    elif name == "sigma_bounds":
        changed.tracking.sigma_bounds = (1.0, 2.0)
    elif name == "evidence_temperature":
        changed.tracking.evidence_temperature = 5.0
    elif name == "power_exponent":
        changed.emission.power_exponent = 1.0
    else:
        changed.emission.primary_weight = 0.9
    well = suffix_particle_well(192)
    wanted = base.make()(well, has_target=True)
    actual = changed.make()(well, has_target=True)
    assert any(not np.array_equal(wanted[key], actual[key]) for key in wanted)


def test_particle_filtering_requires_a_positive_row_merge() -> None:
    config = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
    )
    config.tracking.merge_rows = 0
    with pytest.raises(ValueError, match="positive row merge"):
        config.make()


def test_fewer_seed_runs_than_profiles_skip_the_unallocated_profiles() -> None:
    config = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=3,
        seeds=2,
        paths=2,
    )
    record = config.make()(suffix_particle_well(64), has_target=False)
    assert record["pf_prob"].shape[1] == 7


def test_particle_tracking_requires_its_window_geometry() -> None:
    with pytest.raises(ValueError, match="window geometry"):
        ParticlePrior.Config().make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
