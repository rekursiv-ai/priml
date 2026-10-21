"""Check particle filtering, tuning and caches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.data import load_split
from priml.baselines.wellbore_geology.particles import (
    ParticleCache,
    ParticleDynamics,
    ParticlePrior,
    ParticleWell,
    _normal_numba,
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
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


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


def _streams(
    seed: int,
) -> tuple[np.random.RandomState, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror a legacy generator's full state into the kernel's packed arrays."""
    legacy = np.random.RandomState(seed)
    _, key, position, has_gauss, cached = legacy.get_state(legacy=True)
    return (
        legacy,
        key.astype(np.int64),
        np.asarray([position, has_gauss], dtype=np.int64),
        np.asarray([cached], dtype=np.float64),
    )


@pytest.mark.parametrize("count", [3, 700])
def test_uniform_stream_matches_numpy_across_state_regeneration(count: int) -> None:
    legacy, state, cursor, _ = _streams(5)
    actual = [_uniform_numba(state, cursor) for _ in range(count)]
    assert actual == legacy.random_sample(count).tolist()
    _, key, position, _, _ = legacy.get_state(legacy=True)
    np.testing.assert_array_equal(state, key)
    assert cursor[0] == position


@pytest.mark.parametrize("count", [1, 6, 7])
def test_normal_stream_matches_numpy_with_its_cached_second_draw(count: int) -> None:
    legacy, state, cursor, spare = _streams(11)
    actual = [_normal_numba(state, cursor, spare) for _ in range(count)]
    assert actual == legacy.standard_normal(count).tolist()
    _, _, position, has_gauss, cached = legacy.get_state(legacy=True)
    assert (cursor[0], cursor[1]) == (position, has_gauss)
    if has_gauss:
        assert spare[0] == cached


def test_engine_compiles_the_tracker_once() -> None:
    assert engine() is engine()


_ERROR_SUMMARIES: Final = frozenset(
    f"{name}{suffix}"
    for name in (
        "target_count",
        "tvt_sse",
        "tvt_rmse",
        "prob_row_sum_min",
        "prob_row_sum_max",
    )
    for suffix in ("", "_ffbsi")
)
"""Diagnostics the source minter emitted that no consumer reads."""


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


@pytest.mark.parametrize("suffix_rows", [0, 64, 192])
@pytest.mark.parametrize("has_target", [False, True])
def test_complete_original_mixture_and_backward_records(
    suffix_rows: int,
    has_target: bool,
) -> None:
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
    actual = config.make()(suffix_particle_well(suffix_rows), has_target=has_target)
    prefix = f"{suffix_rows}/{has_target}/"
    record = {
        f"{prefix}{name}": torch.from_numpy(values) for name, values in actual.items()
    }
    expected = read_tensors(_CWD / "testdata" / "particle.pt")
    assert (
        mismatches(
            {
                key: value
                for key, value in expected.items()
                if key.startswith(prefix) and key[len(prefix) :] not in _ERROR_SUMMARIES
            },
            record,
        )
        == []
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
    """Rebuild every deterministic raw observation used by the external minter."""
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


@pytest.mark.parametrize("mutation", ["none", "rough", "jump"])
def test_whole_resampling_and_rare_joint_jump_record(mutation: str) -> None:
    config = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=4,
        seeds=5,
        paths=2,
    )
    for profile in config.profiles:
        if mutation == "rough":
            profile.rough_position = profile.rough_rate = 0
        elif mutation == "jump":
            profile.jump_position = profile.jump_rate = 0
    well = suffix_particle_well(192)
    assert well.depth is not None
    gamma = 70 + 2 * well.depth + (np.arange(224) % 3) / 64
    gamma[2::19] = np.nan
    well = replace(
        well,
        md=64 * well.md,
        gamma=gamma,
        reference_gamma=70 + 2 * well.reference_depth,
    )
    actual = {
        name: torch.from_numpy(values)
        for name, values in config.make()(well, has_target=True).items()
    }
    expected = read_tensors(_CWD / "testdata" / "particle_dynamics.pt")
    differences = mismatches(
        {key: value for key, value in expected.items() if key not in _ERROR_SUMMARIES},
        actual,
    )
    assert bool(differences) == (mutation != "none"), differences


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


@pytest.mark.parametrize("name", ["merge_rows", "blend", "sigma_bounds"])
def test_each_tracking_constant_reaches_the_record(name: str) -> None:
    base = ParticlePrior.Config(
        prefix_rows=32,
        target_rows=192,
        reference_columns=7,
        downsample=32,
        reference_window=100.0,
        particles=3,
        seeds=5,
        paths=2,
    )
    changed = base.copy_tree()
    if name == "merge_rows":
        changed.tracking.merge_rows = 32
    elif name == "blend":
        changed.tracking.calibration.blend = 0.5
    else:
        changed.tracking.sigma_bounds = (1.0, 2.0)
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
