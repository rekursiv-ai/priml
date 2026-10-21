"""Check geographic folds and relative-surface priors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.folds import (
    AnisotropicMetric,
    Clustering,
    GeographicCV,
    GeographicPrior,
    RobustPlane,
    WellTrajectory,
    kmeans,
    lloyd,
    stratified_folds,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


def _kmeans(values: NDArray[np.float64], *, clusters: int) -> Clustering:
    config = GeographicCV.Config()
    return kmeans(
        values,
        clusters=clusters,
        seed=config.cluster_seed,
        extra_trials=config.extra_trials,
        tolerance=config.tolerance,
        max_iterations=config.max_iterations,
    )


@pytest.mark.parametrize(("clusters", "folds"), [(3, 2), (20, 5)])
def test_complete_cluster_and_repeat_records(clusters: int, folds: int) -> None:
    index = np.arange(clusters * (folds + 3), dtype=np.float64)
    group = index % clusters
    values = np.column_stack(
        (group * 100 + (index % 7) / 8, group**2 * 80 + (index % 11) / 16),
    )
    model = _kmeans(values, clusters=clusters)
    prefix = f"{clusters}/{folds}"
    record = {
        f"{prefix}/labels": torch.from_numpy(model.labels),
        f"{prefix}/centers": torch.from_numpy(model.centers),
        f"{prefix}/iterations": torch.tensor(model.iterations),
    }
    complete = GeographicCV.Config(clusters=clusters, folds=folds).make().split(values)
    for repeat in range(3):
        for fold, (training, validation) in enumerate(
            stratified_folds(model.labels, folds=folds, seed=33 + repeat),
        ):
            record[f"{prefix}/{repeat}/{fold}/train"] = torch.from_numpy(training)
            record[f"{prefix}/{repeat}/{fold}/valid"] = torch.from_numpy(validation)
            np.testing.assert_array_equal(
                complete[repeat * folds + fold][0],
                training,
            )
            np.testing.assert_array_equal(
                complete[repeat * folds + fold][1],
                validation,
            )
    expected = read_tensors(_CWD / "testdata" / "folds.pt")
    assert (
        mismatches(
            {
                key: value
                for key, value in expected.items()
                if key.startswith(f"{prefix}/")
            },
            record,
        )
        == []
    )


def test_geographic_population_requirements_and_single_well() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    with pytest.raises(ValueError, match="at least two clusters"):
        GeographicCV.Config(clusters=1).make().split(values)
    with pytest.raises(ValueError, match="one member per fold"):
        GeographicCV.Config(clusters=2, folds=2).make().split(values)
    singleton = GeographicCV.Config().make().split(values[:1])
    assert len(singleton) == 1
    np.testing.assert_array_equal(singleton[0][0], [0])
    assert singleton[0][1].size == 0


def test_lloyd_rescues_empty_centers_and_preserves_input() -> None:
    points = np.asarray([[0, 0], [2, 0], [10, 0], [12, 0]], dtype=np.float64)
    centers = points[[0, 0, 2]].copy()
    result = lloyd(points, centers=centers, tolerance=0, max_iterations=300)
    np.testing.assert_array_equal(centers, points[[0, 0, 2]])
    assert np.bincount(result.labels, minlength=3).min() > 0
    assert np.square(points - result.centers[result.labels]).sum() == 2
    collapsed = lloyd(
        points * 0 + 7,
        centers=centers * 0 + 7,
        tolerance=0,
        max_iterations=300,
    )
    np.testing.assert_array_equal(collapsed.centers, centers * 0 + 7)
    np.testing.assert_array_equal(
        collapsed.labels,
        np.zeros(len(points), dtype=np.int64),
    )


def test_lloyd_termination_and_equal_distance_choose_first_center() -> None:
    points = np.asarray([[0, 0], [1, 0], [3, 0], [7, 0]], dtype=np.float64)
    centers = np.asarray([[0, 0], [2, 0]], dtype=np.float64)
    short = lloyd(points, centers=centers, tolerance=0, max_iterations=1)
    assert short.iterations == 1
    np.testing.assert_array_equal(short.centers, [[0.5, 0], [5, 0]])
    np.testing.assert_array_equal(short.labels, [0, 0, 1, 1])
    complete = lloyd(points, centers=centers, tolerance=0, max_iterations=300)
    assert complete.iterations == 2
    np.testing.assert_array_equal(complete.centers, short.centers)


def test_kmeans_tolerance_uses_feature_variance_and_needs_multiple_updates() -> None:
    points = np.asarray([[0, 0], [2, 0], [10, 0], [12, 0]], dtype=np.float64)
    result = _kmeans(points, clusters=2)
    translated = _kmeans(points + np.asarray([100_000_000, 0]), clusters=2)
    assert result.iterations == translated.iterations == 2
    np.testing.assert_array_equal(result.labels, translated.labels)
    np.testing.assert_array_equal(result.centers, [[11, 0], [1, 0]])


@pytest.mark.parametrize("geometry", ["full", "tiny"])
@pytest.mark.parametrize("leave_one_out", [False, True])
def test_complete_original_geographic_prior(geometry: str, leave_one_out: bool) -> None:
    config = GeographicPrior.Config()
    if geometry == "tiny":
        config.neighbor_wells = 2
        config.point_neighbors = 4
        config.train_bin_size = config.query_bin_size = config.prefix_bin_size = 2
        config.prefix_support_rows = 4
        config.max_support_points = 6
    wells = geographical_wells()
    result = config.make().build(wells, queries=wells, leave_one_out=leave_one_out)
    record = {
        f"{geometry}/{leave_one_out}/{well}/{name}": torch.from_numpy(values)
        for well, prior in result.items()
        for name, values in {"s_rel": prior.values, **prior.diagnostics}.items()
    }
    expected = read_tensors(_CWD / "testdata" / "geography.pt")
    prefix = f"{geometry}/{leave_one_out}/"
    assert (
        mismatches(
            {key: value for key, value in expected.items() if key.startswith(prefix)},
            record,
        )
        == []
    )


def test_metric_promotes_before_rotation_without_mutating_coordinates() -> None:
    points = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = AnisotropicMetric.Config(angle_degrees=0, ratio=2).make()(points)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[0, 1], [1, 3], [2, 5]])
    np.testing.assert_array_equal(points, np.arange(6, dtype=np.float32).reshape(3, 2))


def test_plane_acceptance_bound_does_not_clip_tangent_fallback() -> None:
    # One directional equation is the degenerate plane-fit fallback boundary.
    equations = np.asarray([[[1, 0, 1, 1, 0]]], dtype=np.float64)
    result = RobustPlane.Config().make()(np.ones((1, 1)), equations)
    np.testing.assert_array_equal(result, [[1, 0]])


def test_complete_visible_well_needs_no_support() -> None:
    well = geographical_wells()[0]
    assert well.depth is not None
    complete = replace(well, known_depth=well.depth, suffix_start=len(well.md))
    result = (
        GeographicPrior.Config()
        .make()
        .build(
            [],
            queries=[complete],
            leave_one_out=False,
        )[well.well_id]
    )
    surface = complete.known_depth + complete.z
    np.testing.assert_array_equal(result.values, surface - surface[-1:])
    assert all(np.count_nonzero(values) == 0 for values in result.diagnostics.values())


def test_missing_target_and_missing_support_fail_before_prediction() -> None:
    well = geographical_wells()[0]
    prior = GeographicPrior.Config().make()
    with pytest.raises(ValueError, match=r"well\.depth"):
        prior.build([replace(well, depth=None)], queries=[well], leave_one_out=False)
    with pytest.raises(ValueError, match="requires a support well"):
        prior.build([well], queries=[well], leave_one_out=True)
    stationary = replace(well, x=np.zeros_like(well.x), y=np.zeros_like(well.y))
    with pytest.raises(ValueError, match="no valid surface steps"):
        prior.build([stationary], queries=[stationary], leave_one_out=False)


def test_single_neighbor_is_complete_and_query_targets_are_unused() -> None:
    wells = geographical_wells()
    prior = GeographicPrior.Config(point_neighbors=1).make()
    result = prior.build(wells[1:], queries=[wells[0]], leave_one_out=False)["well0"]
    hidden = replace(wells[0], depth=None)
    replay = prior.build(wells[1:], queries=[hidden], leave_one_out=False)["well0"]
    for expected, actual in zip(
        [result.values, *result.diagnostics.values()],
        [replay.values, *replay.diagnostics.values()],
        strict=True,
    ):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(expected, actual)


def test_inverse_distance_mutation_changes_whole_geographic_record() -> None:
    wells = geographical_wells()
    altered = (
        GeographicPrior.Config(distance_power=1.5)
        .make()
        .build(
            wells,
            queries=wells,
            leave_one_out=False,
        )
    )
    record = {
        f"full/False/{well}/{name}": torch.from_numpy(values)
        for well, prior in altered.items()
        for name, values in {"s_rel": prior.values, **prior.diagnostics}.items()
    }
    expected = read_tensors(_CWD / "testdata" / "geography.pt")
    assert mismatches(
        {
            key: value
            for key, value in expected.items()
            if key.startswith("full/False/")
        },
        record,
    )


def geographical_wells() -> list[WellTrajectory]:
    """Rebuild complete deterministic inputs used by the external numeric minter.

    Returns:
      wells: Four full paths with one abrupt surface change and an anchored suffix.

    """
    wells: list[WellTrajectory] = []
    for index in range(4):
        row = np.arange(15, dtype=np.float32)
        target = (20 + row / 128 + index / 16).astype(np.float32)
        target[8] += np.float32(0.5)
        visible = target.copy()
        visible[4:] = np.nan
        wells.append(
            WellTrajectory(
                well_id=f"well{index}",
                md=row,
                x=(row / 2 + index * 5 + (row % 3) / 16).astype(np.float32),
                y=(row / 4 + index * 7 + (row % 4) ** 2 / 8).astype(np.float32),
                z=(-10 + row / 16).astype(np.float32),
                known_depth=visible,
                depth=target,
                suffix_start=4,
            ),
        )
    return wells


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
