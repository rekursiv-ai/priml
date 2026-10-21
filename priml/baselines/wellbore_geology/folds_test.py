"""Check geographic folds and relative-surface priors."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.wellbore_geology.folds import (
    AnisotropicMetric,
    Clustering,
    GeographicCV,
    GeographicPrior,
    RobustPlane,
    SupportPoints,
    WellTrajectory,
    _nearest,
    kmeans,
    lloyd,
    stratified_folds,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


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
def test_kmeans_recovers_separated_groups_and_each_repeat_stratifies_them(
    clusters: int,
    folds: int,
) -> None:
    values, group = _separated_groups(clusters=clusters, folds=folds)
    model = _kmeans(values, clusters=clusters)
    # One label per whole group: the labels repeat with the group period.
    np.testing.assert_array_equal(model.labels, model.labels[group])
    assert sorted(model.labels[:clusters].tolist()) == list(range(clusters))
    means = np.stack(
        [values[group == member].mean(axis=0) for member in range(clusters)],
    )
    # Centering the coordinates costs the centers float64 rounding only.
    np.testing.assert_allclose(
        model.centers[model.labels[:clusters]],
        means,
        rtol=0,
        atol=1e-12,
    )
    assert model.iterations == 1
    rows = np.arange(len(values))
    members = folds + 3
    complete = GeographicCV.Config(clusters=clusters, folds=folds).make().split(values)
    assert len(complete) == 3 * folds
    for repeat in range(3):
        splits = stratified_folds(model.labels, folds=folds, seed=33 + repeat)
        held = [validation for _, validation in splits]
        np.testing.assert_array_equal(np.sort(np.concatenate(held)), rows)
        for fold, (training, validation) in enumerate(splits):
            np.testing.assert_array_equal(training, np.setdiff1d(rows, validation))
            counts = np.bincount(group[validation], minlength=clusters)
            assert counts.min() >= members // folds
            assert counts.max() <= -(-members // folds)
            np.testing.assert_array_equal(complete[repeat * folds + fold][0], training)
            np.testing.assert_array_equal(
                complete[repeat * folds + fold][1],
                validation,
            )


def test_three_groups_take_these_two_fold_validation_rows() -> None:
    values, _ = _separated_groups(clusters=3, folds=2)
    labels = _kmeans(values, clusters=3).labels
    np.testing.assert_array_equal(labels, [2, 1, 0] * 5)
    assert [
        [
            validation.tolist()
            for _, validation in stratified_folds(labels, folds=2, seed=seed)
        ]
        for seed in (33, 34, 35)
    ] == [
        [[0, 1, 2, 3, 6, 8, 10, 11], [4, 5, 7, 9, 12, 13, 14]],
        [[0, 1, 5, 7, 8, 9, 12, 14], [2, 3, 4, 6, 10, 11, 13]],
        [[0, 2, 5, 6, 10, 12, 13, 14], [1, 3, 4, 7, 8, 9, 11]],
    ]


def _separated_groups(
    *,
    clusters: int,
    folds: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    index = np.arange(clusters * (folds + 3), dtype=np.int64)
    group = index % clusters
    values = np.column_stack(
        (group * 100 + (index % 7) / 8, group**2 * 80 + (index % 11) / 16),
    ).astype(np.float64)
    return values, group


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


def test_rescue_gives_tied_farthest_points_to_empty_centers_in_row_order() -> None:
    points = np.asarray([[0, 0], [1, 0], [4, 0], [-4, 0]], dtype=np.float64)
    # Three equal centers leave two empty; rows 2 and 3 tie as the farthest.
    result = lloyd(points, centers=np.zeros((3, 2)), tolerance=0, max_iterations=1)
    np.testing.assert_array_equal(result.centers, [[0.5, 0], [4, 0], [-4, 0]])
    np.testing.assert_array_equal(result.labels, [0, 0, 1, 2])


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


def test_prior_integrates_a_shared_along_track_slope_exactly() -> None:
    config = GeographicPrior.Config(point_neighbors=1)
    config.train_bin_size = config.query_bin_size = config.prefix_bin_size = 2
    result = config.make().build(
        _sloped_wells(),
        queries=_sloped_wells()[:1],
        leave_one_out=False,
    )["well0"]
    # Every support point has the tangent slope 1/8 along x, and a single point
    # cannot fit a plane, so the suffix integrates 1/8 of each half-unit step.
    np.testing.assert_array_equal(
        result.values,
        (np.arange(15, dtype=np.float32) - 3) / 16,
        strict=True,
    )
    assert not result.diagnostics["geo_nbr_std"].any()
    assert not result.diagnostics["geo_nbr_dS_std"].any()
    np.testing.assert_array_equal(
        result.diagnostics["geo_nbr_path_alignment"],
        np.repeat(np.asarray([0, 1], dtype=np.float32), [4, 11]),
        strict=True,
    )


def test_support_cap_keeps_the_points_nearest_any_query_bin() -> None:
    config = GeographicPrior.Config(max_support_points=8)
    config.train_bin_size = config.query_bin_size = config.prefix_bin_size = 2
    prior = config.make()
    query, *support = _sloped_wells()
    selected = prior._neighbors(
        query,
        support=support,
        points={well.well_id: prior._support(well, prefix=False) for well in support},
        xy=np.column_stack((np.arange(2, 8), np.zeros(6))).astype(np.float32),
    )
    # The query's two visible-prefix points and well1's six suffix points, at y 0
    # and 10, are nearer than well2's six at y 20.
    np.testing.assert_array_equal(selected.is_prefix, [True] * 2 + [False] * 6)
    np.testing.assert_array_equal(selected.values[:, 1], [0] * 2 + [10] * 6)


def test_nearest_points_sort_by_distance_and_keep_ties_in_row_order() -> None:
    # Four copies of five points: every distance ties across rows i, i + 5, ...
    # Twenty rows reach NumPy's unstable default sort.
    pattern = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1], [2, 0]], dtype=np.float64)
    distance, indices = _nearest(
        np.tile(pattern, (4, 1)),
        queries=np.asarray([[0, 0], [2, 0], [0, 3]], dtype=np.float64),
        count=6,
    )
    np.testing.assert_array_equal(
        indices,
        [[0, 1, 2, 3, 5, 6], [4, 9, 14, 19, 0, 5], [1, 6, 11, 16, 0, 2]],
    )
    np.testing.assert_array_equal(
        distance,
        [[1] * 6, [0] * 4 + [1] * 2, [2] * 4 + [10**0.5] * 2],
        strict=True,
    )


def test_non_finite_coordinates_fail_the_neighbor_search() -> None:
    points = np.asarray([[0, 0], [np.inf, 1], [2, 2]], dtype=np.float64)
    queries = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="must be finite"):
        _nearest(points, queries=queries, count=2)
    points[1, 0] = 1
    queries[3, 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _nearest(points, queries=queries, count=2)
    wells = geographical_wells()
    x = wells[0].x.copy()
    x[wells[0].suffix_start + 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        GeographicPrior.Config().make().build(
            wells[1:],
            queries=[replace(wells[0], x=x)],
            leave_one_out=False,
        )


def test_inverse_square_weights_set_the_gradient_and_every_diagnostic() -> None:
    config = GeographicPrior.Config(point_neighbors=3)
    config.metric = AnisotropicMetric.Config(angle_degrees=0, ratio=1)
    # Columns: x, y, step direction, normalized step, tangent gradient, path direction.
    support = SupportPoints(
        values=np.asarray(
            [
                [2, 0, 1, 0, 1, 0.125, 0.5, 1, 0],
                [2, 2, 1, 0, 1, 0.25, -0.25, 0, 1],
                [2, -2, 1, 0, 1, -0.25, 0.75, 0, 1],
                [18, 18, 1, 0, 1, 2, 2, 1, 0],
            ],
            dtype=np.float32,
        ),
        is_prefix=np.asarray([True, True, False, False], dtype=np.bool_),
    )
    gradient, diagnostics = config.make()._predict(
        support,
        xy=np.asarray([[0, 0], [4, 0]], dtype=np.float32),
        dxy=np.asarray([[-1, 0], [2, 0]], dtype=np.float32),
    )
    # Both queries have neighbors at squared distances 4, 8 and 8, so the weights
    # are 1/2, 1/4 and 1/4 around the weighted center (2, 0). One step direction
    # leaves no plane, and the fallback averages the tangent gradients.
    np.testing.assert_array_equal(
        gradient,
        np.asarray([[0.0625, 0.375]] * 2, dtype=np.float32),
        strict=True,
    )
    expected = {
        "geo_nbr_std": [2**0.5] * 2,
        "geo_nbr_dS_std": [0.1875, 0.375],
        # Each query lies 2 from the center, over a deviation of sqrt(2).
        "geo_radial_extrap_score": [2 / 2**0.5] * 2,
        "geo_nbr_distance": [6**0.5] * 2,
        # The first query runs against the first neighbor's path and aligns fully.
        "geo_nbr_path_alignment": [0.5] * 2,
        "geo_prefix_weight_ratio": [0.75] * 2,
    }
    assert list(diagnostics) == list(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(
            diagnostics[name],
            np.asarray(values, dtype=np.float32),
            strict=True,
        )


def test_leave_one_out_equals_building_without_the_query_well() -> None:
    config = GeographicPrior.Config(neighbor_wells=2, point_neighbors=4)
    config.train_bin_size = config.query_bin_size = config.prefix_bin_size = 2
    prior = config.make()
    wells = geographical_wells()
    held = prior.build(wells, queries=wells, leave_one_out=True)
    for well in wells:
        others = [other for other in wells if other.well_id != well.well_id]
        alone = prior.build(others, queries=[well], leave_one_out=False)[well.well_id]
        np.testing.assert_array_equal(held[well.well_id].values, alone.values)
        assert list(held[well.well_id].diagnostics) == list(alone.diagnostics)
        for name, values in alone.diagnostics.items():
            np.testing.assert_array_equal(held[well.well_id].diagnostics[name], values)
        suffix = held[well.well_id].diagnostics
        assert np.isfinite(held[well.well_id].values).all()
        for name in ("geo_nbr_path_alignment", "geo_prefix_weight_ratio"):
            assert 0 <= suffix[name].min() <= suffix[name].max() <= 1
        for name in ("geo_nbr_std", "geo_nbr_dS_std", "geo_nbr_distance"):
            assert suffix[name].min() >= 0


def test_metric_promotes_before_rotation_without_mutating_coordinates() -> None:
    points = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = AnisotropicMetric.Config(angle_degrees=0, ratio=2).make()(points)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[0, 1], [1, 3], [2, 5]])
    np.testing.assert_array_equal(points, np.arange(6, dtype=np.float32).reshape(3, 2))


@pytest.mark.parametrize(
    ("rise", "iterations", "gradient"),
    [
        # With ridge 1/4, [[3/4, 1/4], [1/4, 3/4]] maps (5/32, 1/32) to (1/8, 1/16).
        (1, 0, (5 / 32, 1 / 32)),
        # Twice the rise fits 5/16, past the 1/4 bound; the tangent mean replaces it.
        (2, 0, (1 / 4, 1 / 8)),
        # Residuals 3/32, 1/32, 2/32, 0 over 1/16 keep 9/16 of row 1, all of row 3.
        (1, 1, (9 / 25, 9 / 50)),
    ],
)
def test_ridged_plane_solves_bounds_and_reweights_the_normal_equations(
    rise: int,
    iterations: int,
    gradient: tuple[float, float],
) -> None:
    # Unit steps along x, y and the diagonal form the design [[1/2, 1/4], [1/4, 1/2]].
    # Two queries share the design, so each must solve to the same gradient.
    equations = np.asarray(
        [[[1, 0, 0.25, 0, 0], [0, 1, 0, 1, 0.5], [1, 1, 0.25, 0, 0], [0, 0, 0, 0, 0]]]
        * 2,
        dtype=np.float64,
    )
    equations[..., 2] *= rise
    plane = RobustPlane.Config(
        ridge_fraction=0.25,
        robust_scale=0.0625,
        iterations=iterations,
    ).make()
    np.testing.assert_array_equal(
        plane(np.ones((2, 4)), equations),
        [gradient] * 2,
        strict=True,
    )


def test_support_points_skip_surface_jumps_and_keep_only_the_prefix_tail() -> None:
    config = GeographicPrior.Config(prefix_support_rows=2)
    config.train_bin_size = config.prefix_bin_size = 2
    prior = config.make()
    well = _sloped_wells()[0]
    assert well.depth is not None
    depth = well.depth.copy()
    depth[8:] += np.float32(0.5)
    suffix = prior._support(replace(well, depth=depth), prefix=False)
    # Row 8 jumps 0.5 and leaves bin 4 with row 9 alone, at x 4.5.
    np.testing.assert_array_equal(suffix.values[:, 0], [2.25, 3.25, 4.5, 5.25, 6.25, 7])
    # Each kept half-unit step rises 1/16: a normalized step of 1/8.
    np.testing.assert_array_equal(suffix.values[:, 4], np.full(6, 0.125))
    prefix = prior._support(well, prefix=True)
    # Only visible rows 2 and 3 support the query, as one point at x 1.25.
    np.testing.assert_array_equal(prefix.values[:, 0], [1.25])
    np.testing.assert_array_equal(prefix.is_prefix, [True])


def test_query_interpolates_bin_gradients_onto_rows_and_integrates_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = GeographicPrior.Config(query_bin_size=2).make()
    monkeypatch.setattr(prior, "_predict", _ramp_prediction)
    query, *support = _sloped_wells()
    result = prior.build(
        support,
        queries=[replace(query, suffix_start=3)],
        leave_one_out=False,
    )["well0"]
    # Bins of rows 3-4 to 13-14 center on 3.5 to 13.5, so row r reads (r - 3.5) / 2.
    gradient = np.clip((np.arange(15) - 3.5) / 2, 0, 5)
    gradient[:3] = 0
    np.testing.assert_array_equal(
        result.diagnostics["geo_nbr_std"],
        gradient.astype(np.float32),
        strict=True,
    )
    # Visible rows sit relative to row 2; each half-unit step adds half its gradient.
    expected = np.concatenate([[-0.125, -0.0625], np.cumsum(gradient[2:] / 2)])
    np.testing.assert_array_equal(
        result.values,
        expected.astype(np.float32),
        strict=True,
    )


def _ramp_prediction(
    support: SupportPoints,
    *,
    xy: NDArray[np.float32],
    dxy: NDArray[np.float32],
) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
    """Give query bin k the x gradient k and the neighbor spread k."""
    del support, dxy
    ramp = np.arange(len(xy), dtype=np.float32)
    return np.column_stack((ramp, np.zeros_like(ramp))), {"geo_nbr_std": ramp}


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


def _sloped_wells() -> list[WellTrajectory]:
    """Build three wells along x, ten apart in y, on a surface rising 1/8 per x."""
    row = np.arange(15, dtype=np.float32)
    surface = (20 + row / 16).astype(np.float32)
    z = (-10 + row / 32).astype(np.float32)
    known = surface - z
    known[4:] = np.nan
    return [
        WellTrajectory(
            well_id=f"well{index}",
            md=row,
            x=(row / 2).astype(np.float32),
            y=np.full(15, 10 * index, dtype=np.float32),
            z=z,
            known_depth=known,
            depth=surface - z,
            suffix_start=4,
        )
        for index in range(3)
    ]


def geographical_wells() -> list[WellTrajectory]:
    """Build the deterministic wells of the geographic tests.

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
