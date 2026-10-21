"""Check fresh inference, row prediction and publication."""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING, Final, cast

import csv

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.wellbore_geology.data import PredictionRows, RawWell
from priml.baselines.wellbore_geology.folds import RelativeSurface
from priml.baselines.wellbore_geology.inference import (
    ArrayColumn,
    FamilyArtifacts,
    GeographicGate,
    InferenceRecipe,
    PredictionBuilder,
    PredictionTable,
    PublishResults,
    SingleFamilyPrediction,
    SurfaceSmoothing,
    TableBlend,
    WellBlend,
    diagnostic_mean,
    ordered_table,
    publish_tables,
    read_table,
    submission_rows,
    write_table,
)
from priml.testing.golden import mismatches
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


FAMILY_ROWS: Final = {
    "0719_V1": [1e8, -0.0, 1.25],
    "0724_V1": [1.0, 2.5, 1e8],
    "0729_V3": [8.0, -1.25, -1e8],
    "0801_V1": [-1e8, 5.0, 6.25],
    "0801_V2": [2.0, 4.5, -1e8],
    "0803_V2": [4.0, 6.5, 9.5],
}
"""Three rows per family whose cancellation exposes regrouped float32 arithmetic."""


@pytest.mark.parametrize(
    ("name", "boundary"),
    [
        ("geo_nbr_distance", 2459),
        ("geo_nbr_distance_q10", 1426),
        ("geo_nbr_path_alignment", 0.96),
        ("geo_radial_extrap_score", 1.45),
        ("geo_prefix_weight_ratio", 0.315),
    ],
)
def test_each_geographic_bound_is_strict(name: str, boundary: float) -> None:
    diagnostics = {
        "geo_nbr_distance": 100.0,
        "geo_nbr_distance_q10": 50.0,
        "geo_nbr_path_alignment": 0.99,
        "geo_radial_extrap_score": 1.0,
        "geo_prefix_weight_ratio": 0.1,
    }
    gate = GeographicGate.Config().make()
    assert gate(diagnostics)
    diagnostics[name] = boundary
    assert not gate(diagnostics)
    diagnostics[name] = float("nan")
    assert not gate(diagnostics)


@pytest.mark.parametrize("geographic", [False, True])
def test_blend_uses_all_members_in_reference_order(geographic: bool) -> None:
    predictions = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in FAMILY_ROWS.items()
    }
    result = WellBlend.Config().make()(predictions, geographic=geographic)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        _blend_of_family_rows(geographic=geographic),
        strict=True,
    )


def _blend_of_family_rows(*, geographic: bool) -> NDArray[np.float32]:
    return np.asarray(
        [2.0, 14.75 / 3.5, 9.5 / 3.5]
        if geographic
        else [0.0, 4.375 / 2.5, -49_999_992.0 / 2.5],
        dtype=np.float32,
    )


def test_blend_rejects_misaligned_predictions() -> None:
    cfg = WellBlend.Config()
    cfg.fallback = [("a", 1.0), ("b", 1.0)]
    predictions = {
        "a": np.asarray([1.0, 2.0], dtype=np.float32),
        "b": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="aligned"):
        cfg.make()(predictions, geographic=False)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), float("nan")])
def test_blend_rejects_invalid_weights(weight: float) -> None:
    cfg = WellBlend.Config()
    cfg.fallback = [("a", weight)]
    with pytest.raises(ValueError, match="weights"):
        cfg.make()


def test_blend_rejects_empty_members() -> None:
    cfg = WellBlend.Config()
    cfg.geographic = []
    with pytest.raises(ValueError, match="members"):
        cfg.make()


def test_recipe_defaults_are_a_plain_single_device_prediction() -> None:
    recipe = InferenceRecipe.Config().make()
    assert recipe.config.seed_offset == 100_000
    assert recipe.config.batch_size == -1
    assert recipe.config.dtype_autocast is None
    assert isinstance(recipe.config.ema, NoEMA.Config)
    assert isinstance(recipe.config.parallelism, NoParallel.Config)


def test_native_artifacts_feed_pooled_metric_and_complete_submission(
    tmp_path: Path,
) -> None:
    prediction = np.asarray([1.25, -2.5, 4.75], dtype=np.float32)
    rows = PredictionRows(
        well_ids=("b", "a", "b"),
        submit_indices=np.asarray([8, 4, 3], dtype=np.int64),
        tvt=prediction,
        z=np.asarray([2, 3, 5], dtype=np.float32),
        target=np.zeros(3, dtype=np.float32),
    )
    oof, test = tmp_path / "family-oof.parquet", tmp_path / "family-test.parquet"
    write_table(oof, PredictionTable(rows=rows))
    write_table(test, PredictionTable(rows=rows))
    sample = tmp_path / "sample.csv"
    sample.write_text("id,tvt\nb_8,0\na_4,0\nb_3,0\n")
    config = PublishResults.Config()
    config.prediction = SingleFamilyPrediction.Config()
    result = config.make().publish(
        tmp_path,
        files={"0803_V2": FamilyArtifacts(oof=oof, test=test)},
        sample_path=sample,
    )
    assert result == {"rmse": ((1.25**2 + 2.5**2 + 4.75**2) / 3) ** 0.5}
    with (tmp_path / "submission.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [record["id"] for record in records] == ["b_8", "a_4", "b_3"]
    assert [float(record["tvt"]) for record in records] == [1.25, -2.5, 4.75]
    restored = read_table(tmp_path / "oof.parquet")
    assert restored.rows.well_ids == ("a", "b", "b")
    np.testing.assert_array_equal(
        restored.final,
        np.asarray([-2.5, 4.75, 1.25], dtype=np.float64),
        strict=True,
    )


@pytest.mark.parametrize(
    ("particle", "diagnostics"),
    [(False, 0), (False, 3), (False, 4), (True, 4)],
)
def test_prediction_artifacts_keep_every_column_through_arrow(
    tmp_path: Path,
    *,
    particle: bool,
    diagnostics: int,
) -> None:
    oof, test = _prediction_artifacts(
        particle=particle,
        diagnostics=diagnostics,
        blend=0.7261960515627183,
    )
    assert "TVT" not in test.column_order
    for name, frame in (("oof", oof), ("test", test)):
        path = tmp_path / f"{name}.parquet"
        write_table(path, frame)
        restored = read_table(path)
        assert restored.column_order == frame.column_order
        assert (
            mismatches(
                _capture(frame, prefix=name),
                actual=_capture(restored, prefix=name),
            )
            == []
        )


def test_builder_expands_bins_into_physical_rows_sidecars_and_lookups() -> None:
    config = _small_builder()
    config.expansion.target_stats = (1.0, 2.0)
    config.smoothing.blend = 0
    config.diagnostic_names = ("geo_nbr_distance",)
    builder = config.make()
    # Suffix rows 0 to 5 read bins 2, 2, 3, 3, 4 and 4; row 6 repeats bin 4.
    bins = np.asarray([[9, 9, 1, 2, 3, 9], [9, 9, 4, 5, 6, 9]], dtype=np.float32)
    metadata: list[dict[str, object]] = [
        {
            "well_id": well,
            "suffix_len": count,
            "submit_indices": np.arange(4, 4 + count),
            "suffix_z": np.zeros(count, dtype=np.float32),
            "tvt0": anchor,
            "TVT": np.full(count, anchor + 3, dtype=np.float32),
            "geo_prior_TVT": np.full(count, anchor + 2, dtype=np.float32),
            "pf_tvt_rel_pred": np.asarray([9, 9, 0.5, 0.25, 0.75, 9], dtype=np.float32),
        }
        for well, count, anchor in (("bravo", 5, 1000.0), ("alpha", 7, 1100.0))
    ]
    prior = {
        well: RelativeSurface(
            values=np.zeros(11, dtype=np.float32),
            diagnostics={
                "geo_nbr_distance": (np.arange(11) / 16).astype(np.float32),
            },
        )
        for well in ("bravo", "alpha")
    }
    # Gamma is twice the depth less 1900 on both curves, which stop at 1110.
    curves = {
        well: RawWell(
            well_id=well,
            horizontal={},
            reference_depth=990 + np.arange(121.0),
            reference_gamma=80 + 2 * np.arange(121.0),
        )
        for well in ("bravo", "alpha")
    }
    table = builder.lookup_gamma(
        builder.attach_diagnostics(
            builder(bins, metadata=metadata, include_target=True),
            prior=prior,
        ),
        curves=curves,
    )
    expanded = np.asarray([1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6], dtype=np.float32)
    anchor = np.repeat(np.asarray([1000, 1100], dtype=np.float32), [5, 7])
    relative = expanded * 2 + 1
    physical = anchor + relative
    particle = anchor + np.asarray(
        [0.5, 0.5, 0.25, 0.25, 0.75, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75, 0.75],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(table.rows.tvt, physical, strict=True)
    np.testing.assert_array_equal(table.rows.target, anchor + 3, strict=True)
    columns = table.diagnostics
    for name, values in (
        ("pred", expanded),
        ("target_pred", relative),
        ("TVT_pred_raw", physical),
        ("pred_sg_smooth", physical),
        ("PF_TVT", particle),
        ("geo_prior_TVT", anchor + 2),
        ("a", np.full(12, 2.0)),
        ("b", anchor.astype(np.float64) + 1),
        ("target_a", np.full(12, 2.0)),
        ("target_b", np.full(12, 1.0)),
        (
            "geo_nbr_distance",
            np.concatenate([np.arange(4, 9), np.arange(4, 11)]).astype(np.float32) / 16,
        ),
        # Depths beyond 1110 lie past the curve and look up nothing.
        ("pred_lookup_GR", np.where(physical <= 1110, 2 * physical - 1900, np.nan)),
        ("PF_lookup_GR", 2 * particle - 1900),
        ("geo_prior_lookup_GR", 2 * (anchor + 2) - 1900),
    ):
        np.testing.assert_array_equal(columns[name], values, strict=True)
    assert table.strings == {"target_mode": ("unet",) * 12}


def test_repeats_average_each_row_keep_the_first_sidecars_and_look_up_again() -> None:
    columns = ("well_id", "submit_index", "Z", "TVT_pred", "pred", "a", "TVT")
    tables = [
        PredictionTable(
            rows=PredictionRows(
                well_ids=("bravo", "alpha"),
                submit_indices=np.asarray([4, 5], dtype=np.int64),
                tvt=np.asarray([1000 + shift, 1100 - shift], dtype=np.float32),
                z=np.zeros(2, dtype=np.float32),
                target=np.asarray([1, 2], dtype=np.float32),
            ),
            diagnostics={
                "pred": np.asarray([shift, -shift], dtype=np.float32),
                "a": np.full(2, shift, dtype=np.float64),
                "pred_lookup_GR": np.full(2, 7, dtype=np.float32),
            },
            strings={"target_mode": (f"repeat{shift}",) * 2},
            column_order=(*columns[:6], "pred_lookup_GR", *columns[6:]),
        )
        for shift in (-1, 0, 4)
    ]
    curves = {
        well: RawWell(
            well_id=well,
            horizontal={},
            reference_depth=990 + np.arange(121.0),
            reference_gamma=80 + 2 * np.arange(121.0),
        )
        for well in ("bravo", "alpha")
    }
    averaged = _small_builder().make().average_repeats(tables, curves=curves, repeats=3)
    # Each row's three shifts -1, 0 and 4 average to 1.
    assert averaged.rows.well_ids == ("bravo", "alpha")
    np.testing.assert_array_equal(averaged.rows.submit_indices, [4, 5])
    tvt = np.asarray([1001, 1099], dtype=np.float32)
    np.testing.assert_array_equal(averaged.rows.tvt, tvt, strict=True)
    np.testing.assert_array_equal(
        averaged.diagnostics["pred"],
        np.asarray([1, -1], dtype=np.float32),
        strict=True,
    )
    np.testing.assert_array_equal(averaged.diagnostics["a"], [-1.0, -1.0], strict=True)
    np.testing.assert_array_equal(
        averaged.diagnostics["pred_lookup_GR"],
        2 * tvt - 1900,
        strict=True,
    )
    assert averaged.strings == {"target_mode": ("repeat-1",) * 2}
    assert averaged.column_order == tables[0].column_order


def test_distance_summary_repeats_each_wells_tenth_percentile() -> None:
    nan = np.nan
    table = PredictionTable(
        rows=PredictionRows(
            well_ids=("a",) * 5 + ("b",) * 2 + ("c",),
            submit_indices=np.arange(8),
            tvt=np.zeros(8, dtype=np.float32),
            z=np.zeros(8, dtype=np.float32),
            target=None,
        ),
        diagnostics={
            "geo_nbr_distance": np.asarray(
                [4, 0, 1, 3, 2, nan, 5, nan],
                dtype=np.float32,
            ),
        },
    )
    summary = _small_builder().make().summarize_distance(table)
    # Well a's tenth percentile lies 0.4 of the way from 0 to 1; b keeps its one
    # finite distance and c has none.
    np.testing.assert_array_equal(
        summary.diagnostics["geo_nbr_distance_q10"],
        np.asarray([0.4] * 5 + [5] * 2 + [nan], dtype=np.float32),
        strict=True,
    )


def _prediction_artifacts(
    *,
    particle: bool,
    diagnostics: int,
    blend: float,
) -> tuple[PredictionTable, PredictionTable]:
    config = PredictionBuilder.Config()
    config.expansion.prefix_rows = 4
    config.expansion.target_rows = 6
    config.expansion.downsample = 2
    config.expansion.target_stats = (1.591691, 15.853182)
    config.smoothing.blend = blend
    config.diagnostic_names = config.diagnostic_names[:diagnostics]
    builder = config.make()
    bins = np.arange(12, dtype=np.float32).reshape((2, 6)) / np.float32(
        32,
    ) - np.float32(0.2)
    curves = _curves()
    repeated: list[PredictionTable] = []
    for repeat in range(3):
        metadata, prior = _observations(
            repeat=repeat,
            particle=particle,
            names=config.diagnostic_names,
        )
        frame = builder(
            bins + np.float32(repeat / 128),
            metadata=metadata,
            include_target=True,
        )
        frame = builder.attach_diagnostics(frame, prior=prior)
        assert frame.column_order == (
            "well_id",
            "submit_index",
            "Z",
            "TVT_pred",
            *(("PF_TVT",) if particle else ()),
            "geo_prior_TVT",
            "pred",
            "target_pred",
            "target_mode",
            "a",
            "b",
            "target_a",
            "target_b",
            "TVT",
            "TVT_pred_raw",
            "TVT_sg_surface",
            "pred_sg_smooth",
            *config.diagnostic_names,
        )
        repeated.append(builder.lookup_gamma(frame, curves=curves))
    oof = builder.average_repeats(repeated, curves=curves, repeats=3)
    if diagnostics:
        oof = builder.summarize_distance(oof)
    metadata, prior = _observations(
        repeat=0,
        particle=particle,
        names=config.diagnostic_names,
    )
    test = builder(bins, metadata=metadata, include_target=False)
    test = builder.attach_diagnostics(test, prior=prior)
    if diagnostics:
        test = builder.summarize_distance(test)
    return oof, builder.lookup_gamma(test, curves=curves)


def test_complete_metadata_is_coerced_at_the_original_precision_boundary() -> None:
    config = _small_builder()
    metadata, _ = _observations(repeat=0, particle=True, names=())
    for meta in metadata:
        for name in ("suffix_z", "TVT", "geo_prior_TVT", "pf_tvt_rel_pred"):
            meta[name] = np.asarray(meta[name], dtype=np.float64)
        meta["submit_indices"] = np.asarray(meta["submit_indices"], dtype=np.int32)
    table = config.make()(
        np.zeros((2, 6), dtype=np.float32),
        metadata=metadata,
        include_target=True,
    )
    assert table.rows.submit_indices.dtype == np.int64
    assert table.rows.z.dtype == np.float32
    assert table.rows.target is not None
    assert table.rows.target.dtype == np.float32
    assert table.diagnostics["PF_TVT"].dtype == np.float32
    assert table.diagnostics["geo_prior_TVT"].dtype == np.float32
    assert set(table.diagnostics) == {
        "PF_TVT",
        "geo_prior_TVT",
        "pred",
        "target_pred",
        "a",
        "b",
        "target_a",
        "target_b",
        "TVT_pred_raw",
        "TVT_sg_surface",
        "pred_sg_smooth",
    }


@pytest.mark.parametrize("field", ["submit_indices", "suffix_z", "geo_prior_TVT"])
def test_incomplete_original_metadata_is_rejected(field: str) -> None:
    metadata, _ = _observations(repeat=0, particle=False, names=())
    metadata[0][field] = np.zeros(3, dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=r"^Original prediction (metadata|sidecar) lengths differ\.$",
    ):
        _small_builder().make()(
            np.zeros((2, 6), dtype=np.float32),
            metadata=metadata,
            include_target=False,
        )


@pytest.mark.parametrize("metadata_count", [1, 3])
def test_model_rows_and_complete_metadata_require_equal_counts(
    metadata_count: int,
) -> None:
    metadata, _ = _observations(repeat=0, particle=False, names=())
    metadata = (metadata + metadata[:1])[:metadata_count]
    with pytest.raises(ValueError, match="zip"):
        _small_builder().make()(
            np.zeros((2, 6), dtype=np.float32),
            metadata=metadata,
            include_target=False,
        )


def test_repeat_aggregation_requires_the_declared_complete_row_multiplicity() -> None:
    config = _small_builder()
    builder = config.make()
    metadata, _ = _observations(repeat=0, particle=False, names=())
    table = builder(
        np.zeros((2, 6), dtype=np.float32),
        metadata=metadata,
        include_target=True,
    )
    with pytest.raises(
        ValueError,
        match=r"^Every OOF row must have the configured repeat count\.$",
    ):
        builder.average_repeats([table, table], curves={}, repeats=3)
    with pytest.raises(
        ValueError,
        match=r"^At least one complete prediction table is required\.$",
    ):
        builder.average_repeats([], curves={}, repeats=3)
    single = builder.average_repeats([table], curves={}, repeats=1)
    assert (
        mismatches(
            _capture(table, prefix="rows"),
            actual=_capture(single, prefix="rows"),
        )
        == []
    )
    assert single.column_order == table.column_order


def _small_builder() -> PredictionBuilder.Config:
    config = PredictionBuilder.Config()
    config.expansion.prefix_rows = 4
    config.expansion.target_rows = 6
    config.expansion.downsample = 2
    config.expansion.target_stats = (1.591691, 15.853182)
    return config


@pytest.mark.parametrize("summarize", [False, True])
def test_build_annotates_summarizes_then_looks_up_each_well(summarize: bool) -> None:
    config = _small_builder()
    builder = config.make()
    metadata, prior = _observations(
        repeat=0,
        particle=True,
        names=config.diagnostic_names,
    )
    bins = np.arange(12, dtype=np.float32).reshape(2, 6) / np.float32(8)
    curves = _curves()
    built = builder.build(
        bins,
        metadata=metadata,
        include_target=True,
        prior=prior,
        curves=curves,
        summarize=summarize,
    )
    expected = builder.attach_diagnostics(
        builder(bins, metadata=metadata, include_target=True),
        prior=prior,
    )
    if summarize:
        expected = builder.summarize_distance(expected)
    expected = builder.lookup_gamma(expected, curves=curves)
    assert ("geo_nbr_distance_q10" in built.column_order) == summarize
    assert built.column_order == expected.column_order
    assert (
        mismatches(
            _capture(expected, prefix="rows"),
            actual=_capture(built, prefix="rows"),
        )
        == []
    )


def test_gamma_lookup_keeps_tied_depths_in_row_order_and_includes_curve_ends() -> None:
    # Seventeen descending rows reach NumPy's unstable sort kernels; rows 5 and 6
    # tie at depth 15.
    depth = np.arange(20, 3, -1, dtype=np.float64)
    depth[6] = depth[5]
    table = PredictionTable(
        rows=PredictionRows(
            well_ids=("well",) * 3,
            submit_indices=np.arange(3, dtype=np.int64),
            tvt=np.asarray([15, 4, 20], dtype=np.float32),
            z=np.zeros(3, dtype=np.float32),
            target=None,
        ),
    )
    curve = RawWell(
        well_id="well",
        horizontal={},
        reference_depth=depth,
        reference_gamma=np.arange(17, dtype=np.float64) * 10,
    )
    result = _small_builder().make().lookup_gamma(table, curves={"well": curve})
    # Interpolation at a repeated depth returns the gamma of the last tied row.
    # The shallowest and deepest curve rows, 4 and 20, read their own gamma.
    assert result.diagnostics["pred_lookup_GR"].tolist() == [60, 160, 0]


@pytest.mark.parametrize("window", [5, 4])
def test_surface_smoothing_fits_local_lines_and_end_windows(window: int) -> None:
    raw = np.asarray([0, 10, 0, 10, 0, 10, 0, 10, 40], dtype=np.float32) + 1000
    height = np.arange(9, dtype=np.float32) / 4 - 100
    config = SurfaceSmoothing.Config(window=window, blend=0.5)
    frame = config.make()(_well_rows(raw, height=height))
    # Interior rows average five neighbors; the first and last two rows evaluate the
    # least-squares line through the five end rows.
    np.testing.assert_array_equal(
        frame.diagnostics["TVT_sg_surface"],
        np.asarray([4, 4, 4, 6, 4, 6, 12, 20, 28], dtype=np.float32) + 1000,
        strict=True,
    )
    np.testing.assert_array_equal(
        frame.rows.tvt,
        np.asarray([2, 7, 2, 8, 2, 8, 6, 15, 34], dtype=np.float32) + 1000,
        strict=True,
    )
    np.testing.assert_array_equal(frame.diagnostics["TVT_pred_raw"], raw, strict=True)
    np.testing.assert_array_equal(frame.rows.z, height, strict=True)
    np.testing.assert_array_equal(
        frame.rows.tvt,
        frame.diagnostics["pred_sg_smooth"],
        strict=True,
    )


@pytest.mark.parametrize("rows", [9, 10])
def test_surface_window_shrinks_to_the_longest_odd_well_prefix(rows: int) -> None:
    raw = np.asarray([0, 10, 0, 10, 0, 10, 0, 10, 40, 1], dtype=np.float32)[:rows]
    frame = SurfaceSmoothing.Config(window=1025).make()(
        _well_rows(raw, height=np.zeros(rows, dtype=np.float32)),
    )
    smoothed = frame.diagnostics["TVT_sg_surface"]
    # The first nine rows fit one line: mean 80 / 9, slope 8 / 3 per row.
    np.testing.assert_array_equal(
        smoothed[:5],
        np.asarray([(24 * row - 16) / 9 for row in range(5)], dtype=np.float32),
        strict=True,
    )
    assert smoothed[5] == np.float32(81 / 9 if rows == 10 else 104 / 9)


def test_surface_smoothing_keeps_short_wells_and_fits_each_well_separately() -> None:
    # A local quadratic fit reproduces a quadratic; no value is zero, where the
    # fit's float64 residue would survive the cast.
    quadratic = (np.arange(7, dtype=np.float32) + 1) ** 2
    frame = SurfaceSmoothing.Config(window=5, polyorder=2).make()(
        PredictionTable(
            rows=PredictionRows(
                well_ids=("long", "short", *("long",) * 6, "short"),
                submit_indices=np.arange(9, dtype=np.int64),
                tvt=np.asarray(
                    [quadratic[0], 7, *quadratic[1:], -3],
                    dtype=np.float32,
                ),
                z=np.zeros(9, dtype=np.float32),
                target=None,
            ),
        ),
    )
    np.testing.assert_array_equal(
        frame.diagnostics["TVT_sg_surface"],
        np.asarray([quadratic[0], 7, *quadratic[1:], -3], dtype=np.float32),
        strict=True,
    )
    cubic = SurfaceSmoothing.Config(window=3, polyorder=3).make()(
        _well_rows(np.asarray([0, 5, 1], dtype=np.float32), height=np.zeros(3)),
    )
    np.testing.assert_array_equal(
        cubic.diagnostics["TVT_sg_surface"],
        np.asarray([0, 5, 1], dtype=np.float32),
        strict=True,
    )


def _well_rows(
    raw: NDArray[np.float32],
    *,
    height: NDArray[np.generic],
) -> PredictionTable:
    return PredictionTable(
        rows=PredictionRows(
            well_ids=("well",) * len(raw),
            submit_indices=np.arange(len(raw), dtype=np.int64),
            tvt=raw,
            z=height.astype(np.float32),
            target=None,
        ),
    )


def _curves() -> dict[str, RawWell]:
    curves: dict[str, RawWell] = {}
    for index, well in enumerate(("bravo", "alpha")):
        rows = np.arange(-12, 25, dtype=np.float64)
        curves[well] = RawWell(
            well_id=well,
            horizontal={},
            reference_depth=1000 + rows,
            reference_gamma=60 + index + rows**2 / 16,
        )
    return curves


def _observations(
    *,
    repeat: int,
    particle: bool,
    names: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, RelativeSurface]]:
    metadata: list[dict[str, object]] = []
    prior: dict[str, RelativeSurface] = {}
    for index, (well, count) in enumerate((("bravo", 5), ("alpha", 7))):
        suffix = np.arange(count, dtype=np.float32)
        values: dict[str, object] = {
            "well_id": well,
            "suffix_len": count,
            "submit_indices": np.arange(4, 4 + count, dtype=np.int64),
            "suffix_z": suffix / 4 - 100,
            "tvt0": 1000.125 + index / 8,
            "TVT": suffix / 8 + 1001,
            "geo_prior_TVT": suffix / 16 + 1000.25 + np.float32(repeat / 64),
        }
        if particle:
            values["pf_tvt_rel_pred"] = np.arange(6, dtype=np.float32) / 8
        metadata.append(values)
        prior[well] = RelativeSurface(
            values=np.zeros(4 + count, dtype=np.float32),
            diagnostics={
                name: (
                    np.arange(4 + count, dtype=np.float32) / 16
                    + column / 8
                    + repeat / 32
                ).astype(np.float32)
                for column, name in enumerate(names)
            },
        )
    return metadata, prior


def _capture(table: PredictionTable, *, prefix: str) -> dict[str, torch.Tensor]:
    numeric: dict[str, NDArray[np.generic]] = {
        "submit_index": table.rows.submit_indices,
        "Z": table.rows.z,
        "TVT_pred": table.rows.tvt,
        **table.diagnostics,
    }
    if table.rows.target is not None:
        numeric["TVT"] = table.rows.target
    record = {
        f"{prefix}/{name}": torch.from_numpy(values.copy())
        for name, values in numeric.items()
    }
    for name, values in {"well_id": table.rows.well_ids, **table.strings}.items():
        rows = [value.encode("utf-8") for value in values]
        record[f"{prefix}/{name}/utf8"] = torch.tensor(
            list(b"".join(rows)),
            dtype=torch.uint8,
        )
        record[f"{prefix}/{name}/offsets"] = torch.tensor(
            [0, *accumulate(len(value) for value in rows)],
            dtype=torch.int64,
        )
    return record


@pytest.mark.parametrize("columns_declared", [False, True])
@pytest.mark.parametrize("final_exists", [False, True])
def test_single_family_prediction_preserves_all_rows_and_diagnostics(
    columns_declared: bool,
    final_exists: bool,
) -> None:
    columns = ("well_id", "submit_index", "TVT_pred", "Z", "probe")
    table = PredictionTable(
        rows=PredictionRows(
            well_ids=("b", "a", "b"),
            submit_indices=np.asarray([8, 4, 3], dtype=np.int64),
            tvt=np.asarray([7.25, 2.5, -0.0], dtype=np.float32),
            z=np.asarray([1, 2, 3], dtype=np.float32),
            target=None,
        ),
        diagnostics={"probe": np.asarray([3, 5, 7], dtype=np.float64)},
        final=np.ones(3, dtype=np.float64) if final_exists else None,
        column_order=((*columns, "final_pred") if final_exists else columns)
        if columns_declared
        else (),
    )
    actual = SingleFamilyPrediction.Config().make()({"0803_V2": table})
    assert actual.rows.well_ids == ("a", "b", "b")
    np.testing.assert_array_equal(
        actual.rows.submit_indices,
        np.asarray([4, 3, 8], dtype=np.int64),
        strict=True,
    )
    np.testing.assert_array_equal(
        actual.final,
        np.asarray([2.5, -0.0, 7.25], dtype=np.float64),
        strict=True,
    )
    np.testing.assert_array_equal(
        actual.diagnostics["probe"],
        np.asarray([5, 7, 3], dtype=np.float64),
        strict=True,
    )
    assert actual.column_order == ((*columns, "final_pred") if columns_declared else ())
    if final_exists:
        np.testing.assert_array_equal(
            table.final,
            np.ones(3, dtype=np.float64),
            strict=True,
        )
    else:
        assert table.final is None


def tables() -> dict[str, PredictionTable]:
    """Build one row group per strict geographic bound, plus a safe and a NaN well."""
    safe = {
        "geo_nbr_distance": 100.0,
        "geo_nbr_distance_q10": 50.0,
        "geo_nbr_path_alignment": 0.99,
        "geo_radial_extrap_score": 1.0,
        "geo_prefix_weight_ratio": 0.1,
    }
    cases = {"safe": dict(safe)}
    for name, boundary in zip(safe, (2459, 1426, 0.96, 1.45, 0.315), strict=True):
        cases[name] = {**safe, name: boundary}
    cases["nan"] = {**safe, "geo_nbr_distance": float("nan")}
    return {
        name: PredictionTable(
            rows=PredictionRows(
                well_ids=tuple(well for well in cases for _ in range(3)),
                submit_indices=np.tile(np.arange(2, 5, dtype=np.int64), len(cases)),
                tvt=np.tile(np.asarray(values, dtype=np.float32), len(cases)),
                z=np.zeros(3 * len(cases), dtype=np.float32),
                target=np.zeros(3 * len(cases), dtype=np.float32),
            ),
            diagnostics={
                key: np.repeat(
                    np.asarray(
                        [case[key] for case in cases.values()],
                        dtype=np.float64,
                    ),
                    3,
                )
                for key in safe
            },
        )
        for name, values in FAMILY_ROWS.items()
    }


def test_ensemble_sorts_and_aligns_every_family_row() -> None:
    original = tables()
    frame = original["0719_V1"]
    original["0719_V1"] = replace(
        frame,
        rows=replace(
            frame.rows,
            well_ids=tuple(reversed(frame.rows.well_ids)),
            submit_indices=frame.rows.submit_indices[::-1],
            tvt=frame.rows.tvt[::-1],
            z=frame.rows.z[::-1],
            target=frame.rows.target,
        ),
    )
    actual = TableBlend.Config().make()(original)
    reference = original["0801_V2"].rows
    np.testing.assert_array_equal(actual.final, _sorted_ensemble(), strict=True)
    assert len(actual.rows.tvt) == 21
    assert actual.rows.well_ids == tuple(sorted(reference.well_ids))
    np.testing.assert_array_equal(actual.rows.submit_indices, np.tile([2, 3, 4], 7))
    assert all(table.final is None for table in original.values())


@pytest.mark.parametrize(
    ("values", "mean"),
    [
        # Float32 rounds the sum to 5.0999999, a third of which is 1.6999999, not 1.7.
        (np.asarray([1, np.nan, 3, 1.1], dtype=np.float32), np.float32(1.7)),
        # The mean of 1, 1 and 0.1 rounds to 0.7; their float64 sum / 3 rounds up.
        (np.asarray([1, 1, np.nan, 0.1], dtype=np.float64), np.float64(0.7)),
        # Opposite infinities average to NaN in both precisions.
        (np.asarray([np.inf, 1, -np.inf], dtype=np.float32), np.float32(np.nan)),
        (np.asarray([np.inf, 1, -np.inf], dtype=np.float64), np.float64(np.nan)),
        (np.full(2, np.nan, dtype=np.float32), np.float32(np.nan)),
        (np.zeros(0, dtype=np.float64), np.float64(np.nan)),
    ],
)
def test_diagnostic_means_skip_missing_values_and_round_once_in_their_precision(
    values: NDArray[np.float32] | NDArray[np.float64],
    mean: np.float32 | np.float64,
) -> None:
    actual = diagnostic_mean(values)
    assert type(actual) is type(mean)
    np.testing.assert_array_equal(actual, mean, strict=True)


def _cast_diagnostics(
    frame: PredictionTable,
    dtype: type[np.float32 | np.float64],
) -> dict[str, NDArray[np.float32] | NDArray[np.float64]]:
    """Every diagnostic column of ``frame`` cast to ``dtype``."""
    cast_to: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {}
    for key, values in frame.diagnostics.items():
        cast_to[key] = (
            values.astype(np.float32)
            if dtype is np.float32
            else values.astype(np.float64)
        )
    return cast_to


def test_ensemble_gates_float32_and_float64_diagnostics_at_strict_bounds() -> None:
    for dtype in (np.float32, np.float64):
        frames = {
            name: replace(frame, diagnostics=_cast_diagnostics(frame, dtype))
            for name, frame in tables().items()
        }
        final = TableBlend.Config().make()(frames).final
        np.testing.assert_array_equal(final, _sorted_ensemble(), strict=True)


def _sorted_ensemble() -> NDArray[np.float64]:
    """Return the blend of ``tables()`` in sorted well order; ``safe`` is geographic."""
    return np.concatenate(
        [
            np.tile(_blend_of_family_rows(geographic=False), 6),
            _blend_of_family_rows(geographic=True),
        ],
    ).astype(np.float64)


@pytest.mark.parametrize("final_exists", [False, True])
def test_blend_keeps_the_complete_template_strings_and_column_order(
    final_exists: bool,
) -> None:
    frames = tables()
    template = frames["0801_V2"]
    columns = (
        "well_id",
        "submit_index",
        "TVT_pred",
        "Z",
        "TVT",
        *template.diagnostics,
        "target_mode",
    )
    frames["0801_V2"] = replace(
        template,
        strings={"target_mode": tuple(f"unet_{row}" for row in range(21))},
        column_order=(*columns, "final_pred") if final_exists else columns,
        final=template.rows.tvt.astype(np.float64) if final_exists else None,
    )
    blended = TableBlend.Config().make()(frames)
    assert blended.strings == ordered_table(frames["0801_V2"]).strings
    assert blended.column_order == (*columns, "final_pred")


def test_alignment_rejects_missing_and_shifted_original_rows() -> None:
    frames = tables()
    frame = frames["0724_V1"]
    changed = frame.rows.submit_indices.copy()
    changed[0] = 90
    frames["0724_V1"] = replace(frame, rows=replace(frame.rows, submit_indices=changed))
    with pytest.raises(ValueError, match="Family target rows differ: 0724_V1"):
        TableBlend.Config().make()(frames)


@pytest.mark.parametrize("complete", [False, True])
def test_complete_arrow_round_trip_preserves_all_column_bits(
    tmp_path: Path,
    complete: bool,
) -> None:
    table = tables()["0801_V2"]
    if complete:
        table = replace(table, final=table.rows.tvt.astype(np.float64))
        table.diagnostics["float32"] = np.linspace(-1, 1, 21, dtype=np.float32)
        table = replace(
            table,
            strings={"target_mode": tuple(f"unet_{row}" for row in range(21))},
            column_order=(
                "well_id",
                "submit_index",
                "Z",
                "TVT_pred",
                "target_mode",
                "TVT",
                *table.diagnostics,
                "final_pred",
            ),
        )
    else:
        table = replace(table, rows=replace(table.rows, target=None), diagnostics={})
    path = tmp_path / "rows.parquet"
    write_table(path, table)
    if complete:
        assert parquet.read_table(path).column_names == list(table.column_order)
    restored = read_table(path)
    assert restored.rows.well_ids == table.rows.well_ids
    assert restored.strings == table.strings
    if complete:
        assert restored.column_order == table.column_order
        order = np.lexsort((table.rows.submit_indices, np.asarray(table.rows.well_ids)))
        assert ordered_table(table).strings == {
            name: tuple(values[int(index)] for index in order.flat)
            for name, values in table.strings.items()
        }
    for actual, expected in zip(
        (
            restored.rows.submit_indices,
            restored.rows.tvt,
            restored.rows.z,
            restored.rows.target,
        ),
        (table.rows.submit_indices, table.rows.tvt, table.rows.z, table.rows.target),
        strict=True,
    ):
        np.testing.assert_array_equal(
            actual,
            expected,
            strict=True,
        )
    assert list(restored.diagnostics) == list(table.diagnostics)
    for name in table.diagnostics:
        np.testing.assert_array_equal(
            restored.diagnostics[name],
            table.diagnostics[name],
            strict=True,
        )
    np.testing.assert_array_equal(restored.final, table.final, strict=True)


@pytest.mark.parametrize("invalid", ["diagnostic", "final"])
def test_arrow_rejects_incompatible_numeric_column_precision(
    tmp_path: Path,
    invalid: str,
) -> None:
    columns: dict[str, object] = {
        "well_id": ["a"],
        "submit_index": np.asarray([1], dtype=np.int64),
        "TVT_pred": np.asarray([1], dtype=np.float32),
        "Z": np.asarray([0], dtype=np.float32),
    }
    if invalid == "diagnostic":
        columns["bad"] = np.asarray([1], dtype=np.int64)
    else:
        columns["final_pred"] = np.asarray([1], dtype=np.float32)
    path = tmp_path / "invalid.parquet"
    parquet.write_table(pa.Table.from_pydict(columns), path)
    with pytest.raises(
        ValueError,
        match="Diagnostic.*bad"
        if invalid == "diagnostic"
        else "Final prediction.*float64",
    ):
        read_table(path)


@pytest.mark.parametrize("dtype", ["U", "S", "O"])
def test_arrow_text_conversion_uses_object_arrays(dtype: str) -> None:
    values = np.asarray(["alpha", "bravo", "charlie"], dtype=dtype)
    column = cast("ArrayColumn", pa.Table.from_pydict({"text": values}).column("text"))
    actual = column.to_numpy()
    assert actual.dtype.kind == "O"
    np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("duplicate", [False, True])
def test_publication_rejects_incomplete_or_duplicate_column_declarations(
    tmp_path: Path,
    duplicate: bool,
) -> None:
    table = tables()["0801_V2"]
    columns = ("well_id", "submit_index", "TVT_pred", "Z", "TVT", *table.diagnostics)
    table = replace(
        table,
        column_order=(*columns, "well_id") if duplicate else columns[:-1],
    )
    with pytest.raises(ValueError, match="Declared prediction columns"):
        write_table(tmp_path / "invalid.parquet", table)


def test_official_identifiers_require_exact_set_and_keep_sample_order() -> None:
    table = tables()["0801_V2"]
    table = replace(table, final=table.rows.tvt.astype(np.float64))
    identifiers = [
        f"{well}_{row}"
        for well, row in zip(
            table.rows.well_ids,
            table.rows.submit_indices.flat,
            strict=True,
        )
    ]
    actual = submission_rows(table, identifiers=identifiers[::-1])
    np.testing.assert_array_equal(
        actual,
        table.final[::-1] if table.final is not None else None,
    )
    with pytest.raises(ValueError, match="Submission identifiers differ"):
        submission_rows(table, identifiers=identifiers[:-1])
    duplicate = replace(
        table,
        rows=replace(
            table.rows,
            well_ids=(table.rows.well_ids[0],) * len(table.rows.tvt),
            submit_indices=np.zeros_like(table.rows.submit_indices),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate submission identifiers"):
        submission_rows(duplicate, identifiers=identifiers)
    with pytest.raises(ValueError, match="final predictions"):
        submission_rows(replace(table, final=None), identifiers=identifiers)


def test_publication_scores_every_original_row_and_writes_official_csv(
    tmp_path: Path,
) -> None:
    frame = TableBlend.Config().make()(tables())
    sample = tmp_path / "sample.csv"
    identifiers = [
        f"{well}_{row}"
        for well, row in zip(
            frame.rows.well_ids,
            frame.rows.submit_indices.flat,
            strict=True,
        )
    ][::-1]
    with sample.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "tvt"])
        writer.writerows((identifier, 0) for identifier in identifiers)
    result = publish_tables(tmp_path, oof=frame, test=frame, sample=sample)
    assert frame.final is not None
    assert result == float(
        cast("np.float64", np.sqrt(np.mean(frame.final * frame.final))),
    )
    assert read_table(tmp_path / "oof.parquet").rows.well_ids == frame.rows.well_ids
    assert read_table(tmp_path / "test.parquet").rows.well_ids == frame.rows.well_ids
    published = (tmp_path / "submission.csv").read_bytes()
    assert published.startswith(b"id,tvt\n")
    assert b"\r" not in published
    with (tmp_path / "submission.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["id"] for row in rows] == identifiers
    np.testing.assert_array_equal(
        np.asarray([float(row["tvt"]) for row in rows]),
        frame.final[::-1],
    )


@pytest.mark.parametrize("missing", ["target", "final"])
def test_publication_requires_every_target_and_final_prediction(
    tmp_path: Path,
    missing: str,
) -> None:
    frame = TableBlend.Config().make()(tables())
    incomplete = (
        replace(frame, final=None)
        if missing == "final"
        else replace(frame, rows=replace(frame.rows, target=None))
    )
    with pytest.raises(
        ValueError,
        match=r"^OOF scoring requires every target and final prediction\.$",
    ):
        publish_tables(
            tmp_path,
            oof=incomplete,
            test=frame,
            sample=tmp_path / "unused.csv",
        )
    assert not list(tmp_path.iterdir())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
