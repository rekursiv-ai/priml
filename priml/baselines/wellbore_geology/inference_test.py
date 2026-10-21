"""Check fresh inference, row prediction and publication."""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate
from pathlib import Path
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
from priml.baselines.wellbore_geology.metrics import pooled_rmse
from priml.testing.golden import mismatches, read_tensors
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


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
        "0719_V1": np.asarray([1e8, -0.0, 1.25], dtype=np.float32),
        "0724_V1": np.asarray([1.0, 2.5, 1e8], dtype=np.float32),
        "0729_V3": np.asarray([8.0, -1.25, -1e8], dtype=np.float32),
        "0801_V1": np.asarray([-1e8, 5.0, 6.25], dtype=np.float32),
        "0801_V2": np.asarray([2.0, 4.5, -1e8], dtype=np.float32),
        "0803_V2": np.asarray([4.0, 6.5, 9.5], dtype=np.float32),
    }
    result = WellBlend.Config().make()(predictions, geographic=geographic)
    # The cancellation in these inputs makes regrouped arithmetic observably wrong.
    expected = np.asarray(
        [2.0, 14.75 / 3.5, 9.5 / 3.5]
        if geographic
        else [0.0, 4.375 / 2.5, -49_999_992.0 / 2.5],
        dtype=np.float32,
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected, strict=True)


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
    ("family", "particle", "diagnostics", "window", "blend"),
    [
        ("0719_v1", False, 0, 1025, 0.6828376753426914),
        ("0724_v1", False, 3, 1025, 0.6828376753426914),
        ("0729_v3", True, 4, 1025, 0.6828376753426914),
        ("0801_v1", False, 4, 577, 0.7261960515627183),
        ("0801_v2", False, 4, 577, 0.7261960515627183),
        ("0803_v2", True, 4, 577, 0.7261960515627183),
    ],
)
def test_complete_original_prediction_artifacts(
    tmp_path: Path,
    *,
    family: str,
    particle: bool,
    diagnostics: int,
    window: int,
    blend: float,
) -> None:
    config = PredictionBuilder.Config()
    config.expansion.prefix_rows = 4
    config.expansion.target_rows = 6
    config.expansion.downsample = 2
    config.expansion.target_stats = (1.591691, 15.853182)
    config.smoothing.window = window
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
    test = builder.lookup_gamma(test, curves=curves)
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
    assert oof.rows.target is not None
    oof_rmse = pooled_rmse(oof.rows.target, prediction=oof.rows.tvt)
    record = {
        **_capture(oof, prefix="oof"),
        **_capture(test, prefix="test"),
        "oof_rmse": torch.tensor(oof_rmse, dtype=torch.float64),
    }
    report = mismatches(
        read_tensors(_CWD / "testdata" / f"predictions_{family}.pt"),
        actual=record,
    )
    assert not report, "\n".join(report)


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


@pytest.mark.parametrize(
    ("family", "window", "blend"),
    [("0719_v1", 1025, 0.6828376753426914), ("0803_v2", 577, 0.7261960515627183)],
)
def test_complete_original_unclipped_surface_smoothing(
    family: str,
    window: int,
    blend: float,
) -> None:
    rows = np.arange(1031, dtype=np.float32)
    raw = (1000 + (rows % 17) / 19 + (rows // 17) / 31).astype(np.float32)
    height = (rows / 13 - 100).astype(np.float32)
    config = SurfaceSmoothing.Config()
    config.window = window
    config.blend = blend
    frame = config.make()(
        PredictionTable(
            rows=PredictionRows(
                well_ids=("well",) * len(rows),
                submit_indices=np.arange(len(rows), dtype=np.int64),
                tvt=raw,
                z=height,
                target=None,
            ),
        ),
    )
    record = {
        "TVT_sg_surface": torch.from_numpy(frame.diagnostics["TVT_sg_surface"]),
        "TVT_pred": torch.from_numpy(frame.rows.tvt),
        "window": torch.tensor(config.window),
        "polyorder": torch.tensor(config.polyorder),
        "blend": torch.tensor(config.blend, dtype=torch.float64),
    }
    report = mismatches(
        read_tensors(_CWD / "testdata" / f"smoothing_{family}.pt"),
        actual=record,
    )
    assert not report, "\n".join(report)
    np.testing.assert_array_equal(frame.diagnostics["TVT_pred_raw"], raw, strict=True)
    np.testing.assert_array_equal(frame.rows.z, height, strict=True)
    np.testing.assert_array_equal(
        frame.rows.tvt,
        frame.diagnostics["pred_sg_smooth"],
        strict=True,
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
    """Build every exact-boundary case represented in the immutable source PT."""
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
    predictions = {
        "0719_V1": [1e8, -0.0, 1.25],
        "0724_V1": [1.0, 2.5, 1e8],
        "0729_V3": [8.0, -1.25, -1e8],
        "0801_V1": [-1e8, 5.0, 6.25],
        "0801_V2": [2.0, 4.5, -1e8],
        "0803_V2": [4.0, 6.5, 9.5],
    }
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
        for name, values in predictions.items()
    }


def test_complete_original_ensemble_rows_and_alignment_are_unchanged() -> None:
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
    ordering = np.lexsort((reference.submit_indices, np.asarray(reference.well_ids)))
    expected = read_tensors(_CWD / "testdata" / "components.pt")["ensemble"].numpy()[
        ordering
    ]
    np.testing.assert_array_equal(actual.final, expected, strict=True)
    assert len(actual.rows.tvt) == 21
    assert actual.rows.well_ids == tuple(sorted(reference.well_ids))
    np.testing.assert_array_equal(actual.rows.submit_indices, np.tile([2, 3, 4], 7))
    assert all(table.final is None for table in original.values())


def test_complete_grouped_means_keep_each_original_column_precision() -> None:
    values = np.linspace(0.9, 1.1, 35, dtype=np.float32)
    values[2::11] = np.nan
    diagnostics = {
        "geo_nbr_distance": values * np.float32(2459),
        "geo_radial_extrap_score": values * np.float32(1.45),
        "geo_nbr_path_alignment": values * np.float32(0.96),
        "geo_nbr_distance_q10": values * np.float32(1426),
        "geo_prefix_weight_ratio": values * np.float32(0.315),
    }
    record: dict[str, torch.Tensor] = {}
    for dtype in (np.float32, np.float64):
        for name, data in diagnostics.items():
            groups = np.asarray(
                [diagnostic_mean(data.astype(dtype)[group::3]) for group in range(3)],
                dtype=dtype,
            )
            record[f"{np.dtype(dtype).name}/{name}"] = torch.from_numpy(groups)
    report = mismatches(
        read_tensors(_CWD / "testdata" / "diagnostic_mean.pt"),
        actual=record,
    )
    assert not report, "\n".join(report)


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


def test_complete_ensemble_preserves_source_column_precision_at_strict_bounds() -> None:
    record: dict[str, torch.Tensor] = {}
    for dtype in (np.float32, np.float64):
        frames = {
            name: replace(frame, diagnostics=_cast_diagnostics(frame, dtype))
            for name, frame in tables().items()
        }
        final = TableBlend.Config().make()(frames).final
        assert final is not None
        record[np.dtype(dtype).name] = torch.from_numpy(final)
    report = mismatches(
        read_tensors(_CWD / "testdata" / "ensemble_precision.pt"),
        actual=record,
    )
    assert not report, "\n".join(report)


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
