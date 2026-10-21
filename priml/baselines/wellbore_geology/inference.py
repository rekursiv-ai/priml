"""Fresh family inference, original-row prediction and study publication."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import csv
import io
import math
import statistics

from configgle import Fig, Makeable
from numpy.typing import NDArray
from treekle import from_plain

import numpy as np

from priml.baselines.wellbore_geology.data import (
    ExpandPrediction,
    PredictionRows,
)
from priml.baselines.wellbore_geology.metrics import pooled_rmse
from priml.train.custom_types import EMAProtocol, ParallelStrategyProtocol
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyarrow import parquet

    import pyarrow as pa
    import torch

    from priml.baselines.wellbore_geology.data import RawWell
    from priml.baselines.wellbore_geology.folds import RelativeSurface
else:
    from wrapt import lazy_import

    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")
    torch = lazy_import("torch")


class GeographicGate:
    """Select the geographic ensemble using the reference's five strict bounds."""

    class Config(Fig["GeographicGate"]):
        """Well-mean geographic diagnostics and reference thresholds."""

        distance_max: float = 2459
        """Exclusive upper bound for mean neighbor distance."""

        distance_q10_max: float = 1426
        """Exclusive upper bound for mean tenth-percentile neighbor distance."""

        path_alignment_min: float = 0.96
        """Exclusive lower bound for mean neighbor-path alignment."""

        radial_extrapolation_max: float = 1.45
        """Exclusive upper bound for mean radial extrapolation score."""

        prefix_weight_ratio_max: float = 0.315
        """Exclusive upper bound for mean prefix support-weight ratio."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        diagnostics: Mapping[str, float | np.float32 | np.float64],
    ) -> bool:
        """Return whether a well passes every geographic diagnostic."""
        return bool(
            diagnostics["geo_nbr_distance"] < self.config.distance_max
            and diagnostics["geo_nbr_distance_q10"] < self.config.distance_q10_max
            and diagnostics["geo_nbr_path_alignment"] > self.config.path_alignment_min
            and diagnostics["geo_radial_extrap_score"]
            < self.config.radial_extrapolation_max
            and diagnostics["geo_prefix_weight_ratio"]
            < self.config.prefix_weight_ratio_max,
        )


class WellBlend:
    """Blend one aligned well without changing the notebook's operation order."""

    class Config(Fig["WellBlend"]):
        """Ordered family coefficients for the two notebook branches."""

        geographic: list[tuple[str, float]] = field(
            default_factory=lambda: [
                ("0719_V1", 0.25),
                ("0801_V1", 0.25),
                ("0724_V1", 1.0),
                ("0801_V2", 1.0),
                ("0803_V2", 1.0),
            ],
        )
        """Members in the geographic branch, in arithmetic accumulation order."""

        fallback: list[tuple[str, float]] = field(
            default_factory=lambda: [
                ("0719_V1", 1.0),
                ("0729_V3", 0.5),
                ("0801_V1", 1.0),
            ],
        )
        """Members in the GR-only branch, in arithmetic accumulation order."""

    def __init__(self, config: Config) -> None:
        self.config = config
        for members in (config.geographic, config.fallback):
            if not members or any(
                weight <= 0 or not math.isfinite(weight) for _, weight in members
            ):
                raise ValueError("Blend members must have finite positive weights.")

    def __call__(
        self,
        predictions: Mapping[str, NDArray[np.float32]],
        *,
        geographic: bool,
    ) -> NDArray[np.float32]:
        """Return float32 predictions for one well in submission-row order.

        Args:
          predictions: Aligned float32 predictions for each family.
          geographic: Whether the well passed every geographic diagnostic.

        Returns:
          prediction: Ordered float32 weighted mean for the selected branch.

        """
        members = self.config.geographic if geographic else self.config.fallback
        first = predictions[members[0][0]]
        prediction = np.zeros_like(first)
        weight_sum = 0.0
        for name, weight in members:
            values = predictions[name]
            if values.shape != first.shape:
                raise ValueError("Family predictions must be aligned by well and row.")
            prediction += values * weight
            weight_sum += weight
        prediction /= weight_sum
        return prediction


class ArrayColumn(Protocol):
    """The complete Arrow array conversion missing from its upstream stub."""

    def to_numpy(self) -> NDArray[np.generic]:
        """Return every ChunkedArray row, allowing copies as its default does."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionTable:
    """Owned original-row observations and every geographic diagnostic."""

    rows: PredictionRows
    diagnostics: dict[str, NDArray[np.float32] | NDArray[np.float64]] = field(
        default_factory=dict,
    )
    final: NDArray[np.float64] | None = None
    strings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    column_order: tuple[str, ...] = ()


class TableBlend:
    """Align complete family tables before applying the ordered geographic blend."""

    class Config(Fig["TableBlend"]):
        diagnostic_family: str = "0801_V2"
        """Family supplying complete target metadata and geographic diagnostics."""

        gate: GeographicGate.Config = field(default_factory=GeographicGate.Config)
        """The five public strict geographic bounds."""

        blend: WellBlend.Config = field(default_factory=WellBlend.Config)
        """Complete ordered public coefficients for both geographic branches."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.gate = config.gate.make()
        self.blend = config.blend.make()

    def __call__(self, tables: Mapping[str, PredictionTable]) -> PredictionTable:
        """Return every aligned original row and its final float64 prediction."""
        aligned = {name: ordered_table(table) for name, table in tables.items()}
        template = aligned[self.config.diagnostic_family]
        for name, table in aligned.items():
            if table.rows.well_ids != template.rows.well_ids or not np.array_equal(
                table.rows.submit_indices,
                template.rows.submit_indices,
            ):
                raise ValueError(f"Family target rows differ: {name}.")
        final = np.zeros(len(template.rows.tvt), dtype=np.float64)
        groups: dict[str, list[int]] = {}
        for index, well in enumerate(template.rows.well_ids):
            groups.setdefault(well, []).append(index)
        for indices in groups.values():
            rows = np.asarray(indices, dtype=np.int64)
            diagnostics = {
                name: diagnostic_mean(values[rows])
                for name, values in template.diagnostics.items()
            }
            final[rows] = self.blend(
                {name: table.rows.tvt[rows] for name, table in aligned.items()},
                geographic=self.gate(diagnostics),
            )
        return PredictionTable(
            rows=template.rows,
            diagnostics=template.diagnostics,
            final=final,
            strings=template.strings,
            column_order=(
                *(name for name in template.column_order if name != "final_pred"),
                "final_pred",
            )
            if template.column_order
            else (),
        )


def ordered_table(table: PredictionTable) -> PredictionTable:
    """Copy every column into lexicographic well and original-row order.

    Args:
      table: Complete row-bearing arrays and diagnostic columns.

    Returns:
      ordered: Owned columns with a shared stable original-row permutation.

    """
    rows = table.rows
    order = np.lexsort((rows.submit_indices, np.asarray(rows.well_ids)))
    return PredictionTable(
        rows=PredictionRows(
            well_ids=tuple(rows.well_ids[int(index)] for index in order.flat),
            submit_indices=rows.submit_indices[order],
            tvt=rows.tvt[order],
            z=rows.z[order],
            target=rows.target[order] if rows.target is not None else None,
        ),
        diagnostics={name: values[order] for name, values in table.diagnostics.items()},
        final=table.final[order] if table.final is not None else None,
        strings={
            name: tuple(values[int(index)] for index in order.flat)
            for name, values in table.strings.items()
        },
        column_order=table.column_order,
    )


def diagnostic_mean[Float: np.float32 | np.float64](values: NDArray[Float]) -> Float:
    """Average every nonmissing diagnostic, rounded once to its precision.

    Args:
      values: Complete diagnostic observations for one well.

    Returns:
      mean: Float64 mean of float32 values rounded once, the exactly rounded mean
        of float64 values, or NaN when empty.

    """
    observed = cast("list[float]", values[~np.isnan(values)].tolist())
    if not observed:
        return values.dtype.type(math.nan)
    if values.dtype == np.float32:
        # math.fsum raises on inf + -inf, whose mean is NaN in the float64 branch too.
        try:
            total = math.fsum(observed)
        except ValueError:
            total = math.nan
        return values.dtype.type(total / len(observed))
    # The sum rounds before its division and misrounds a quarter of float64 means.
    return values.dtype.type(statistics.mean(observed))


def write_table(path: Path, table: PredictionTable) -> None:
    """Write every original row and column into a complete Arrow file.

    Args:
      path: Complete output Arrow table filename.
      table: Every original row, diagnostic and optional final prediction.

    """
    rows = table.rows
    columns: dict[str, object] = {
        "well_id": list(rows.well_ids),
        "submit_index": rows.submit_indices,
        "TVT_pred": rows.tvt,
        "Z": rows.z,
    }
    if rows.target is not None:
        columns["TVT"] = rows.target
    columns.update(table.diagnostics)
    columns.update(table.strings)
    if table.final is not None:
        columns["final_pred"] = table.final
    if table.column_order:
        if set(table.column_order) != set(columns) or len(table.column_order) != len(
            columns,
        ):
            raise ValueError("Declared prediction columns differ from complete data.")
        columns = {name: columns[name] for name in table.column_order}
    parquet.write_table(pa.Table.from_pydict(columns), path)


def read_table(path: Path) -> PredictionTable:
    """Read every original row and column from a complete Arrow file.

    Args:
      path: Complete native Arrow artifact.

    Returns:
      table: Complete owned rows, diagnostic arrays and optional final output.

    """
    table = parquet.read_table(path)
    arrays = {
        name: cast("ArrayColumn", table.column(name)).to_numpy()
        for name in table.column_names
        if name != "well_id"
    }
    diagnostics: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {}
    strings: dict[str, tuple[str, ...]] = {}
    for name, values in arrays.items():
        if name not in {"submit_index", "TVT_pred", "Z", "TVT", "final_pred"}:
            if values.dtype == np.float32:
                diagnostics[name] = cast("NDArray[np.float32]", values.copy())
            elif values.dtype == np.float64:
                diagnostics[name] = cast("NDArray[np.float64]", values.copy())
            elif values.dtype.kind == "O":
                strings[name] = tuple(cast("list[str]", table.column(name).to_pylist()))
            else:
                raise ValueError(
                    f"Diagnostic must have float32 or float64 precision: {name}.",
                )
    return PredictionTable(
        rows=PredictionRows(
            well_ids=tuple(cast("list[str]", table.column("well_id").to_pylist())),
            submit_indices=arrays["submit_index"].astype(np.int64),
            tvt=arrays["TVT_pred"].astype(np.float32),
            z=arrays["Z"].astype(np.float32),
            target=arrays["TVT"].astype(np.float32) if "TVT" in arrays else None,
        ),
        diagnostics=diagnostics,
        final=final_column(arrays["final_pred"]) if "final_pred" in arrays else None,
        strings=strings,
        column_order=tuple(table.column_names),
    )


def final_column(values: NDArray[np.generic]) -> NDArray[np.float64]:
    """Require the serialized final-column precision before exposing its values."""
    if values.dtype != np.float64:
        raise ValueError("Final prediction column requires float64 precision.")
    return cast("NDArray[np.float64]", values.copy())


def submission_rows(
    table: PredictionTable,
    *,
    identifiers: Sequence[str],
) -> NDArray[np.float64]:
    """Require the complete official ID set and return its exact row ordering.

    Args:
      table: Every complete blended original row.
      identifiers: Complete official sample IDs in their required output order.

    Returns:
      prediction: Complete float64 final values in the sample ID order.

    """
    if table.final is None:
        raise ValueError("Submission requires complete final predictions.")
    original = [
        f"{well}_{row}"
        for well, row in zip(
            table.rows.well_ids,
            table.rows.submit_indices.flat,
            strict=True,
        )
    ]
    if len(set(original)) != len(original):
        raise ValueError("Duplicate submission identifiers.")
    if set(original) != set(identifiers) or len(identifiers) != len(original):
        raise ValueError("Submission identifiers differ from the official sample.")
    positions = {identifier: index for index, identifier in enumerate(original)}
    return table.final[
        np.asarray(
            [positions[identifier] for identifier in identifiers],
            dtype=np.int64,
        )
    ]


def publish_tables(
    directory: Path,
    *,
    oof: PredictionTable,
    test: PredictionTable,
    sample: Path,
) -> float:
    """Publish complete OOF/test tables, pooled RMSE and the official CSV.

    Args:
      directory: External output directory for complete final artifacts.
      oof: Complete blended original validation rows with their true depths.
      test: Complete blended original test rows.
      sample: Official sample CSV defining the exact identifier order.

    Returns:
      score: Float64 pooled original-row RMSE over the full validation split.

    """
    if oof.rows.target is None or oof.final is None:
        raise ValueError("OOF scoring requires every target and final prediction.")
    identifiers = [row["id"] for row in csv.DictReader(sample.read_text().splitlines())]
    predictions = submission_rows(test, identifiers=identifiers)
    score = pooled_rmse(oof.rows.target, prediction=oof.final)
    write_table(directory / "oof.parquet", oof)
    write_table(directory / "test.parquet", test)
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(("id", "tvt"))
    writer.writerows(
        (identifier, predictions.flat[index])
        for index, identifier in enumerate(identifiers)
    )
    (directory / "submission.csv").write_text(text.getvalue())
    return score


class InferenceRecipe:
    """Hold the printed differences between held-out training and fresh prediction."""

    class Config(Fig["InferenceRecipe"]):
        seed_offset: int = 100_000
        """Inference loader generator seed offset from the family seed."""

        batch_size: int = -1
        """Wells per inference batch; -1 inherits the held-out training batch size."""

        dtype_autocast: torch.dtype | None = None
        """Inference precision; None keeps binary32 end to end."""

        ema: Makeable[EMAProtocol] = field(default_factory=NoEMA.Config)
        """Averaging policy of the reconstructed models; none averages selected states."""

        parallelism: Makeable[ParallelStrategyProtocol] = field(
            default_factory=NoParallel.Config,
        )
        """Placement of the reconstructed models; the family fills the runtime device."""

    def __init__(self, config: Config) -> None:
        self.config = config


class TableCombiner(Protocol):
    """Combine complete family original-row tables into final predictions."""

    def __call__(self, tables: Mapping[str, PredictionTable]) -> PredictionTable:
        """Return complete final original rows from all configured families."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilyArtifacts:
    """Complete native files emitted by one trained public family."""

    oof: Path
    test: Path


class PublishResults:
    """Combine every family before pooled scoring and exact sample-ID export."""

    class Config(Fig["PublishResults"]):
        prediction: Makeable[TableCombiner] = field(default_factory=TableBlend.Config)
        """Ordered complete original-row prediction combiner."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def publish(
        self,
        directory: Path,
        *,
        files: Mapping[str, FamilyArtifacts],
        sample_path: Path,
    ) -> dict[str, float]:
        """Publish complete OOF/test tables and the official submission.

        Args:
          directory: Existing external run directory receiving final artifacts.
          files: Complete per-family native OOF and inference artifact paths.
          sample_path: Official sample CSV specifying the exact complete ID order.

        Returns:
          metrics: Official pooled float64 original-row RMSE.

        """
        combine = self.config.prediction.make()
        oof = combine({name: read_table(paths.oof) for name, paths in files.items()})
        test = combine({name: read_table(paths.test) for name, paths in files.items()})
        return {
            "rmse": publish_tables(directory, oof=oof, test=test, sample=sample_path),
        }


class SurfaceSmoothing:
    """Smooth predicted absolute surfaces with float64 Savitzky-Golay fits."""

    class Config(Fig["SurfaceSmoothing"]):
        window: int = 577
        """Requested odd number of original rows in each local polynomial fit."""

        polyorder: int = 1
        """Degree of the local surface polynomial."""

        blend: float = 0.7261960515627183
        """Mass of the smoothed surface correction in the final depth."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, table: PredictionTable) -> PredictionTable:
        """Return every raw and smoothed original-row column.

        Args:
          table: Complete physical predictions and heights in original row order.

        Returns:
          smoothed: Complete table with the three surface correction sidecars.

        """
        raw = table.rows.tvt.astype(np.float64)
        height = table.rows.z.astype(np.float64)
        candidate = raw.copy()
        for indices in _well_groups(table).values():
            rows = np.asarray(indices, dtype=np.int64)
            surface = raw[rows] + height[rows]
            window = self.config.window + (self.config.window % 2 == 0)
            window = min(window, len(rows) if len(rows) % 2 else len(rows) - 1)
            if len(rows) >= 3 and window >= 3 and window > self.config.polyorder:
                surface = _savgol(
                    surface,
                    window=window,
                    polyorder=self.config.polyorder,
                )
            candidate[rows] = surface - height[rows]
        final = (raw + self.config.blend * (candidate - raw)).astype(np.float32)
        added = {
            "TVT_pred_raw": table.rows.tvt.copy(),
            "TVT_sg_surface": candidate.astype(np.float32),
            "pred_sg_smooth": final,
        }
        return replace(
            table,
            rows=replace(table.rows, tvt=final),
            diagnostics={**table.diagnostics, **added},
            column_order=_appended_columns(table, names=tuple(added)),
        )


class PredictionBuilder:
    """Retain all public row fields, repeat means and raw reference lookups."""

    class Config(Fig["PredictionBuilder"]):
        expansion: ExpandPrediction.Config = field(
            default_factory=ExpandPrediction.Config,
        )
        """Normalized bin geometry and physical target normalization."""

        smoothing: SurfaceSmoothing.Config = field(
            default_factory=SurfaceSmoothing.Config,
        )
        """Original-row surface smoothing before repeat averaging."""

        diagnostic_names: tuple[str, ...] = (
            "geo_nbr_distance",
            "geo_radial_extrap_score",
            "geo_nbr_path_alignment",
            "geo_prefix_weight_ratio",
        )
        """Ordered inference-safe pointwise fields retained by this family."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        prediction: NDArray[np.float32],
        *,
        metadata: Sequence[dict[str, object]],
        include_target: bool,
    ) -> PredictionTable:
        """Expand complete bins, smooth them and retain every sidecar.

        Args:
          prediction: Every normalized horizontal bin for each input well.
          metadata: Complete identifiers, anchors, suffix heights and prior fields.
          include_target: Retain every physical validation target.

        Returns:
          table: Every original numeric and text field in its publication order.

        """
        expansion = self.config.expansion.make()
        pieces: list[PredictionTable] = []
        for index, meta in zip(range(len(prediction)), metadata, strict=True):
            count = from_plain(meta["suffix_len"], int)
            relative = (
                expansion.expand(prediction[index, :], count=count)
                * expansion.target_std
                + expansion.target_mean
            )
            anchor = from_plain(meta["tvt0"], float)
            physical = np.float32(anchor) + relative
            columns: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {}
            if "pf_tvt_rel_pred" in meta:
                columns["PF_TVT"] = np.float32(anchor) + expansion.expand(
                    np.asarray(meta["pf_tvt_rel_pred"]),
                    count=count,
                )
            if "geo_prior_TVT" in meta:
                columns["geo_prior_TVT"] = np.asarray(
                    meta["geo_prior_TVT"],
                    dtype=np.float32,
                )
            columns.update(
                pred=(relative - expansion.target_mean) / expansion.target_std,
                target_pred=relative,
                a=np.full(count, expansion.target_std),
                b=np.full(count, anchor + expansion.target_mean),
                target_a=np.full(count, expansion.target_std),
                target_b=np.full(count, expansion.target_mean),
            )
            rows = PredictionRows(
                well_ids=(from_plain(meta["well_id"], str),) * count,
                submit_indices=np.asarray(meta["submit_indices"], dtype=np.int64),
                tvt=physical,
                z=np.asarray(meta["suffix_z"], dtype=np.float32),
                target=np.asarray(meta["TVT"], dtype=np.float32)
                if include_target
                else None,
            )
            if any(len(values) != count for values in (rows.submit_indices, rows.z)):
                raise ValueError("Original prediction metadata lengths differ.")
            if any(len(values) != count for values in columns.values()):
                raise ValueError("Original prediction sidecar lengths differ.")
            pieces.append(
                PredictionTable(
                    rows=rows,
                    diagnostics=columns,
                    strings={"target_mode": ("unet",) * count},
                    column_order=(
                        "well_id",
                        "submit_index",
                        "Z",
                        "TVT_pred",
                        *(name for name in columns if name.endswith("_TVT")),
                        "pred",
                        "target_pred",
                        "target_mode",
                        "a",
                        "b",
                        "target_a",
                        "target_b",
                        *(("TVT",) if include_target else ()),
                    ),
                ),
            )
        table = _concatenate(pieces)
        return self.config.smoothing.make()(table)

    def build(
        self,
        prediction: NDArray[np.float32],
        *,
        metadata: Sequence[dict[str, object]],
        include_target: bool,
        prior: Mapping[str, RelativeSurface],
        curves: Mapping[str, RawWell],
        summarize: bool,
    ) -> PredictionTable:
        """Expand, smooth, annotate and look up one complete batch of wells.

        Args:
          prediction: Every normalized horizontal bin for each input well.
          metadata: Complete identifiers, anchors, suffix heights and prior fields.
          include_target: Retain every physical validation target.
          prior: Every complete raw-row geographic diagnostic for each input well.
          curves: Original float64 paired curves before feature calibration.
          summarize: Add each well's neighbor-distance quantile, when it is declared.

        Returns:
          table: Every original-row field the family publishes for these wells.

        """
        table = self(prediction, metadata=metadata, include_target=include_target)
        table = self.attach_diagnostics(table, prior=prior)
        if summarize and self.config.diagnostic_names:
            table = self.summarize_distance(table)
        return self.lookup_gamma(table, curves=curves)

    def attach_diagnostics(
        self,
        table: PredictionTable,
        *,
        prior: Mapping[str, RelativeSurface],
    ) -> PredictionTable:
        """Attach configured complete original-row geographic observations.

        Args:
          table: Complete predicted suffix rows.
          prior: Every complete raw-row diagnostic for each input well.

        Returns:
          annotated: Original table with every configured pointwise field.

        """
        columns = {
            name: np.full(len(table.rows.tvt), np.nan, dtype=np.float32)
            for name in self.config.diagnostic_names
        }
        for well, indices in _well_groups(table).items():
            rows = np.asarray(indices, dtype=np.int64)
            for name, values in columns.items():
                values[rows] = prior[well].diagnostics[name][
                    table.rows.submit_indices[rows]
                ]
        return replace(
            table,
            diagnostics={**table.diagnostics, **columns},
            column_order=_appended_columns(table, names=tuple(columns)),
        )

    def lookup_gamma(
        self,
        table: PredictionTable,
        *,
        curves: Mapping[str, RawWell],
    ) -> PredictionTable:
        """Look up every predicted depth in the complete uncalibrated paired curve.

        Args:
          table: Complete raw, particle and geographic physical predictions.
          curves: Original float64 paired curves before feature calibration.

        Returns:
          annotated: Every input field plus all available curve lookup columns.

        """
        inputs = {"pred_lookup_GR": table.rows.tvt}
        for name in ("PF", "geo_prior"):
            if f"{name}_TVT" in table.diagnostics:
                inputs[f"{name}_lookup_GR"] = table.diagnostics[f"{name}_TVT"].astype(
                    np.float32,
                )
        columns = {
            name: np.full(len(values), np.nan, dtype=np.float32)
            for name, values in inputs.items()
        }
        for well, indices in _well_groups(table).items():
            rows = np.asarray(indices, dtype=np.int64)
            curve = curves[well]
            order = np.argsort(curve.reference_depth, kind="stable")
            depth, gamma = curve.reference_depth[order], curve.reference_gamma[order]
            lower = cast("np.float64", depth[0])
            upper = cast("np.float64", depth[-1])
            for name, values in inputs.items():
                query = values[rows].astype(np.float64)
                valid = np.isfinite(query) & (query >= lower) & (query <= upper)
                columns[name][rows[valid]] = np.interp(
                    query[valid],
                    depth,
                    gamma,
                ).astype(
                    np.float32,
                )
        return replace(
            table,
            diagnostics={**table.diagnostics, **columns},
            column_order=_appended_columns(table, names=tuple(columns)),
        )

    def average_repeats(
        self,
        tables: Sequence[PredictionTable],
        *,
        curves: Mapping[str, RawWell],
        repeats: int,
    ) -> PredictionTable:
        """Average complete repeated rows before recomputing reference lookups.

        Args:
          tables: Every complete fold output in repeat-major insertion order.
          curves: Original float64 paired curves used to recompute gamma lookups.
          repeats: Required observations per original validation row.

        Returns:
          averaged: Complete first-appearance rows and all averaged sidecars.

        """
        combined = _concatenate(tables)
        if repeats == 1:
            return combined
        groups: dict[tuple[str, int], list[int]] = {}
        for index, (well, row) in enumerate(
            zip(combined.rows.well_ids, combined.rows.submit_indices.flat, strict=True),
        ):
            groups.setdefault((well, int(row)), []).append(index)
        if any(len(rows) != repeats for rows in groups.values()):
            raise ValueError("Every OOF row must have the configured repeat count.")
        first = np.asarray([rows[0] for rows in groups.values()], dtype=np.int64)
        columns: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {}
        averaged_names = {
            "pred",
            "target_pred",
            "TVT_pred_raw",
            "TVT_sg_surface",
            "pred_sg_smooth",
            *self.config.diagnostic_names,
        }
        for name, values in combined.diagnostics.items():
            if name.endswith("_lookup_GR"):
                continue
            if name in averaged_names or name.endswith("_TVT"):
                columns[name] = _group_means(values, groups=tuple(groups.values()))
            else:
                columns[name] = values[first]
        prediction = _group_means(combined.rows.tvt, groups=tuple(groups.values()))
        result = PredictionTable(
            rows=PredictionRows(
                well_ids=tuple(well for well, _ in groups),
                submit_indices=combined.rows.submit_indices[first],
                tvt=prediction,
                z=combined.rows.z[first],
                target=combined.rows.target[first]
                if combined.rows.target is not None
                else None,
            ),
            diagnostics=columns,
            strings={
                name: tuple(values[int(index)] for index in first.flat)
                for name, values in combined.strings.items()
            },
            column_order=tuple(
                name
                for name in combined.column_order
                if not name.endswith("_lookup_GR")
            ),
        )
        result = self.lookup_gamma(result, curves=curves)
        return replace(result, column_order=combined.column_order)

    def summarize_distance(self, table: PredictionTable) -> PredictionTable:
        """Repeat each complete well's tenth-percentile neighbor distance.

        Args:
          table: Complete original rows after repeated observations are averaged.

        Returns:
          summarized: Every row plus its well's float32 tenth-percentile distance.

        """
        distance = table.diagnostics["geo_nbr_distance"]
        quantiles = np.full(len(distance), np.nan, dtype=np.float32)
        for indices in _well_groups(table).values():
            rows = np.asarray(indices, dtype=np.int64)
            values = distance[rows].astype(np.float64)
            values = values[~np.isnan(values)]
            quantiles[rows] = np.quantile(values, 0.1) if len(values) else math.nan
        return replace(
            table,
            diagnostics={**table.diagnostics, "geo_nbr_distance_q10": quantiles},
            column_order=_appended_columns(table, names=("geo_nbr_distance_q10",)),
        )


class SingleFamilyPrediction:
    """Select one complete family without claiming the winning ensemble."""

    class Config(Fig["SingleFamilyPrediction"]):
        family: str = "0803_V2"
        """Family whose complete original-row prediction becomes the output."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, tables: Mapping[str, PredictionTable]) -> PredictionTable:
        """Preserve every original row and metadata field of the chosen family."""
        table = ordered_table(tables[self.config.family])
        return replace(
            table,
            final=table.rows.tvt.astype(np.float64),
            column_order=(
                *(name for name in table.column_order if name != "final_pred"),
                "final_pred",
            )
            if table.column_order
            else (),
        )


def _savgol(
    values: NDArray[np.float64],
    *,
    window: int,
    polyorder: int,
) -> NDArray[np.float64]:
    """Fit a centered least-squares polynomial; edge rows use the end window's fit."""
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    vandermonde = offsets ** np.arange(polyorder + 1, dtype=np.float64)[:, None]
    weights = np.linalg.lstsq(vandermonde, np.eye(polyorder + 1)[0], rcond=None)[0]
    smoothed = np.empty_like(values)
    smoothed[half : len(values) - half] = np.correlate(values, weights, mode="valid")
    position = np.arange(window, dtype=np.float64)
    head = np.polyfit(position, values[:window], polyorder)
    tail = np.polyfit(position, values[-window:], polyorder)
    smoothed[:half] = np.polyval(head, position[:half])
    smoothed[len(values) - half :] = np.polyval(tail, position[window - half :])
    return smoothed


def _well_groups(table: PredictionTable) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, well in enumerate(table.rows.well_ids):
        groups.setdefault(well, []).append(index)
    return groups


def _appended_columns(
    table: PredictionTable,
    *,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    if not table.column_order:
        return ()
    return (
        *table.column_order,
        *(name for name in names if name not in table.column_order),
    )


def _concatenate(tables: Sequence[PredictionTable]) -> PredictionTable:
    if not tables:
        raise ValueError("At least one complete prediction table is required.")
    first = tables[0]
    return PredictionTable(
        rows=PredictionRows(
            well_ids=tuple(well for table in tables for well in table.rows.well_ids),
            submit_indices=np.concatenate(
                [table.rows.submit_indices for table in tables],
            ),
            tvt=np.concatenate([table.rows.tvt for table in tables]),
            z=np.concatenate([table.rows.z for table in tables]),
            target=np.concatenate(
                [
                    table.rows.target
                    for table in tables
                    if table.rows.target is not None
                ],
            )
            if first.rows.target is not None
            else None,
        ),
        diagnostics={
            name: cast(
                "NDArray[np.float32] | NDArray[np.float64]",
                np.concatenate([table.diagnostics[name] for table in tables]),
            )
            for name in first.diagnostics
        },
        strings={
            name: tuple(value for table in tables for value in table.strings[name])
            for name in first.strings
        },
        column_order=first.column_order,
    )


def _group_means[Float: np.float32 | np.float64](
    values: NDArray[Float],
    *,
    groups: tuple[list[int], ...],
) -> NDArray[np.float32]:
    return np.asarray(
        [diagnostic_mean(values[np.asarray(rows, dtype=np.int64)]) for rows in groups],
        dtype=np.float32,
    )
