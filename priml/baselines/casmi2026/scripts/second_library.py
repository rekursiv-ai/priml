"""Prepare the second retrieval engine's complete raw spectral rows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import math
import re

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    FORMULA_ELEMENTS,
    second_neutral_mass,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from pyarrow import parquet
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


_THIS: Final = Path(__file__).resolve()
_ELEMENT_MASSES: Final = {
    element: ELEMENT_MASSES[element]
    for element in (*FORMULA_ELEMENTS, "Si", "Se", "Na", "K", "B", "As", "Fe", "D")
}


def prepare_second_library(input_path: Path, *, pool: Path, output: Path) -> None:
    """Write every raw row against the first matching sorted-pool key.

    Args:
      input_path: Original training Parquet with full, uncleaned peak lists.
      pool: Ordered second-engine pool Parquet containing its complete key column.
      output: New external directory; existing datasets are never overwritten.

    """
    source = parquet.ParquetFile(input_path)
    destination = validated_output_path(output, protected=[input_path, pool])
    destination.mkdir(parents=True)
    keys = from_plain(
        cast(
            "object",
            parquet.read_table(pool, columns=["key"]).column("key").to_pylist(),
        ),
        list[str | None],
    )
    positions: dict[str | None, int] = {}
    for row, key in enumerate(keys):
        if key is not None:
            _ = positions.setdefault(key, row)
    peak_count = 0
    for batch in source.iter_batches(
        batch_size=256,
        columns=["ms2_mzs", "ms2_normalized_intensities"],
    ):
        masses = from_plain(
            cast(object, batch.column(0).to_pylist()),
            list[list[float] | None],
        )
        intensities = from_plain(
            cast(object, batch.column(1).to_pylist()),
            list[list[float] | None],
        )
        for row, mz in enumerate(masses):
            intensity = intensities[row]
            if len(mz or []) != len(intensity or []):
                raise ValueError("Peak mass and intensity lengths differ.")
            peak_count += len(mz or [])
    count = source.metadata.num_rows
    offsets = np.zeros(count + 1, dtype=np.int64)
    mz_values = np.lib.format.open_memmap(
        destination / "mz.npy",
        mode="w+",
        dtype=np.float32,
        shape=(peak_count,),
    )
    intensities = np.lib.format.open_memmap(
        destination / "it.npy",
        mode="w+",
        dtype=np.float32,
        shape=(peak_count,),
    )
    precursor = np.empty(count)
    neutral = np.empty(count)
    pool_rows = np.empty(count, dtype=np.int64)
    polarity = np.empty(count, dtype=np.int8)
    timstof = np.empty(count, dtype=np.bool_)
    metadata_schema = (
        source.read_row_group(0)
        .drop("ms2_mzs")
        .drop(
            "ms2_normalized_intensities",
        )
        .schema
    )
    row_offset, peak_offset = 0, 0
    with parquet.ParquetWriter(
        destination / "metadata.parquet",
        metadata_schema,
    ) as writer:
        for group in range(source.num_row_groups):
            table = source.read_row_group(group)
            writer.write_table(table.drop("ms2_mzs").drop("ms2_normalized_intensities"))
            rows = slice(row_offset, row_offset + table.num_rows)
            precursor[rows] = np.array(
                from_plain(
                    cast(object, table.column("precursor_mz").to_pylist()),
                    list[float | None],
                ),
            )
            fallback = second_neutral_mass(
                precursor[rows],
                from_plain(
                    cast(object, table.column("adduct").to_pylist()),
                    list[str | None],
                ),
            )
            formulas = np.array(
                [
                    formula_mass(value)
                    for value in from_plain(
                        cast(object, table.column("molecular_formula").to_pylist()),
                        list[str | None],
                    )
                ],
            )
            neutral[rows] = np.where(np.isfinite(formulas), formulas, fallback)
            polarity[rows] = [
                1 if value == "positive" else -1
                for value in from_plain(
                    cast(object, table.column("ionization_mode").to_pylist()),
                    list[str | None],
                )
            ]
            timstof[rows] = [
                value is not None
                and value.lower().replace("-", "").replace(" ", "") == "timstof"
                for value in from_plain(
                    cast(object, table.column("instrument_type").to_pylist()),
                    list[str | None],
                )
            ]
            pool_rows[rows] = [
                positions.get(value, -1)
                for value in from_plain(
                    cast(object, table.column("inchikey14").to_pylist()),
                    list[str | None],
                )
            ]
            masses = from_plain(
                cast(object, table.column("ms2_mzs").to_pylist()),
                list[list[float] | None],
            )
            weights = from_plain(
                cast(object, table.column("ms2_normalized_intensities").to_pylist()),
                list[list[float] | None],
            )
            for row, mass in enumerate(masses):
                weight = weights[row]
                end = peak_offset + len(mass or [])
                mz_values[peak_offset:end] = mass or []
                intensities[peak_offset:end] = weight or []
                row_offset += 1
                offsets[row_offset] = end
                peak_offset = end
    for name, array in (
        ("off", offsets),
        ("prec", precursor),
        ("nm", neutral),
        ("row_p", pool_rows),
        ("pol", polarity),
        ("tims", timstof),
    ):
        np.save(destination / f"{name}.npy", array)
    mz_values.flush()
    intensities.flush()
    manifest: dict[str, object] = {
        "format_version": 1,
        "numpy": np.__version__,
        "rows": count,
        "peaks": peak_count,
        "raw_peaks": True,
        "inputs": input_digests([input_path, pool]),
        "generator": source_digests([_THIS]),
    }
    write_manifest(destination, manifest)


def formula_mass(value: str | None) -> float:
    """Sum the public formula tokens in their original left-to-right order.

    Args:
      value: Nullable original formula; punctuation between tokens is ignored.

    Returns:
      mass: Complete double mass, or NaN for empty formulas and unknown elements.

    """
    if value is None or value == "":
        return math.nan
    total = 0.0
    for token in re.finditer(r"([A-Z][a-z]?)(\d*)", value):
        symbol, count = token.group(1, 2)
        if symbol not in _ELEMENT_MASSES:
            return math.nan
        total += _ELEMENT_MASSES[symbol] * (int(count) if count else 1)
    return total
