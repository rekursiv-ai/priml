"""Checks for the second engine's raw spectral library preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import math

from pyarrow import Table, parquet

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    second_neutral_mass,
)
from priml.baselines.casmi2026.scripts.second_library import (
    formula_mass,
    prepare_second_library,
)


if TYPE_CHECKING:
    from pathlib import Path


def _write_input(
    path: Path,
    *,
    mzs: list[list[float] | None],
    intensities: list[list[float] | None],
) -> None:
    count = len(mzs)
    parquet.write_table(
        Table.from_pydict(
            {
                "ms2_mzs": mzs,
                "ms2_normalized_intensities": intensities,
                "precursor_mz": [200.0, 300.0, 400.0, 500.0][:count],
                "adduct": ["[M+H]+", "[M-H]-", "unknown", "[M+Na]+"][:count],
                "molecular_formula": ["C2H6O", None, "Xx2", ""][:count],
                "ionization_mode": ["positive", "negative", "positive", "positive"][
                    :count
                ],
                "instrument_type": ["timsTOF", "Orbitrap", None, "Tims TOF"][:count],
                "inchikey14": ["KEYB", "KEYA", "MISSING", None][:count],
            },
        ),
        path,
        row_group_size=2,
    )


def _write_pool(path: Path) -> None:
    parquet.write_table(
        Table.from_pydict({"key": ["KEYA", None, "KEYB", "KEYA"]}),
        path,
    )


def test_prepare_writes_raw_peaks_aligned_metadata_and_pool_rows(
    tmp_path: Path,
) -> None:
    source, pool, output = (
        tmp_path / "in.parquet",
        tmp_path / "pool.parquet",
        tmp_path / "o",
    )
    _write_input(
        source,
        mzs=[[100.0, 50.0], None, [10.0], []],
        intensities=[[1.0, 0.5], None, [0.25], []],
    )
    _write_pool(pool)
    prepare_second_library(source, pool=pool, output=output)
    np.testing.assert_array_equal(np.load(output / "off.npy"), [0, 2, 2, 3, 3])
    np.testing.assert_array_equal(np.load(output / "mz.npy"), [100.0, 50.0, 10.0])
    np.testing.assert_array_equal(np.load(output / "it.npy"), [1.0, 0.5, 0.25])
    np.testing.assert_array_equal(
        np.load(output / "prec.npy"),
        [200.0, 300.0, 400.0, 500.0],
    )
    np.testing.assert_array_equal(np.load(output / "pol.npy"), [1, -1, 1, 1])
    np.testing.assert_array_equal(
        np.load(output / "tims.npy"),
        [True, False, False, True],
    )
    np.testing.assert_array_equal(np.load(output / "row_p.npy"), [2, 0, -1, -1])
    fallback = second_neutral_mass(
        np.array([200.0, 300.0, 400.0, 500.0]),
        ["[M+H]+", "[M-H]-", "unknown", "[M+Na]+"],
    )
    neutral = np.load(output / "nm.npy")
    assert (
        neutral[0]
        == 2 * ELEMENT_MASSES["C"] + 6 * ELEMENT_MASSES["H"] + (ELEMENT_MASSES["O"])
    )
    np.testing.assert_array_equal(neutral[1], fallback[1])
    assert math.isnan(neutral[2])
    np.testing.assert_array_equal(neutral[3], fallback[3])
    metadata = parquet.read_table(output / "metadata.parquet")
    assert metadata.column_names == [
        "precursor_mz",
        "adduct",
        "molecular_formula",
        "ionization_mode",
        "instrument_type",
        "inchikey14",
    ]
    assert metadata.num_rows == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert (manifest["rows"], manifest["peaks"], manifest["raw_peaks"]) == (4, 3, True)
    assert set(manifest["generator"]) == {"second_library.py"}
    assert set(manifest["inputs"]) == {str(source.resolve()), str(pool.resolve())}


@pytest.mark.parametrize(
    ("mzs", "intensities"),
    [([[1.0]], [[]]), ([None], [[1.0]]), ([[1.0]], [None])],
)
def test_prepare_rejects_peak_lists_of_different_lengths(
    tmp_path: Path,
    mzs: list[list[float] | None],
    intensities: list[list[float] | None],
) -> None:
    source, pool = tmp_path / "in.parquet", tmp_path / "pool.parquet"
    _write_input(source, mzs=mzs, intensities=intensities)
    _write_pool(pool)
    with pytest.raises(ValueError, match=r"^Peak mass and intensity lengths differ.$"):
        prepare_second_library(source, pool=pool, output=tmp_path / "o")


def test_formula_mass_sums_counts_in_order_and_ignores_punctuation() -> None:
    expected = 2 * ELEMENT_MASSES["C"] + 6 * ELEMENT_MASSES["H"]
    assert formula_mass("C2H6") == expected
    assert formula_mass("C2 H6") == expected
    assert formula_mass("NaCl") == ELEMENT_MASSES["Na"] + ELEMENT_MASSES["Cl"]


@pytest.mark.parametrize("value", [None, "", "C2Xx"])
def test_formula_mass_is_nan_when_empty_or_an_element_is_unknown(
    value: str | None,
) -> None:
    assert math.isnan(formula_mass(value))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
