"""Checks for the chemistry module."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import hashlib
import importlib
import json
import math
import sys
import types

from pyarrow import Table, parquet
from treekle import ReadError, from_plain, json_loads, to_plain

import numpy as np
import pytest

from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    ChemicalTable,
    IdentityTable,
    MainSearch,
    OfficialIdentity,
    PeakStore,
    SecondSearch,
    SpectralLibrary,
    clean_block,
    clean_main,
    instrument_family,
    merge_main,
)
from priml.baselines.casmi2026.custom_types import ReactionRequest
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    Reaction,
    normalize_request,
    reaction_rules,
)
from priml.baselines.casmi2026.scripts import (
    derivation_chemistry,
    derivatives,
    identity_data,
    rdkit_backend,
    reaction_data,
)
from priml.baselines.casmi2026.scripts.derivation_chemistry import (
    DerivationChemistry,
)
from priml.baselines.casmi2026.scripts.derivatives import (
    Derivative,
    DerivativeGenerator,
)
from priml.baselines.casmi2026.scripts.graph_properties import atom_properties
from priml.baselines.casmi2026.scripts.products import unique_products
from priml.baselines.casmi2026.scripts.rdkit_backend import (
    Chemistry,
    MolecularContext,
)
from priml.baselines.casmi2026.testing import (
    identity_fingerprints,
    identity_rows,
    write_identity,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from rdkit import Chem

    from priml.baselines.casmi2026.custom_types import IdentityRow


_CWD: Final = Path(__file__).resolve().parent


def test_complete_numeric_chemical_assets_and_selected_bit_order(
    tmp_path: Path,
) -> None:
    parquet.write_table(
        Table.from_pydict(
            {
                "input_smiles": ["CCO", "OCC", "bad"],
                "smiles": ["CCO", "CCO", ""],
                "valid": [True, True, False],
                "formula": ["C2H6O", "C2H6O", ""],
                "mass": [46.041864812, 46.041864812, float("nan")],
                "heavy_atoms": np.array([3, 3, 0], dtype=np.int32),
                "key": ["engine", "engine", ""],
                "raw_key": ["raw", "raw", ""],
                "metric_key": ["metric", "metric", ""],
                "composition": ['[["C",2],["O",1]]', '[["C",2],["O",1]]', "[]"],
            },
        ),
        tmp_path / "structures.parquet",
    )
    np.save(
        tmp_path / "raw_fp.npy",
        np.array([[132, 128], [132, 128], [0, 0]], np.uint8),
    )
    _ = (tmp_path / "fingerprint_bits.txt").write_text("9\n")
    table = ChemicalTable.load(tmp_path)
    assert table.input_smiles == ("CCO", "OCC", "bad")
    assert table.smiles == ("CCO", "CCO", "")
    assert table.formula == ("C2H6O", "C2H6O", "")
    assert table.keys == ("engine", "engine", "")
    assert table.raw_keys == ("raw", "raw", "")
    assert table.metric_keys == ("metric", "metric", "")
    assert table.compositions == ((("C", 2), ("O", 1)), (("C", 2), ("O", 1)), ())
    np.testing.assert_array_equal(table.valid, [True, True, False])
    np.testing.assert_array_equal(
        table.mass,
        [46.041864812, 46.041864812, float("nan")],
    )
    np.testing.assert_array_equal(table.heavy_atoms, np.array([3, 3, 0], np.int32))
    assert table.mass.dtype == np.float64
    assert table.heavy_atoms.dtype == np.int32
    assert table.raw_bits == 9
    assert table.metric_identity("CCO") == ("metric", (("C", 2), ("O", 1)))
    assert table.metric_identity("OCC") == table.metric_identity("CCO")
    assert table.metric_identity("bad") is None
    with pytest.raises(KeyError, match="CN"):
        _ = table.metric_identity("CN")
    np.testing.assert_array_equal(
        table.fingerprints(
            np.array([1, 2, 0], np.int64),
            np.array([8, 0, 5], np.int64),
        ),
        [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
    )
    for bits in ([-1], [9]):
        with pytest.raises(
            ValueError,
            match="Selected fingerprint bits exceed the generated layout",
        ):
            _ = table.fingerprints(np.array([0], np.int64), np.array(bits, np.int64))
    # ChemicalTable.load must reject two fingerprint rows for three molecules.
    np.save(tmp_path / "raw_fp.npy", np.zeros((2, 2), np.uint8))
    with pytest.raises(
        ValueError,
        match=r"^Generated fingerprint shape differs from metadata\.$",
    ):
        _ = ChemicalTable.load(tmp_path)
    # ChemicalTable.load requires two packed bytes for the declared nine bits.
    np.save(tmp_path / "raw_fp.npy", np.zeros((3, 3), np.uint8))
    with pytest.raises(
        ValueError,
        match=r"^Generated fingerprint shape differs from metadata\.$",
    ):
        _ = ChemicalTable.load(tmp_path)
    _ = (tmp_path / "fingerprint_bits.txt").write_text("0\n")
    with pytest.raises(
        ValueError,
        match=r"^Generated fingerprint width must be positive\.$",
    ):
        _ = ChemicalTable.load(tmp_path)
    for width in (1, 8):
        _ = (tmp_path / "fingerprint_bits.txt").write_text(f"{width}\n")
        # ChemicalTable.load packs widths one through eight into one byte.
        np.save(tmp_path / "raw_fp.npy", np.zeros((3, 1), np.uint8))
        assert ChemicalTable.load(tmp_path).raw_bits == width
    empty = parquet.read_table(tmp_path / "structures.parquet").slice(0, 0)
    parquet.write_table(empty, tmp_path / "structures.parquet")
    # ChemicalTable.load accepts an empty molecule table with one packed byte.
    np.save(tmp_path / "raw_fp.npy", np.zeros((0, 1), np.uint8))
    assert ChemicalTable.load(tmp_path).valid.dtype == np.bool_


def test_clean_block_keeps_the_strongest_ties_in_encounter_order() -> None:
    values = np.arange(19, dtype=np.float64)
    offsets, mz, intensity = clean_block(
        np.array([0, 19], dtype=np.int64),
        mz=values % 5 + 1,
        intensity=values % 3 + 1,
        precursor=np.array([20.0]),
        floor=0.001,
        topk=7,
        over=2.0,
    )
    # Six intensity-three peaks and the last intensity-two peak (index 16) survive.
    np.testing.assert_array_equal(offsets, [0, 7])
    np.testing.assert_array_equal(mz, np.array([1, 2, 2, 3, 3, 4, 5], np.float32))
    np.testing.assert_array_equal(
        intensity,
        np.array([1, 2 / 3, 1, 1, 1, 1, 1], np.float32),
    )


def test_instrument_labels_map_to_the_six_main_engine_families() -> None:
    labels = (
        None,
        np.nan,
        3,
        "timsTOF",
        "ftms",
        "itft",
        "Exactive",
        "FT-ICR",
        "fticr",
        "QTOF",
        "trap",
        "lit",
        "quattro",
        "triple",
        "unmapped",
        "Orbitrap TOF",
    )
    assert [instrument_family(label) for label in labels] == [
        5,
        5,
        5,
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        3,
        4,
        4,
        5,
        1,
    ]


def test_cleaning_keeps_finite_peaks_and_an_empty_block() -> None:
    offsets, mz, intensity = clean_block(
        np.array([0, 3], dtype=np.int64),
        mz=np.array([1.0, 2.0, np.nan]),
        intensity=np.array([2.0, np.nan, 4.0]),
        precursor=np.array([20.0]),
        floor=0.001,
        topk=512,
        over=2.0,
    )
    # The NaN mass exceeds no cap but fails every comparison, so peak 0 alone stays.
    np.testing.assert_array_equal(offsets, [0, 1])
    np.testing.assert_array_equal(mz, np.array([1.0], np.float32))
    np.testing.assert_array_equal(intensity, np.array([1.0], np.float32))
    offsets, mz, intensity = clean_block(
        np.array([0], dtype=np.int64),
        mz=np.array([], dtype=np.float64),
        intensity=np.array([], dtype=np.float64),
        precursor=np.array([], dtype=np.float64),
        floor=0.001,
        topk=512,
        over=2.0,
    )
    np.testing.assert_array_equal(offsets, [0])
    assert mz.dtype == intensity.dtype == np.float32
    assert mz.shape == intensity.shape == (0,)


def observe_identity(table: IdentityTable) -> list[tuple[object, ...]]:
    """Read every staged identity row back through the public native lookups."""
    observed: list[tuple[object, ...]] = []
    for expected in identity_rows():
        value = expected["smiles"]
        row = table.for_smiles(value)
        standardized = table.standardize(value)
        fingerprint = (
            table.fingerprint(standardized) if standardized is not None else None
        )
        observed.append(
            (
                row["smiles"],
                standardized,
                table.key(value),
                row["standard_key"],
                table.formula(value),
                json.dumps(to_plain(table.identity(value)), separators=(",", ":")),
                row["standard_metric"],
                table.properties(standardized) if standardized is not None else None,
                None if fingerprint is None else fingerprint.tolist(),
            ),
        )
    return observed


def staged_identity() -> list[tuple[object, ...]]:
    """Return what ``observe_identity`` reads back from the staged identity rows."""
    fingerprints = identity_fingerprints()
    return [
        (
            row["smiles"],
            row["standardized"],
            row["key"],
            row["standard_key"],
            row["formula"],
            row["metric"],
            row["standard_metric"],
            (row["standard_formula"], row["standard_mass"], row["standard_heavy"])
            if row["properties_valid"]
            else None,
            fingerprints[index].tolist() if row["fingerprint_valid"] else None,
        )
        for index, row in enumerate(identity_rows())
    ]


def test_complete_identity_and_product_table_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    write_identity(tmp_path)
    table = IdentityTable.Config(working_dir=tmp_path).make()
    assert observe_identity(table) == staged_identity()
    for row in identity_rows():
        standardized = row["standardized"]
        if standardized is not None:
            fingerprint = table.fingerprint(standardized)
            if fingerprint is not None:
                assert not fingerprint.flags.writeable
            assert table.key(standardized) == row["standard_key"]
            assert table.formula(standardized) == row["standard_formula"]
            assert (
                json.dumps(
                    to_plain(table.identity(standardized)),
                    separators=(",", ":"),
                )
                == row["standard_metric"]
            )
    assert table.key("*C") is None
    assert table.fingerprint("*C") is not None
    assert table.identity("*C") is None
    assert table.fingerprint("") is not None
    assert table.properties("") is not None
    for lookup in (
        table.for_smiles,
        table.standardize,
        table.key,
        table.formula,
        table.identity,
        table.properties,
        table.fingerprint,
    ):
        with pytest.raises(KeyError, match=r"^'unstaged'$"):
            lookup("unstaged")


def test_identity_rows_preserve_first_duplicate_and_reject_malformed_layouts(
    tmp_path: Path,
) -> None:
    write_identity(tmp_path)
    rows = identity_rows()
    rows[1] = rows[0].copy()
    rows[1].update(key="second", standard_key="second")
    parquet.write_table(
        Table.from_pylist([dict(row) for row in rows]),
        tmp_path / "molecules.parquet",
    )
    table = IdentityTable.Config(working_dir=tmp_path).make()
    assert table.key(rows[0]["smiles"]) == rows[0]["key"]
    np.save(tmp_path / "fingerprints.npy", np.zeros((8, 2069), np.int8))
    with pytest.raises(TypeError, match=r"expected uint8, got int8\.$"):
        IdentityTable.Config(working_dir=tmp_path).make()
    for malformed in (
        np.zeros(2069, np.uint8),
        np.zeros((7, 2069), np.uint8),
        np.zeros((8, 2068), np.uint8),
    ):
        np.save(tmp_path / "fingerprints.npy", malformed)
        with pytest.raises(
            ValueError,
            match=r"^Identity fingerprints require complete U8\[rows,2069\] layout\.$",
        ):
            IdentityTable.Config(working_dir=tmp_path).make()
    write_identity(tmp_path)
    for header in (
        {"format_version": 2, "rdkit": IDENTITY_RDKIT, "raw_bits": 16_551},
        {"format_version": 1, "rdkit": SIMULATOR_RDKIT, "raw_bits": 16_551},
        {"format_version": 1, "rdkit": IDENTITY_RDKIT, "raw_bits": 16_552},
    ):
        (tmp_path / "manifest.json").write_text(json.dumps(to_plain(header)))
        with pytest.raises(
            ValueError,
            match=r"^Identity data requires format 1, RDKit 2026\.03\.3 and 16551 bits\.$",
        ):
            IdentityTable.Config(working_dir=tmp_path).make()
    (tmp_path / "manifest.json").write_text(
        f'{{"format_version":true,"rdkit":"{IDENTITY_RDKIT}","raw_bits":16551}}',
    )
    with pytest.raises(ReadError):
        IdentityTable.Config(working_dir=tmp_path).make()


def test_identity_types_partial_failures_and_restricted_numpy_loading(
    tmp_path: Path,
) -> None:
    for field in ("metric", "standard_metric"):
        write_identity(tmp_path)
        rows = [dict(row) for row in identity_rows()]
        rows[0][field] = json.dumps(to_plain((123, ())))
        parquet.write_table(Table.from_pylist(rows), tmp_path / "molecules.parquet")
        with pytest.raises(ReadError):
            IdentityTable.Config(working_dir=tmp_path).make()
    write_identity(tmp_path)
    rows = [dict(row) for row in identity_rows()]
    rows[0].update(properties_valid=False, fingerprint_valid=False)
    parquet.write_table(Table.from_pylist(rows), tmp_path / "molecules.parquet")
    table = IdentityTable.Config(working_dir=tmp_path).make()
    standardized = table.for_smiles(identity_rows()[0]["smiles"])["standardized"]
    assert standardized is not None
    assert table.properties(standardized) is None
    assert table.fingerprint(standardized) is None
    assert table.identity(standardized) is not None
    (tmp_path / "fingerprints.npy").write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(ValueError, match=r"pickled.*data"):
        IdentityTable.Config(working_dir=tmp_path).make()


def test_complete_provided_pool_and_library_state(
    tmp_path: Path,
) -> None:
    counts = np.array([3, 7, 4, 5, 7, 1, 6, 4, 5], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(tmp_path / "spec_off.npy", offsets)
    np.save(
        tmp_path / "spec_mz.npy",
        np.arange(offsets.item(-1), dtype=np.float32) / 8 + 10,
    )
    np.save(
        tmp_path / "spec_it.npy",
        (np.arange(offsets.item(-1), dtype=np.float32) % 5 + 1) / 5,
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 1, 2, 3, 4],
                "mass": [100.0, 100.001, 99.999, 150.0, 300.0],
                "key": ["a", "b", "c", "d", "e"],
                "smiles": ["C", "CC", "CCC", "O", "N"],
            },
        ),
        where=tmp_path / "train_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 0, 1, 2, 3, 4, -1, 1, 2],
                "lib": np.arange(9, dtype=np.int8) % 3,
                "mode": [1, 1, -1, 1, -1, 1, -1, 1, 1],
                "instr": np.arange(9, dtype=np.int8) % 6,
                "adduct_ix": np.arange(9, dtype=np.int16),
                "precursor_mz": np.arange(9, dtype=np.float64) / 8 + 101,
                "n_clean": counts,
            },
        ),
        where=tmp_path / "spec_meta.parquet",
    )
    formulas = [
        "C2H6O",
        "H2O",
        "C+H-2",
        "+",
        "C0H0",
        "CH4Na",
        "C(OH)2",
        "X2",
        "Cl2BrFI",
    ]
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": [
                    99.999,
                    100.0,
                    100.001,
                    130.0,
                    150.0,
                    170.0,
                    200.0,
                    300.0,
                    400.0,
                ],
                "key": ["a", "b", "a", "d", "e", "f", "g", "h", "i"],
                "smiles": ["C", "CC", "CCC", "O", "N", "CO", "CN", "Cl", "Br"],
                "formula": formulas,
                "n_heavy": np.arange(9, dtype=np.int32) + 1,
                "src": np.arange(9, dtype=np.int8) % 3,
                "train_sid": [0, 1, 2, -1, -1, 3, 4, -1, -1],
            },
        ),
        where=tmp_path / "pool_meta.parquet",
    )
    np.save(tmp_path / "fp_bits.npy", np.arange(9, dtype=np.int64) * 3)
    np.save(tmp_path / "pool_fp.npy", np.arange(18, dtype=np.uint8).reshape(9, 2) * 13)
    np.save(tmp_path / "pool_frag_off.npy", np.arange(10, dtype=np.int64) * 2)
    np.save(tmp_path / "pool_frag_mass.npy", np.arange(18, dtype=np.float32) / 9 + 1)
    library = SpectralLibrary.load(tmp_path)
    pool = CandidatePool.load(tmp_path)
    assert library.keys == ("a", "b", "c", "d", "e")
    assert library.smiles == ("C", "CC", "CCC", "O", "N")
    assert pool.keys == ("a", "b", "a", "d", "e", "f", "g", "h", "i")
    assert pool.smiles == ("C", "CC", "CCC", "O", "N", "CO", "CN", "Cl", "Br")
    assert pool.formulas == tuple(formulas)
    assert library.offsets.tolist() == [0, *np.cumsum(counts).tolist()]
    # The unknown SID -1 indexes the last structure, as NumPy does.
    assert library.mass.tolist() == [
        100.0,
        100.0,
        100.001,
        99.999,
        150.0,
        300.0,
        300.0,
        100.001,
        99.999,
    ]
    assert library.order.tolist() == [3, 8, 0, 1, 2, 7, 4, 5, 6]
    assert np.array_equal(library.mass_sorted, np.sort(library.mass))
    assert library.mode.dtype == np.int8
    assert library.adduct.dtype == np.int16
    excluded = np.isin(np.arange(9), [1, 3])
    near = library.window(100.0, ppm=20.0)
    assert near.tolist() == [3, 8, 0, 1, 2, 7]
    assert library.window(100.0, ppm=20.0, mode=1).tolist() == [3, 8, 0, 1, 7]
    assert library.window(100.0, ppm=20.0, mode=-1).tolist() == [2]
    assert library.window(100.0, ppm=20.0, excluded=excluded).tolist() == [8, 0, 2, 7]
    assert library.window(300.0, ppm=10.0, excluded=excluded).tolist() == [5, 6]
    assert not len(library.window(40.0, ppm=10.0, mode=1))
    assert pool.window(100.0, ppm=20.0).tolist() == [0, 1, 2]
    assert pool.window(150.0, ppm=10.0).tolist() == [4]
    # One most-populated spectrum per SID and polarity; ties keep encounter order.
    default = library.representatives()
    assert default.indices.tolist() == [3, 1, 2, 7, 4, 6, 5]
    assert default.sid.tolist() == [2, 0, 1, 1, 3, -1, 4]
    assert default.mode.tolist() == [1, 1, -1, 1, -1, -1, 1]
    assert default.mass.tolist() == library.mass[default.indices].tolist()
    restricted = library.representatives(excluded=excluded, key="excluded")
    assert restricted.indices.tolist() == [8, 0, 2, 7, 4, 6, 5]
    everything = np.ones(9, dtype=np.bool_)
    assert not len(library.representatives(excluded=everything, key="empty").indices)
    assert library.representatives(excluded=everything) is default
    # Packed rows 8, 1 and 2 hold 208/221, 26/39 and 52/65; nine bits are kept.
    assert pool.fingerprints(np.array([8, 1, 2, 1], dtype=np.int64)).tolist() == [
        [1, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0],
    ]
    assert list(pool.key_to_pid.values()) == [2, 1, 3, 4, 5, 6, 7, 8]
    fragments = pool.fragments(3)
    assert fragments.dtype == np.float64
    assert fragments.tolist() == pool.fragment_mass[6:8].tolist()
    vectors = [pool.formula_vector(pid) for pid in range(9)]
    assert [None if vector is None else vector.tolist() for vector in vectors] == [
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0] * 10,
        [0] * 10,
        None,
        None,
        None,
        [0, 0, 0, 0, 0, 0, 1, 2, 1, 1],
    ]
    assert library.representatives().indices is library.representatives().indices
    assert pool.formula_vector(0) is pool.formula_vector(0)
    assert pool.fingerprints(np.empty(0, dtype=np.int64)).shape == (0, 9)


def test_runtime_reads_all_staged_identities_without_rdkit(tmp_path: Path) -> None:
    # Formula, mass, heavy atoms, key and heavy composition of each valid structure.
    valid: dict[str, tuple[str, float, int, str, tuple[tuple[str, int], ...]]] = {
        "CCO": ("C2H6O", 46.041864812, 3, "LFQSCWFLJHTTHZ", (("C", 2), ("O", 1))),
        "O=c1cccc[nH]1": (
            "C5H5NO",
            95.03711378,
            7,
            "UBQKCCHYAOITMY",
            (("C", 5), ("N", 1), ("O", 1)),
        ),
        "C[C@H](O)F": (
            "C2H5FO",
            64.032443,
            4,
            "OGBOTYGRYZDLMG",
            (("C", 2), ("F", 1), ("O", 1)),
        ),
        "[13CH3]CO": ("C2H6O", 47.045219652, 3, "LFQSCWFLJHTTHZ", (("C", 2), ("O", 1))),
        "CCO.[Na+]": (
            "C2H6NaO+",
            69.03108551209,
            4,
            "CDPFDOHSBWKTDU",
            (("C", 2), ("Na", 1), ("O", 1)),
        ),
        "[NH4+]": ("H4N+", 18.03382554809, 1, "QGZKDVFQNNGYKY", (("N", 1),)),
    }
    structures = (*valid, "not-smiles")
    prepared = [valid.get(smiles) for smiles in structures]
    identities = [None if row is None else (row[3], row[4]) for row in prepared]
    parquet.write_table(
        Table.from_pydict(
            {
                "input_smiles": list(structures),
                "smiles": [smiles if smiles in valid else "" for smiles in structures],
                "valid": [row is not None for row in prepared],
                "formula": [row[0] if row else "" for row in prepared],
                "mass": [row[1] if row else float("nan") for row in prepared],
                "heavy_atoms": [row[2] if row else 0 for row in prepared],
                "key": [row[3] if row else "" for row in prepared],
                "raw_key": [row[3] if row else "" for row in prepared],
                "metric_key": [row[3] if row else "" for row in prepared],
                "composition": [
                    json.dumps(to_plain(row[4] if row else ())) for row in prepared
                ],
            },
        ),
        tmp_path / "structures.parquet",
    )
    np.save(tmp_path / "raw_fp.npy", np.zeros((7, (16_551 + 7) // 8), np.uint8))
    _ = (tmp_path / "fingerprint_bits.txt").write_text("16551\n")
    _ = (tmp_path / "manifest.json").write_text(f'{{"rdkit":"{IDENTITY_RDKIT}"}}\n')
    original_import = importlib.import_module

    def no_rdkit(name: str, package: str | None = None) -> ModuleType:
        if name.startswith("rdkit"):
            raise AssertionError("Runtime attempted an RDKit import.")
        return original_import(name, package)

    config = OfficialIdentity.Config()
    config.working_dir = tmp_path
    with patch("importlib.import_module", side_effect=no_rdkit):
        lookup = config.make()
        assert [lookup(smiles) for smiles in structures] == identities
        with pytest.raises(KeyError, match="unprepared"):
            _ = lookup("unprepared")


def test_wrong_prepared_chemistry_version_is_rejected_before_loading(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "manifest.json").write_text(f'{{"rdkit":"{SIMULATOR_RDKIT}"}}\n')
    config = OfficialIdentity.Config()
    config.working_dir = tmp_path
    with pytest.raises(
        ValueError,
        match=rf"^Official identity data requires RDKit {IDENTITY_RDKIT}; got {SIMULATOR_RDKIT}\.$",
    ):
        _ = config.make()


@pytest.mark.parametrize("manifest", ["null", "[]", '{"rdkit":7}'])
def test_identity_provenance_requires_the_declared_schema(
    tmp_path: Path,
    manifest: str,
) -> None:
    _ = (tmp_path / "manifest.json").write_text(manifest)
    config = OfficialIdentity.Config()
    config.working_dir = tmp_path
    with pytest.raises(ReadError):
        _ = config.make()


def search_store() -> PeakStore:
    """Hold an exact copy, a disjoint pair, a silent pair, no peaks and a +10 copy."""
    masses: tuple[list[int], ...] = (
        [10, 20, 30],
        [110, 120],
        [10, 20],
        [],
        [20, 30, 40],
    )
    intensity: tuple[list[int], ...] = ([1, 1, 1], [1, 1], [0, 0], [], [1, 1, 1])
    return PeakStore(
        offsets=np.concatenate([[0], np.cumsum([len(row) for row in masses])]).astype(
            np.int64,
        ),
        mz=np.concatenate([np.array(row, np.float32) for row in masses]),
        intensity=np.concatenate([np.array(row, np.float32) for row in intensity]),
    )


def test_main_search_scores_exact_disjoint_silent_and_shifted_rows() -> None:
    store = search_store()
    search = MainSearch.Config().make()
    query = (np.array([10.0, 20.0, 30.0]), np.ones(3))
    candidates = np.arange(5, dtype=np.int64)
    direct = search.score(query, store=store, candidates=candidates)
    assert direct.dtype == np.float32
    assert direct.item(0) == 1.0
    assert 0 <= direct.item(1) < 1e-12
    assert direct.tolist()[2:4] == [0.0, 0.0]
    # The +10 copy shares two of three peaks: mixture 1/6, 1/3, 1/3, 1/6 scores 2/3.
    assert direct.item(4) == np.float32(2 / 3)
    hybrid = search.score(
        query,
        store=store,
        candidates=candidates,
        shifts=np.array([5e-5, -100.0, 0.0, 0.0, 0.0]),
    )
    assert hybrid.item(0) == direct.item(0)
    assert np.array_equal(hybrid[2:], direct[2:])
    # Shifted by -100, the pair explains two query peaks: mixture 5/12, 5/12, 1/6.
    mixture = 5 / 6 * math.log(12 / 5) + math.log(6) / 6
    assert hybrid.item(1) == np.float32((math.log(6) - 2 * mixture) / math.log(4) + 1)
    pair = (np.array([10.0, 20.0]), np.ones(2))
    cosine = search.score(pair, store=store, candidates=candidates, cosine=True)
    # Cosine over norms sqrt(1/2 * 1/3): dot 1/3 for the copy, 1/6 for the +10 copy.
    assert cosine.tolist() == [
        float(np.float32((2 / 3) ** 0.5)),
        0.0,
        0.0,
        0.0,
        float(np.float32(6**-0.5)),
    ]
    silent = (np.zeros(0), np.zeros(0))
    assert not search.score(silent, store=store, candidates=candidates).any()
    assert not search.score(
        silent,
        store=store,
        candidates=candidates,
        cosine=True,
    ).any()
    assert not search.score(
        silent,
        store=store,
        candidates=candidates,
        shifts=np.full(5, -10.0),
    ).any()


def test_second_search_keeps_encounter_order_and_takes_the_better_shift() -> None:
    store = search_store()
    search = SecondSearch.Config().make()
    capped = SecondSearch.Config(max_peaks=2).make()
    mz, weights = capped.prepare(
        (np.array([30, 10, 20, 40], np.float32), np.array([1, 4, 2, 3], np.float32)),
    )
    assert mz.tolist() == [10.0, 40.0]
    assert weights.dtype == np.float32
    assert weights.item(0) > weights.item(1) > 0
    query = search.prepare(store.row(0))
    candidates = np.arange(5, dtype=np.int64)
    direct = search.score(query, store=store, candidates=candidates)
    assert direct.item(0) == direct.max()
    assert direct.tolist()[2:4] == [0.0, 0.0]
    shifts = np.array([5e-4, -100.0, 0.0, 0.0, -10.0], np.float32)
    shifted = search.score(query, store=store, candidates=candidates, shifts=shifts)
    assert shifted.item(0) == direct.item(0)
    assert bool((shifted >= direct).all())
    assert shifted.item(4) > direct.item(4)
    cleaned = search.clean_store(store, indices=candidates)
    assert cleaned.offsets.tolist() == [0, 3, 5, 5, 5, 8]
    assert np.array_equal(
        search.score_prepared(query, store=cleaned, start=1, stop=5, shifts=shifts[1:]),
        search.score(query, store=store, candidates=candidates[1:], shifts=shifts[1:]),
    )
    empty = search.clean_store(store, indices=np.empty(0, dtype=np.int64))
    assert empty.offsets.tolist() == [0]
    assert empty.mz.shape == empty.intensity.shape == (0,)


def test_merging_keeps_the_stronger_near_duplicate_then_caps_by_intensity() -> None:
    later = [[float(np.float32(10.004)), 20.0], [0.5, 1.0]]
    assert _merged([([10, 10.004, 20], [1, 2, 4])]) == later
    assert _merged([([10, 10.004], [2, 1])]) == [[10.0], [1.0]]
    # Each spectrum normalizes by its own base peak before the two merge.
    assert _merged([([10], [2]), ([10.003, 30], [4, 8])]) == [
        [10.0, 30.0],
        [1.0, 1.0],
    ]
    assert _merged([([10, 20, 30], [1, 3, 2])], maximum=1) == [[20.0], [1.0]]
    assert _merged([([3, 4], [0, 0])]) == _merged([]) == [[], []]


def _merged(
    spectra: list[tuple[list[float], list[float]]],
    maximum: int | None = None,
) -> list[list[float]]:
    """Merge plain spectra and return float32 masses and intensities as lists."""
    merged = merge_main(
        [
            (np.array(mz, np.float64), np.array(intensity, np.float64))
            for mz, intensity in spectra
        ],
        tolerance=0.005,
        max_peaks=maximum,
    )
    assert merged[0].dtype == merged[1].dtype == np.float32
    return [merged[0].tolist(), merged[1].tolist()]


def test_complete_query_cleaning_keeps_stable_peak_orders() -> None:
    # Each case: masses, intensities, precursor, floor, peak cap and mass margin.
    cases: tuple[tuple[list[float], list[float], float, float, int, float], ...] = (
        ([], [], 100.0, 0.001, 512, 2.0),
        ([10.0, 20.0], [0.0, -1.0], 100.0, 0.001, 512, 2.0),
        (
            [-2.0, 0.0, 10.0, 20.0, 103.0],
            [1000.0, 900.0, 2.0, 1.0, 2000.0],
            100.0,
            0.001,
            512,
            2.0,
        ),
        (
            [102.0, 102.000001, 100.0, 1.0],
            [7.0, 10.0, 1.0, 0.0069999],
            100.0,
            0.001,
            512,
            2.0,
        ),
        ([40.0 - index for index in range(40)], [1.0] * 40, 100.0, 0.001, 7, 2.0),
        (
            [20.0, 10.0, 20.0, 20.0, 10.0, 30.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            100.0,
            0.0,
            4,
            2.0,
        ),
        ([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], float("nan"), 0.001, 512, 2.0),
        (
            [11.0, 15.0, 10.0, 14.0, 12.0, 13.0],
            [1.0, 4.0, 2.0, 3.0, 1.0, 2.0],
            10.0,
            0.5,
            2,
            4.0,
        ),
    )
    expected: tuple[tuple[list[float], list[float]], ...] = (
        ([], []),
        ([], []),
        ([10.0, 20.0], [0.002, 0.001]),
        ([100.0, 102.0], [1 / 7, 1.0]),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0] * 7),
        ([10.0, 20.0, 20.0, 30.0], [5 / 6, 3 / 6, 4 / 6, 1.0]),
        ([], []),
        ([13.0, 14.0], [2 / 3, 1.0]),
    )
    for case, peaks in zip(cases, expected, strict=True):
        mass, intensity, precursor, floor, maximum, over = case
        actual = clean_main(
            (np.array(mass, np.float64), np.array(intensity, np.float64)),
            precursor=precursor,
            floor=floor,
            max_peaks=maximum,
            over=over,
        )
        assert actual[0].dtype == actual[1].dtype == np.float32
        np.testing.assert_array_equal(actual[0], np.array(peaks[0], np.float32))
        np.testing.assert_array_equal(actual[1], np.array(peaks[1], np.float32))


def test_main_preparation_normalizes_reweights_low_entropy_and_caps_peaks() -> None:
    linear = MainSearch.Config(entropy_weight=False)
    assert _prepared([3.0, 1.0], linear) == [[10, 11], [0.75, 0.25]]
    assert _prepared(
        [4.0, 1.0],
        MainSearch.Config(entropy_weight=False, power=0.5),
    ) == [[10, 11], [2 / 3, 1 / 3]]
    # Twenty-one equal peaks have entropy log 21 > 3, so no reweighting applies.
    assert _prepared([1.0] * 21, MainSearch.Config()) == [
        list(range(10, 31)),
        [1 / 21] * 21,
    ]
    # A peak exactly at the relative floor stays.
    assert _prepared([1.0, 0.5, 0.2, 0.1], MainSearch.Config(floor=0.5))[0] == [10, 11]
    assert _prepared([0.2, 1.0, 0.5], MainSearch.Config(max_peaks=2))[0] == [11, 12]
    assert _prepared([0.0, 0.0], MainSearch.Config()) == [[], []]
    assert _prepared([], MainSearch.Config()) == [[], []]


def _prepared(intensity: list[float], config: MainSearch.Config) -> list[list[float]]:
    """Prepare peaks at masses 10, 11, ... and return masses and weights as lists."""
    values = np.array(intensity, np.float64)
    mass = np.arange(len(values), dtype=np.float64) + 10
    mz, weights = config.make().prepare((mass, values))
    return [mz.tolist(), weights.tolist()]


def test_low_entropy_reweighting_gives_both_engines_the_closed_form_shares() -> None:
    # Weights 3/4 and 1/4 have entropy log 4 - (3/4) log 3 < 3; raised to
    # 1/4 + entropy/4 and renormalized, the weak share is 1 / (1 + 3**exponent).
    exponent = 0.25 + (math.log(4) - 0.75 * math.log(3)) / 4
    weak = 1 / (1 + 3**exponent)
    mass = np.array([10.0, 11.0])
    main = MainSearch.Config().make().prepare((mass, np.array([3.0, 1.0])))[1]
    # Both the weak weight and this closed form sit about 1 ulp from the exact share.
    assert main.tolist() == pytest.approx([1 - weak, weak], rel=1e-15, abs=0)
    second = (
        SecondSearch.Config()
        .make()
        .prepare((mass.astype(np.float32), np.array([3.0, 1.0], np.float32)))[1]
    )
    assert second.tolist() == np.array([1 - weak, weak], np.float32).tolist()


def test_unmatched_query_peaks_enter_both_mixtures_at_their_own_weight() -> None:
    # Query weights 1/2 and 1/2 at 10 and 20 Da; one candidate peak each.
    query = (np.array([10.0, 20.0]), np.array([1.0, 1.0]))
    shifted = PeakStore(
        offsets=np.array([0, 1], np.int64),
        mz=np.array([15.0], np.float32),
        intensity=np.array([1.0], np.float32),
    )
    rows = np.array([0], np.int64)
    # One peak explains 20 Da and 10 Da stays unmatched, so the mixture is
    # [3/4, 1/4] and the score (log 2 - 2 H(3/4, 1/4)) / log 4 + 1.
    expected = [float(np.float32(0.75 * math.log2(3) - 0.5))]
    main = MainSearch.Config().make()
    hybrid = main.score(query, store=shifted, candidates=rows, shifts=np.array([5.0]))
    assert hybrid.tolist() == expected
    direct = replace(shifted, mz=np.array([10.0], np.float32))
    # Shifted by 10, the directly matched peak keeps 10 Da and leaves 20 Da unmatched.
    assert (
        main.score(query, store=direct, candidates=rows, shifts=np.array([10.0]))
    ).tolist() == expected
    search = SecondSearch.Config().make()
    prepared = search.prepare(
        (query[0].astype(np.float32), query[1].astype(np.float32)),
    )
    # The second engine sums the matched pair, then divides the mixture by 2.
    assert search.score(prepared, store=direct, candidates=rows).tolist() == expected
    assert (
        search.score(
            prepared,
            store=shifted,
            candidates=rows,
            shifts=np.array([5.0], np.float32),
        ).tolist()
        == expected
    )


def test_prepared_spectra_compare_without_narrowing_to_float32() -> None:
    search = MainSearch.Config().make()
    peaks = (np.array([100.0, 200.0, 300.0]), np.array([1.0, 0.5, 0.25]))
    other = (np.array([150.0, 250.0]), np.array([1.0, 1.0]))
    query, same, different = (
        search.prepare(spectrum) for spectrum in (peaks, peaks, other)
    )
    assert search.similarity(query, same) == pytest.approx(1.0, abs=1e-12)
    # The exact value is 2.8e-17; three rounded entropies cancel to one ulp of 1.
    assert search.similarity(query, different) == math.ulp(1.0)


def test_prepared_similarity_clips_a_rounded_excess_to_one() -> None:
    search = MainSearch.Config().make()
    masses = np.array([100.0, 200.0])
    query = search.prepare((masses, np.array([6.0, 2.0])))
    # Unclipped, the rounded entropies of this near-identical pair give 1 + 2**-52.
    near = search.prepare((masses, np.array([6.0, 2.0 + 2.0**-31])))
    assert search.similarity(query, near) == 1.0


_SCRIPTS_DIR: Final = _CWD / "scripts"


# derivation_chemistry_test.py.
class _Extra:
    """Hold the bit information a Morgan generator reports."""

    def AllocateBitInfoMap(self) -> None:  # noqa: N802 -- Mirrors the library.
        pass

    def GetBitInfoMap(self) -> dict[int, list[tuple[int, int]]]:  # noqa: N802 -- Mirrors the library.
        return {
            10: [(0, 0), (1, 0)],
            11: [(0, 1)],
            12: [(0, 3)],
            13: [(1, 2)],
        }


class _Morgan:
    def GetSparseCountFingerprint(  # noqa: N802 -- Mirrors the library.
        self,
        molecule: object,
        *,
        additionalOutput: _Extra,  # noqa: N803 -- Mirrors the library.
    ) -> None:
        del molecule, additionalOutput


class _DerivationChemistryAtom:
    def __init__(self, **properties: int) -> None:
        self.properties = properties

    def HasProp(self, name: str) -> bool:  # noqa: N802 -- Mirrors the library.
        return name in self.properties

    def GetIntProp(self, name: str) -> int:  # noqa: N802 -- Mirrors the library.
        return self.properties[name]


class _Product:
    def __init__(self, name: str, *, atoms: int = 3, site: int | None = None) -> None:
        self.name = name
        self.count = atoms
        self.mapped = (
            [
                _DerivationChemistryAtom(),
                _DerivationChemistryAtom(old_mapno=1, react_atom_idx=site),
            ]
            if site is not None
            else [_DerivationChemistryAtom(old_mapno=2, react_atom_idx=9)]
        )

    def GetNumAtoms(self) -> int:  # noqa: N802 -- Mirrors the library.
        return self.count

    def GetAtoms(self) -> list[_DerivationChemistryAtom]:  # noqa: N802 -- Mirrors the library.
        return self.mapped


class _Parent:
    def GetNumHeavyAtoms(self) -> int:  # noqa: N802 -- Mirrors the library.
        return 2

    def GetNumAtoms(self) -> int:  # noqa: N802 -- Mirrors the library.
        return 3


class _Reaction:
    def __init__(self, groups: list[tuple[_Product, ...]] | None) -> None:
        self.groups = groups

    def RunReactants(self, reactants: tuple[object, ...]) -> list[tuple[_Product, ...]]:  # noqa: N802 -- Mirrors the library.
        del reactants
        if self.groups is None:
            raise RuntimeError("reaction failed")
        return self.groups


class _Recorder:
    """Log every property edit as a labelled pair."""

    def __init__(self, label: str, edits: list[tuple[str, object]]) -> None:
        self.label = label
        self.edits = edits

    def _log(self, name: str, value: object) -> None:
        self.edits.append((self.label + name, value))

    def SetBondType(self, value: object) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("type", value)

    def SetIsAromatic(self, value: bool) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("aromatic", value)

    def SetFormalCharge(self, value: int) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("charge", value)

    def SetNumExplicitHs(self, value: int) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("hs", value)

    def SetNoImplicit(self, value: bool) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("implicit", value)

    def SetChiralTag(self, value: object) -> None:  # noqa: N802 -- Mirrors the library.
        self._log("chiral", value)


class _Chem:
    BondType = types.SimpleNamespace(SINGLE="single")
    ChiralType = types.SimpleNamespace(CHI_UNSPECIFIED="unspecified")

    def __init__(self) -> None:
        self.edited: list[tuple[str, object]] = []

    def SanitizeMol(self, product: _Product) -> None:  # noqa: N802 -- Mirrors the library.
        if product.name == "unsanitary":
            raise ValueError(product.name)

    def MolToSmiles(  # noqa: N802 -- Mirrors the library.
        self,
        molecule: object,
        *,
        isomericSmiles: bool = True,  # noqa: N803 -- Mirrors the library.
    ) -> str:
        if isinstance(molecule, _Product):
            return molecule.name
        return "isomeric" if isomericSmiles else "skeleton"

    def MolFromSmiles(self, smiles: str) -> object | None:  # noqa: N802 -- Mirrors the library.
        if smiles == "invalid":
            return None
        if smiles == "broken":
            raise RuntimeError("parse failed")
        return types.SimpleNamespace(smiles=smiles)

    def MolToInchiKey(self, molecule: object) -> str:  # noqa: N802 -- Mirrors the library.
        del molecule
        return "ABCDEFGHIJKLMN-OPQRSTUVWX-N"

    def RWMol(self, molecule: object) -> object:  # noqa: N802 -- Mirrors the library.
        del molecule
        edits = self.edited

        return types.SimpleNamespace(
            GetBonds=lambda: [_Recorder("bond ", edits)],
            GetAtoms=lambda: [_Recorder("atom ", edits)],
        )


class _Tautomers:
    def Canonicalize(self, molecule: object) -> object:  # noqa: N802 -- Mirrors the library.
        if getattr(molecule, "smiles", "") == "tautomer-error":
            raise RuntimeError("tautomer failed")
        return molecule


class _Standard:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.chem = _Chem()
        self.tautomers = _Tautomers()


def _morgan(**options: object) -> _Morgan:
    del options
    return _Morgan()


@pytest.fixture
def chemistry(monkeypatch: pytest.MonkeyPatch) -> DerivationChemistry:
    monkeypatch.setattr(derivation_chemistry, "Chemistry", _Standard)
    monkeypatch.setattr(derivation_chemistry, "AdditionalOutput", _Extra)
    monkeypatch.setattr(derivation_chemistry, "GetMorganGenerator", _morgan)
    return DerivationChemistry(IDENTITY_RDKIT)


def test_a_wrong_scientific_release_is_rejected() -> None:
    with pytest.raises(ValueError, match=rf"requires RDKit {IDENTITY_RDKIT}"):
        _ = DerivationChemistry("0.0.0")


def test_the_rule_table_adds_the_declared_dehexose_to_the_v6_vocabulary(
    chemistry: DerivationChemistry,
) -> None:
    assert "dehexose2" in chemistry.rules
    assert chemistry.rules["dehexose2"].name == "dehexose2"
    assert len(chemistry.rules) > 1


def test_environments_back_off_to_the_nearest_smaller_radius_and_to_missing_sites(
    chemistry: DerivationChemistry,
) -> None:
    molecule = types.SimpleNamespace(GetNumAtoms=lambda: 3)
    assert chemistry.environments(cast("Chem.Mol", molecule)) == {
        0: (11, 11, 12),
        1: (10, 13, 13),
        2: (-1, -1, -1),
    }


def _rule() -> Reaction:
    return Reaction(name="rule", smarts="[C:1]>>[C:1]", delta=(("C", 1), ("H", 2)))


def test_products_filter_atoms_deduplicate_reject_and_report_the_mapped_site(
    chemistry: DerivationChemistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _Product("first", site=1)
    groups: list[tuple[_Product, ...]] = [
        (kept, _Product("ignored second member")),
        (_Product("wrong size", atoms=9),),
        (_Product("first"),),
        (_Product("unsanitary"),),
        (_Product("second"),),
    ]
    compiled: list[str] = []

    def compile_reaction(smarts: str) -> _Reaction:
        compiled.append(smarts)
        return _Reaction(groups)

    monkeypatch.setattr(derivation_chemistry, "ReactionFromSmarts", compile_reaction)
    parent = cast("Chem.Mol", _Parent())
    result = chemistry.products(parent, rule=_rule(), limit=10)
    assert [(cast("_Product", product).name, site) for product, site in result] == [
        ("first", 1),
        ("second", -1),
    ]
    assert result[0][0] is kept
    result = chemistry.products(parent, rule=_rule(), limit=0)
    assert [cast("_Product", product).name for product, _ in result] == ["first"]
    assert compiled == ["[C:1]>>[C:1]"]


def test_a_failing_reaction_yields_no_products(
    chemistry: DerivationChemistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(smarts: str) -> _Reaction:
        del smarts
        return _Reaction(None)

    monkeypatch.setattr(derivation_chemistry, "ReactionFromSmarts", reject)
    parent = cast("Chem.Mol", _Parent())
    assert chemistry.products(parent, rule=_rule(), limit=3) == []


def test_identity_keeps_the_key_prefix_and_resets_every_graph_property(
    chemistry: DerivationChemistry,
) -> None:
    assert chemistry.identity("CCO") == ("ABCDEFGHIJKLMN", "skeleton")
    chem = cast("_Chem", chemistry.chem)
    assert chem.edited == [
        ("bond type", "single"),
        ("bond aromatic", False),
        ("atom aromatic", False),
        ("atom charge", 0),
        ("atom hs", 0),
        ("atom implicit", True),
        ("atom chiral", "unspecified"),
    ]


@pytest.mark.parametrize("smiles", ["invalid", "broken", "tautomer-error"])
def test_unparsable_or_failing_structures_have_explicit_missing_identities(
    chemistry: DerivationChemistry,
    smiles: str,
) -> None:
    expected_skeleton = "skeleton" if smiles == "tautomer-error" else None
    assert chemistry.identity(smiles) == (None, expected_skeleton)


# derivatives_test.py.
@dataclass(frozen=True, slots=True, kw_only=True)
class _DerivativesStructure:
    name: str


class _DerivativesPrimitives(ModuleType):
    def __init__(self) -> None:
        super().__init__("rdkit.Chem")
        self.compiled: list[str] = []
        self.applied: list[str] = []

    def MolFromSmiles(self, smiles: str) -> _DerivativesStructure | None:  # noqa: N802 -- Scientific boundary.
        return None if smiles == "invalid" else _DerivativesStructure(name=smiles)

    def MolToSmiles(self, molecule: Chem.Mol) -> str:  # noqa: N802 -- Scientific boundary.
        return cast("_DerivativesStructure", molecule).name.removeprefix("duplicate:")

    def SanitizeMol(self, molecule: _DerivativesStructure) -> None:  # noqa: N802 -- Scientific boundary.
        if molecule.name == "bad":
            raise ValueError("Invalid reaction product.")

    def ReactionFromSmarts(self, smarts: str) -> _DerivativesPrimitives:  # noqa: N802 -- Scientific boundary.
        self.compiled.append(smarts)
        return self

    def RunReactants(  # noqa: N802 -- Scientific boundary.
        self,
        molecules: tuple[_DerivativesStructure],
    ) -> tuple[tuple[_DerivativesStructure, ...], ...]:
        molecule = molecules[0]
        self.applied.append(molecule.name)
        if molecule.name == "error":
            raise RuntimeError("Reaction rejected input.")
        if molecule.name == "empty":
            return ()
        return tuple(
            (
                _DerivativesStructure(name=name),
                _DerivativesStructure(name="unused coproduct"),
            )
            for name in (
                "bad",
                molecule.name,
                "duplicate:" + molecule.name,
                molecule.name + "A",
                molecule.name + "B",
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _DerivativesPreparation:
    version: str = IDENTITY_RDKIT


def _derivatives_install(
    monkeypatch: pytest.MonkeyPatch,
    library: _DerivativesPrimitives,
) -> None:
    """Replace the lazily imported scientific names of the generator by one fake."""
    monkeypatch.setattr(derivatives, "Chem", library)
    monkeypatch.setattr(derivatives, "ReactionFromSmarts", library.ReactionFromSmarts)


def derive(
    generator: DerivativeGenerator,
    smiles: str,
    *,
    delta: float,
    tolerance: float = 0.003,
    max_reactions: int = 2,
    max_products: int = 60,
    allow_isomerization: bool = True,
    application_budget: int = 800,
    stage_parents: int = 24,
    first_products: int = 32,
    later_products: int = 16,
) -> list[Derivative]:
    """Derive with the released budgets unless a case overrides one."""
    return generator.derive(
        smiles,
        delta=delta,
        tolerance=tolerance,
        max_reactions=max_reactions,
        max_products=max_products,
        allow_isomerization=allow_isomerization,
        application_budget=application_budget,
        stage_parents=stage_parents,
        first_products=first_products,
        later_products=later_products,
    )


def apply(
    generator: DerivativeGenerator,
    item: Chem.Mol,
    *,
    rule: Reaction,
    max_products: int = 64,
) -> list[Chem.Mol]:
    """Apply one rule with the released cap unless a case overrides it."""
    return generator.apply(item, rule=rule, max_products=max_products)


def molecule(name: str) -> Chem.Mol:
    """Stand in for a parsed scientific molecule."""
    return cast("Chem.Mol", _DerivativesStructure(name=name))


def test_apply_sanitizes_deduplicates_caps_and_reuses_compiled_reactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    generator = DerivativeGenerator()
    rule = reaction_rules()[0]
    actual = apply(generator, molecule("parent"), rule=rule)
    assert [library.MolToSmiles(item) for item in actual] == [
        "parent",
        "parentA",
        "parentB",
    ]
    assert [
        library.MolToSmiles(item)
        for item in apply(generator, molecule("parent"), rule=rule, max_products=0)
    ] == ["parent"]
    assert apply(generator, molecule("error"), rule=rule) == []
    assert apply(generator, molecule("empty"), rule=rule) == []
    assert library.compiled == [rule.smarts]


def test_derive_retains_first_plan_and_exact_parent_string_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    generator = DerivativeGenerator()
    products = derive(generator, "parent", delta=15.9949146196, max_reactions=1)
    assert products == [
        Derivative(smiles="parentA", plan=(0,)),
        Derivative(smiles="parentB", plan=(0,)),
    ]
    assert library.applied == ["parent"] * 3
    assert derive(generator, "invalid", delta=0) == []
    assert derive(generator, "parent", delta=0, allow_isomerization=False) == []
    assert derive(generator, "parent", delta=1000) == []
    assert derive(generator, "parent", delta=0, max_products=0) == []
    assert derive(
        generator,
        "duplicate:parent",
        delta=15.9949146196,
        max_reactions=1,
        max_products=1,
    ) == [Derivative(smiles="parent", plan=(0,))]


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_application_budget_keeps_partial_depth_products_and_first_provenance(
    budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    generator = DerivativeGenerator()
    generator.rules = reaction_rules()[:3]
    products = derive(
        generator,
        "parent",
        delta=2 * 15.9949146196,
        application_budget=budget,
        stage_parents=1,
        first_products=2,
        later_products=2,
    )
    assert products == [Derivative(smiles="parentA", plan=(0, 0))]
    assert len(library.applied) == budget


def test_empty_stage_stops_the_current_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    generator = DerivativeGenerator()
    assert derive(generator, "empty", delta=2 * 15.9949146196) == []
    assert library.applied == ["empty"] * 7


def test_explicit_vocabulary_and_application_provenance_keep_each_actual_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    calls: list[tuple[str, str, int, int]] = []

    def observe(item: Chem.Mol, rule: Reaction, limit: int, count: int) -> None:
        calls.append((library.MolToSmiles(item), rule.name, limit, count))

    generator = DerivativeGenerator(
        rules=reaction_rules()[:1],
        on_application=observe,
    )
    assert derive(generator, "parent", delta=15.9949146196, max_reactions=1) == [
        Derivative(smiles="parentA", plan=(0,)),
        Derivative(smiles="parentB", plan=(0,)),
    ]
    assert calls == [("parent", "hydroxyl_arom", 32, 3)]
    for name in ("error", "empty"):
        assert apply(generator, molecule(name), rule=generator.rules[0]) == []
    assert calls[1:] == [
        ("error", "hydroxyl_arom", 64, 0),
        ("empty", "hydroxyl_arom", 64, 0),
    ]


# graph_properties_test.py.
def test_complete_isotope_formula_hydrogen_and_mass_properties() -> None:
    elements = (
        "C",
        "N",
        "P",
        "O",
        "S",
        "Si",
        "I",
        "H",
        "Cl",
        "F",
        "Br",
        "B",
        "Se",
        "Fe",
        "Co",
        "As",
        "Na",
        "K",
    )
    masses = np.zeros(len(elements))
    masses[[0, 1, 7]] = 12.0, 14.003074, 1.007825032
    table = _Table(13.00335484)
    actual = atom_properties(
        _atoms(
            _GraphPropertiesAtom("C", isotope=13, implicit=0, explicit=3),
            _GraphPropertiesAtom("N", isotope=0, implicit=3, explicit=0),
        ),
        formula="CH6N+",
        elements=elements,
        masses=masses,
        periodic_table=cast("Chem.PeriodicTable", table),
    )
    composition = np.zeros(len(elements))
    composition[[0, 1, 7]] = 1, 1, 6
    np.testing.assert_array_equal(actual.formula, composition)
    np.testing.assert_array_equal(actual.hydrogens, [3, 3])
    np.testing.assert_array_equal(actual.masses, [13.00335484, 14.003074])
    hydrogenated = np.array([3, 3]) * 1.007825032 + np.array([13.00335484, 14.003074])
    np.testing.assert_array_equal(actual.hydrogenated_masses, hydrogenated)
    assert actual.mass == hydrogenated[0] + hydrogenated[1]
    assert actual.total_hydrogens == 6
    np.testing.assert_array_equal(actual.elements, [0, 1])
    forms = np.zeros((2, 18), dtype=np.float64)
    forms[0, 0] = forms[1, 1] = 1
    forms[:, 7] = 3
    np.testing.assert_array_equal(actual.atom_formulas, forms)
    assert actual.atom_formulas.dtype == actual.formula.dtype == np.float64
    assert actual.hydrogens.dtype == actual.elements.dtype == np.int64
    assert table.calls == [("C", 13)]


def test_zero_atom_properties_retain_all_element_axes() -> None:
    actual = atom_properties(
        [],
        formula="",
        elements=("C", "H"),
        masses=np.array([12.0, 1.0]),
        periodic_table=cast("Chem.PeriodicTable", _Table(13.0)),
    )
    assert actual.formula.tolist() == [0.0, 0.0]
    assert actual.atom_formulas.shape == (0, 2)
    assert (
        actual.elements.shape == actual.hydrogens.shape == actual.masses.shape == (0,)
    )
    assert actual.hydrogenated_masses.shape == (0,)
    assert actual.hydrogens.dtype == actual.elements.dtype == np.int64
    assert actual.masses.dtype == actual.hydrogenated_masses.dtype == np.float64
    assert actual.mass == 0.0
    assert actual.total_hydrogens == 0


def test_formula_accumulates_repeated_elements_and_explicit_hydrogen_atom() -> None:
    actual = atom_properties(
        _atoms(
            _GraphPropertiesAtom("C", isotope=0, implicit=3, explicit=0),
            _GraphPropertiesAtom("C", isotope=0, implicit=3, explicit=0),
            _GraphPropertiesAtom("H", isotope=0, implicit=0, explicit=0),
        ),
        formula="CH3CH3H",
        elements=("C", "H"),
        masses=np.array([12.0, 1.0]),
        periodic_table=cast("Chem.PeriodicTable", _Table(13.0)),
    )
    np.testing.assert_array_equal(actual.formula, [2.0, 7.0])
    np.testing.assert_array_equal(actual.atom_formulas, [[1, 3], [1, 3], [0, 1]])
    np.testing.assert_array_equal(actual.elements, [0, 0, 1])
    np.testing.assert_array_equal(actual.hydrogens, [3, 3, 0])
    np.testing.assert_array_equal(actual.masses, [12.0, 12.0, 1.0])
    np.testing.assert_array_equal(actual.hydrogenated_masses, [15.0, 15.0, 1.0])
    assert actual.mass == 31.0
    assert actual.total_hydrogens == 6


def _atoms(*atoms: object) -> Sequence[Chem.Atom]:
    return cast("Sequence[Chem.Atom]", atoms)


class _GraphPropertiesAtom:
    def __init__(
        self,
        symbol: str,
        *,
        isotope: int,
        implicit: int,
        explicit: int,
    ) -> None:
        self.GetSymbol: Callable[[], str] = lambda: symbol
        self.GetIsotope: Callable[[], int] = lambda: isotope
        self.GetNumImplicitHs: Callable[[], int] = lambda: implicit
        self.GetNumExplicitHs: Callable[[], int] = lambda: explicit


class _Table:
    def __init__(self, mass: float) -> None:
        self.mass = mass
        self.calls: list[tuple[str, int]] = []
        self.GetMassForIsotope: Callable[[str, int], float] = self._isotope_mass

    def _isotope_mass(self, symbol: str, isotope: int) -> float:
        self.calls.append((symbol, isotope))
        return self.mass


# identity_data_test.py.
def test_identity_writer_keeps_every_decision_duplicate_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity_data, "IdentityChemistry", ObservedIdentity)
    strings = [row["smiles"] for row in identity_rows()]
    source, output = tmp_path / "input.parquet", tmp_path / "nested" / "prepared"
    parquet.write_table(Table.from_pydict({"smiles": strings + strings}), source)
    identity_data.prepare_identity(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    metadata = parquet.ParquetFile(output / "molecules.parquet")
    assert metadata.num_row_groups == 6
    assert metadata.read().column("smiles").to_pylist() == strings + strings
    assert (
        observe_identity(IdentityTable.Config(working_dir=output).make())
        == staged_identity()
    )
    header = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert header["format_version"] == 1
    assert header["rdkit"] == IDENTITY_RDKIT
    assert header["raw_bits"] == 16_551
    assert header["rows"] == 16
    assert header["numpy"] == np.__version__
    assert from_plain(header["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert from_plain(header["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "identity_data.py",
            _SCRIPTS_DIR / "pubchem_data.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
        )
    }
    with pytest.raises(FileExistsError):
        identity_data.prepare_identity(source, output=output, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        identity_data.prepare_identity(source, output=source, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        identity_data.prepare_identity(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    empty = tmp_path / "empty"
    identity_data.prepare_identity(source, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "molecules.parquet").num_rows == 0
    packed = cast(
        "NDArray[np.uint8]",
        np.load(empty / "fingerprints.npy", allow_pickle=False),
    )
    assert packed.shape == (0, 2069)
    _ = IdentityTable.Config(working_dir=empty).make()


def test_identity_scientific_release_is_explicit() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Identity preparation requires RDKit 2026\.03\.3\.$",
    ):
        identity_data.IdentityChemistry(SIMULATOR_RDKIT)


def test_complete_independent_sdk_assembly_and_failure_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = PrimitiveIdentity(IDENTITY_RDKIT)

    def construct(version: str) -> PrimitiveIdentity:
        assert version == IDENTITY_RDKIT
        return observed

    monkeypatch.setattr(identity_data, "PubChemChemistry", construct)
    monkeypatch.setattr(identity_data, "CalcMolFormula", observed.formula)
    monkeypatch.setattr(identity_data, "ExactMolWt", observed.mass)
    chemistry = identity_data.IdentityChemistry(IDENTITY_RDKIT)
    for index, row in enumerate(observed.rows):
        actual, packed = chemistry.record(row["smiles"])
        assert actual == row
        np.testing.assert_array_equal(packed, identity_fingerprints()[index])
    for name in ("standard-error", "properties-error", "canonical-error"):
        actual, _ = chemistry.record(name)
        assert actual["standardized"] is None
        assert actual["formula"] == name
        assert actual["metric"] == "null"
        assert not actual["properties_valid"]
        assert not actual["fingerprint_valid"]

    def mass_failure(molecule: PrimitiveMolecule) -> float:
        del molecule
        raise OSError("independent exact-mass failure")

    monkeypatch.setattr(identity_data, "ExactMolWt", mass_failure)
    row, fingerprint = chemistry.record(observed.rows[0]["smiles"])
    assert not row["properties_valid"]
    assert row["fingerprint_valid"]
    assert row["formula"] == observed.rows[0]["formula"]
    assert row["standard_formula_lookup"] == observed.rows[0]["standard_formula_lookup"]
    np.testing.assert_array_equal(fingerprint, identity_fingerprints()[0])


def test_properties_and_identity_reject_what_the_sdk_stops_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = PrimitiveIdentity(IDENTITY_RDKIT)

    def construct(version: str) -> PrimitiveIdentity:
        assert version == IDENTITY_RDKIT
        return observed

    monkeypatch.setattr(identity_data, "PubChemChemistry", construct)
    chemistry = identity_data.IdentityChemistry(IDENTITY_RDKIT)
    assert chemistry.properties("invalid") is None
    observed.parses_left = 1
    assert chemistry.identity(observed.rows[0]["smiles"]) is None


class ObservedIdentity:
    """Supply the staged identity rows as SDK outputs at the writer boundary."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.rows = identity_rows()
        self.fingerprints = identity_fingerprints()

    def record(self, smiles: str) -> tuple[IdentityRow, NDArray[np.uint8]]:
        index = next(
            index for index, row in enumerate(self.rows) if row["smiles"] == smiles
        )
        return self.rows[index], self.fingerprints[index]


class PrimitiveIdentity(ObservedIdentity):
    """Replay the staged SDK fields while preserving independent call boundaries."""

    def __init__(self, version: str) -> None:
        super().__init__(version)
        self.parses_left: int | None = None
        self.positions = {row["smiles"]: index for index, row in enumerate(self.rows)}
        for index, row in enumerate(self.rows):
            value = row["standardized"]
            if value is not None:
                self.positions.setdefault(value, index)
        self.chemistry = types.SimpleNamespace(
            standardize=self.standardize,
            chem=types.SimpleNamespace(
                MolFromSmiles=self.parse,
                RemoveStereochemistry=self.remove_stereo,
                MolToInchiKey=self.metric_key,
            ),
            tautomers=types.SimpleNamespace(Canonicalize=self.canonicalize),
        )

    def standardize(self, value: str) -> str | None:
        if value == "standard-error":
            raise OSError("independent standardization failure")
        return (
            self.rows[self.positions[value]]["standardized"]
            if value in self.positions
            else None
        )

    def parse(self, value: str) -> PrimitiveMolecule | None:
        if self.parses_left is not None:
            if self.parses_left == 0:
                return None
            self.parses_left -= 1
        if value in {"properties-error", "canonical-error"}:
            return PrimitiveMolecule(value, formula="", mass=0.0, heavy=0, elements=())
        if value not in self.positions or value == "invalid":
            return None
        row = self.rows[self.positions[value]]
        metric = cast(  # pyrefly: ignore[redundant-cast] -- ty or basedpyright needs this cast; pyrefly already infers it.
            "tuple[str, tuple[tuple[str, int], ...]] | None",
            json_loads(
                row["metric"] if value == row["smiles"] else row["standard_metric"],
                tuple[str, tuple[tuple[str, int], ...]] | None,
            ),
        )
        return PrimitiveMolecule(
            value,
            formula=row["formula"]
            if value == row["smiles"]
            else row["standard_formula"],
            mass=row["standard_mass"],
            heavy=row["standard_heavy"],
            elements=metric[1] if metric is not None else (),
        )

    def remove_stereo(self, molecule: PrimitiveMolecule) -> None:
        molecule.stereo = False

    def canonicalize(self, molecule: PrimitiveMolecule) -> PrimitiveMolecule:
        if molecule.value in {"properties-error", "canonical-error"}:
            raise ValueError("independent canonicalization failure")
        assert not molecule.stereo
        return molecule

    def metric_key(self, molecule: PrimitiveMolecule) -> str:
        row = self.rows[self.positions[molecule.value]]
        metric = cast(  # pyrefly: ignore[redundant-cast] -- ty or basedpyright needs this cast; pyrefly already infers it.
            "tuple[str, tuple[tuple[str, int], ...]] | None",
            json_loads(
                row["metric"]
                if molecule.value == row["smiles"]
                else row["standard_metric"],
                tuple[str, tuple[tuple[str, int], ...]] | None,
            ),
        )
        return metric[0] + "-UHFFFAOYSA-N" if metric is not None else ""

    def formula(self, molecule: PrimitiveMolecule) -> str:
        if molecule.value in {"properties-error", "canonical-error"}:
            raise OSError("independent formula failure")
        return molecule.formula

    def mass(self, molecule: PrimitiveMolecule) -> float:
        return molecule.mass

    def score_key(self, value: str) -> str | None:
        if value not in self.positions:
            return None
        row = self.rows[self.positions[value]]
        return row["key"] if value == row["smiles"] else row["standard_key"]

    def raw_fingerprint(self, value: str) -> NDArray[np.uint8] | None:
        index = self.positions[value]
        return (
            self.fingerprints[index] if self.rows[index]["fingerprint_valid"] else None
        )


class PrimitiveMolecule:
    def __init__(
        self,
        value: str,
        *,
        formula: str,
        mass: float,
        heavy: int,
        elements: tuple[tuple[str, int], ...],
    ) -> None:
        self.value, self.formula, self.mass, self.heavy = value, formula, mass, heavy
        self.elements = elements
        self.stereo = True
        self.GetAtoms = self.atoms
        self.GetNumHeavyAtoms = self.heavy_count

    def atoms(self) -> list[PrimitiveAtom]:
        return [
            PrimitiveAtom(element)
            for element, count in self.elements
            for _ in range(count)
        ] + [PrimitiveAtom("H")]

    def heavy_count(self) -> int:
        return self.heavy


class PrimitiveAtom:
    def __init__(self, element: str) -> None:
        self.element = element
        self.GetSymbol = self.symbol

    def symbol(self) -> str:
        return self.element


# products_test.py.
def _name(product: str) -> str:
    if product == "invalid":
        raise ValueError(product)
    return product.lower()


def test_each_group_contributes_its_first_product_once_per_canonical_name() -> None:
    groups = [("A", "ignored"), ("a",), ("B",), ("invalid",), ("skip",), ("C",)]
    assert unique_products(
        groups,
        canonical=_name,
        limit=10,
        keep=lambda product: product != "skip",
    ) == [("A", "a"), ("B", "b"), ("C", "c")]


def test_the_cap_counts_accepted_products_and_zero_still_admits_the_first() -> None:
    groups = [("invalid",), ("A",), ("B",), ("C",)]
    assert unique_products(groups, canonical=_name, limit=2) == [
        ("A", "a"),
        ("B", "b"),
    ]
    assert unique_products(groups, canonical=_name, limit=0) == [("A", "a")]
    assert unique_products([], canonical=_name, limit=3) == []


# rdkit_backend_test.py.
@dataclass(frozen=True, slots=True, kw_only=True)
class Element:
    symbol: str

    def GetSymbol(self) -> str:  # noqa: N802 -- RDKit boundary fake.
        return self.symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class _RdkitBackendStructure:
    name: str

    def GetAtoms(self) -> list[Element]:  # noqa: N802 -- RDKit boundary fake.
        return [Element(symbol=name) for name in ("O", "C", "H", "C")]

    def GetNumHeavyAtoms(self) -> int:  # noqa: N802 -- RDKit boundary fake.
        return 3


class _RdkitBackendPrimitives(ModuleType):
    __version__: Final = IDENTITY_RDKIT

    def __init__(self) -> None:
        super().__init__("rdkit")
        self.calls: list[tuple[str, object]] = []

    def MolFromSmiles(self, text: str) -> _RdkitBackendStructure | None:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("parse", text))
        if text == "error":
            raise ValueError("Bad chemistry input.")
        return None if text == "invalid" else _RdkitBackendStructure(name=text)

    def MolToSmiles(self, mol: _RdkitBackendStructure) -> str:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("smiles", mol.name))
        return mol.name + "-canonical"

    def MolToInchiKey(self, mol: _RdkitBackendStructure) -> str:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("key", mol.name))
        return "" if mol.name == "empty" else "ABCDEFGHIJKLMN-" + mol.name

    def Mol(self, mol: _RdkitBackendStructure) -> _RdkitBackendStructure:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("copy", mol.name))
        return _RdkitBackendStructure(name=mol.name + "-copy")

    def RemoveStereochemistry(self, mol: _RdkitBackendStructure) -> None:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("stereo", mol.name))

    def GetFormalCharge(self, mol: _RdkitBackendStructure) -> int:  # noqa: N802 -- RDKit boundary fake.
        return int(mol.name == "charged")

    def TautomerEnumerator(self) -> _RdkitBackendPrimitives:  # noqa: N802 -- RDKit boundary fake.
        return self

    def LargestFragmentChooser(self) -> _RdkitBackendPrimitives:  # noqa: N802 -- RDKit boundary fake.
        return self

    def Uncharger(self) -> _RdkitBackendPrimitives:  # noqa: N802 -- RDKit boundary fake.
        return self

    def Canonicalize(self, mol: _RdkitBackendStructure) -> _RdkitBackendStructure:  # noqa: N802 -- RDKit boundary fake.
        self.calls.append(("tautomer", mol.name))
        return mol

    def choose(self, mol: _RdkitBackendStructure) -> _RdkitBackendStructure:
        self.calls.append(("largest", mol.name))
        return mol

    def uncharge(self, mol: _RdkitBackendStructure) -> _RdkitBackendStructure:
        self.calls.append(("uncharge", mol.name))
        return mol

    def ExactMolWt(self, mol: _RdkitBackendStructure) -> float:  # noqa: N802 -- RDKit boundary fake.
        del mol
        return 46.041864812

    def CalcMolFormula(self, mol: _RdkitBackendStructure) -> str:  # noqa: N802 -- RDKit boundary fake.
        del mol
        return "C2H6O"

    def GetMorganFeatureAtomInvGen(self) -> _RdkitBackendPrimitives:  # noqa: N802 -- RDKit boundary fake.
        return self

    def GetMorganGenerator(  # noqa: N802 -- RDKit boundary fake.
        self,
        *,
        radius: int,
        fpSize: int,  # noqa: N803 -- RDKit boundary keyword.
        atomInvariantsGenerator: object = None,  # noqa: N803 -- RDKit boundary keyword.
    ) -> Fingerprint:
        self.calls.append(("morgan", (radius, fpSize, atomInvariantsGenerator is self)))
        return Fingerprint(index=radius - 2 + 2 * (atomInvariantsGenerator is self))

    def GetRDKitFPGenerator(self, *, fpSize: int, maxPath: int) -> Fingerprint:  # noqa: N802, N803 -- RDKit boundary fake.
        self.calls.append(("rdk", (fpSize, maxPath)))
        return Fingerprint(index=3)

    def GetAtomPairGenerator(self, *, fpSize: int) -> Fingerprint:  # noqa: N802, N803 -- RDKit boundary fake.
        self.calls.append(("pair", fpSize))
        return Fingerprint(index=4)

    def GetTopologicalTorsionGenerator(self, *, fpSize: int) -> Fingerprint:  # noqa: N802, N803 -- RDKit boundary fake.
        self.calls.append(("torsion", fpSize))
        return Fingerprint(index=5)

    def GenMACCSKeys(self, mol: _RdkitBackendStructure) -> list[int]:  # noqa: N802 -- RDKit boundary fake.
        del mol
        return [1, 0, 1]


@dataclass(frozen=True, slots=True, kw_only=True)
class Fingerprint:
    index: int

    def GetFingerprintAsNumPy(  # noqa: N802 -- RDKit boundary fake.
        self,
        mol: _RdkitBackendStructure,
    ) -> NDArray[np.uint8]:
        del mol
        return np.array([self.index % 2, self.index // 2 % 2], dtype=np.uint8)


def _rdkit_backend_install(
    monkeypatch: pytest.MonkeyPatch,
    library: _RdkitBackendPrimitives,
) -> None:
    """Replace every lazily imported scientific name of the backend by one fake."""
    for name, value in (
        ("TautomerEnumerator", library.TautomerEnumerator),
        ("LargestFragmentChooser", library.LargestFragmentChooser),
        ("Uncharger", library.Uncharger),
        ("GetMorganGenerator", library.GetMorganGenerator),
        ("GetMorganFeatureAtomInvGen", library.GetMorganFeatureAtomInvGen),
        ("GetRDKitFPGenerator", library.GetRDKitFPGenerator),
        ("GetAtomPairGenerator", library.GetAtomPairGenerator),
        ("GetTopologicalTorsionGenerator", library.GetTopologicalTorsionGenerator),
        ("GenMACCSKeys", library.GenMACCSKeys),
        ("ExactMolWt", library.ExactMolWt),
        ("CalcMolFormula", library.CalcMolFormula),
    ):
        monkeypatch.setattr(rdkit_backend, name, value)
    monkeypatch.setattr(rdkit_backend, "rdkit", library)
    monkeypatch.setattr(rdkit_backend, "Chem", library)


def test_complete_structure_preparation_and_primitive_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _RdkitBackendPrimitives()
    _rdkit_backend_install(monkeypatch, library)
    chemistry = Chemistry(IDENTITY_RDKIT)
    assert library.calls == [
        ("morgan", (2, 4096, False)),
        ("morgan", (3, 4096, False)),
        ("morgan", (2, 2048, True)),
        ("rdk", (2048, 6)),
        ("pair", 2048),
        ("torsion", 2048),
    ]
    record = chemistry.record("CCO")
    assert record is not None
    assert (record.smiles, record.formula, record.mass, record.heavy_atoms) == (
        "CCO",
        "C2H6O",
        46.041864812,
        3,
    )
    assert (record.key, record.raw_key, record.metric_key) == ("ABCDEFGHIJKLMN",) * 3
    assert record.composition == (("C", 2), ("O", 1))
    np.testing.assert_array_equal(
        record.packed,
        np.packbits([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]),
    )
    assert library.calls[-8:] == [
        ("parse", "CCO"),
        ("key", "CCO"),
        ("tautomer", "CCO"),
        ("key", "CCO"),
        ("copy", "CCO"),
        ("stereo", "CCO-copy"),
        ("tautomer", "CCO-copy"),
        ("key", "CCO-copy"),
    ]
    assert chemistry.standardize("CCO") == "CCO-canonical"
    assert library.calls[-5:] == [
        ("parse", "CCO"),
        ("largest", "CCO"),
        ("uncharge", "CCO"),
        ("stereo", "CCO"),
        ("smiles", "CCO"),
    ]
    assert chemistry.standardize("charged") is None
    assert chemistry.standardize("charged", keep_charged=True) == "charged-canonical"
    for value in ("invalid", "error"):
        assert chemistry.record(value) is None
        assert chemistry.standardize(value) is None
    assert chemistry.record("empty") is None


def test_library_version_is_checked_before_any_primitive_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _RdkitBackendPrimitives()
    _rdkit_backend_install(monkeypatch, library)
    with pytest.raises(
        ValueError,
        match=rf"^Preparation requires RDKit {SIMULATOR_RDKIT}; got {IDENTITY_RDKIT}\.$",
    ):
        _ = Chemistry(SIMULATOR_RDKIT)
    assert library.calls == []


def test_reaction_context_builds_only_molecular_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _RdkitBackendPrimitives()
    _rdkit_backend_install(monkeypatch, library)
    context = MolecularContext(IDENTITY_RDKIT)
    assert context.version == IDENTITY_RDKIT
    assert context.chem is library
    assert library.calls == []


# reaction_data_test.py.
def test_reaction_asset_writer_keeps_all_requests_outputs_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _DerivativesPrimitives()
    _derivatives_install(monkeypatch, library)
    monkeypatch.setattr(reaction_data, "Chem", library)
    generator = DerivativeGenerator()
    generator.rules = reaction_rules()[:3]

    def construct(
        *,
        rules: Sequence[Reaction] | None,
        on_application: Callable[[Chem.Mol, Reaction, int, int], None],
    ) -> DerivativeGenerator:
        assert rules is None
        generator.on_application = on_application
        return generator

    source = tmp_path / "requests.parquet"
    requests = [
        {"smiles": "parent", "delta": 15.9949146196},
        {"smiles": "parent", "delta": 0.0, "operation": "apply", "rule": 0},
        {"smiles": "invalid", "delta": 0.0, "operation": "apply", "rule": 0},
        {"smiles": "parent", "delta": 1000.0, "operation": "match"},
    ]
    fields = {name for row in requests for name in row}
    parquet.write_table(
        Table.from_pydict(
            {name: [row.get(name) for row in requests] for name in fields},
        ),
        source,
    )
    output = tmp_path / "reactions"
    with patch.object(reaction_data, "DerivativeGenerator", side_effect=construct):
        reaction_data.prepare_reactions(
            source,
            output=output,
            chemistry=_DerivativesPreparation(),
        )
    data = parquet.read_table(output / "results.parquet").to_pylist()
    assert parquet.read_table(output / "requests.parquet").to_pylist() == [
        normalize_request(from_plain(row, ReactionRequest)) for row in requests
    ]
    assert data == [
        {
            "input_smiles": "parent",
            "products": ["parentA", "parentB"],
            "plans": [[0], [0]],
            "matches": [],
            "applications": "\n".join(
                f"parent\t{rule.name}\t32\t3" for rule in generator.rules
            ),
        },
        {
            "input_smiles": "parent",
            "products": ["parent", "parentA", "parentB"],
            "plans": [[0], [0], [0]],
            "matches": [],
            "applications": "parent\thydroxyl_arom\t64\t3",
        },
        *(
            {
                "input_smiles": value,
                "products": [],
                "plans": [],
                "matches": [],
                "applications": "",
            }
            for value in ("invalid", "parent")
        ),
    ]
    assert parquet.read_table(output / "products.parquet").to_pylist() == [
        {"request": request, "position": position, "smiles": smiles, "plan": [0]}
        for request, values in (
            (0, ["parentA", "parentB"]),
            (1, ["parent", "parentA", "parentB"]),
        )
        for position, smiles in enumerate(values)
    ]
    vocabulary = parquet.read_table(output / "rules.parquet")
    assert from_plain(vocabulary.column("name").to_pylist(), list[str]) == [
        rule.name for rule in generator.rules
    ]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["requests"] == 4
    assert manifest["products"] == 5
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["format_version"] == 1
    assert manifest["default_products"] == 60
    assert set(from_plain(manifest["input"], dict[str, str])) == {str(source.resolve())}
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "reaction_data.py",
        "derivatives.py",
        "../chemistry.py",
        "rdkit_backend.py",
        "../v6_reactions.tsv",
        "../main_engine.py",
    }
    with (
        patch.object(reaction_data, "DerivativeGenerator", side_effect=construct),
        pytest.raises(FileExistsError),
    ):
        reaction_data.prepare_reactions(
            source,
            output=output,
            chemistry=_DerivativesPreparation(),
        )
    for index, request in enumerate(
        [
            {"smiles": "parent", "delta": 0.0, "operation": "apply", "rule": -1},
            {
                "smiles": "parent",
                "delta": 0.0,
                "operation": "apply",
                "rule": len(generator.rules),
            },
            {"smiles": "parent", "delta": 0.0, "operation": "unknown"},
        ],
    ):
        parquet.write_table(Table.from_pylist([request]), source)
        with (
            patch.object(reaction_data, "DerivativeGenerator", side_effect=construct),
            pytest.raises(
                ValueError,
                match=r"Reaction rule index exceeds the vocabulary\.|operation",
            ),
        ):
            reaction_data.prepare_reactions(
                source,
                output=tmp_path / str(index),
                chemistry=_DerivativesPreparation(),
            )
    parquet.write_table(Table.from_pydict({"smiles": [], "delta": []}), source)
    with patch.object(reaction_data, "DerivativeGenerator", side_effect=construct):
        reaction_data.prepare_reactions(
            source,
            output=tmp_path / "empty",
            chemistry=_DerivativesPreparation(),
        )
    assert parquet.read_table(tmp_path / "empty" / "results.parquet").num_rows == 0


def test_shifted_search_keeps_direct_matches_and_adds_the_shifted_peak() -> None:
    store = PeakStore(
        offsets=np.array([0, 2], np.int64),
        mz=np.array([10.0, 15.0], np.float32),
        intensity=np.array([1.0, 1.0], np.float32),
    )
    search = MainSearch.Config().make()
    query = (np.array([10.0, 20.0]), np.array([1.0, 1.0]))
    rows = np.array([0], np.int64)
    # Unshifted, one of two equal peaks matches: the mixture entropy costs one half.
    assert search.score(query, store=store, candidates=rows).tolist() == pytest.approx(
        [0.5],
    )
    # A 5 Da shift keeps the direct 10 Da match and explains 20 Da by the 15 Da peak.
    shifted = search.score(query, store=store, candidates=rows, shifts=np.array([5.0]))
    assert shifted.tolist() == pytest.approx([1.0])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
