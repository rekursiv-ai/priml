"""Checks for the ice_data module."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import ice_data
from priml.baselines.casmi2026.scripts.ice_data import prepare_ice_graphs


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import IceRow


def test_preparation_requires_the_simulator_version_and_positive_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"requires RDKit 2025\.03\.6"):
        prepare_ice_graphs(
            tmp_path / "request.parquet",
            output=tmp_path / "roots",
            version=IDENTITY_RDKIT,
        )
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        prepare_ice_graphs(
            tmp_path / "request.parquet",
            output=tmp_path / "roots",
            version=SIMULATOR_RDKIT,
            batch_rows=0,
        )


def test_canonical_root_retains_isotopes_hydrogens_and_bond_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carbon = types.SimpleNamespace(
        GetSymbol=lambda: "C",
        GetIsotope=lambda: 13,
        GetNumImplicitHs=lambda: 2,
        GetNumExplicitHs=lambda: 1,
    )
    nitrogen = types.SimpleNamespace(
        GetSymbol=lambda: "N",
        GetIsotope=lambda: 0,
        GetNumImplicitHs=lambda: 1,
        GetNumExplicitHs=lambda: 2,
    )
    bonds = [_bond(kind) for kind in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")]
    molecule = types.SimpleNamespace(
        GetAtoms=lambda: [carbon, nitrogen],
        GetBonds=lambda: bonds,
    )
    events: list[str] = []

    def parse(value: str) -> object:
        events.append("parse:" + value)
        return None if value == "invalid" else molecule

    def remove_stereo(value: object) -> None:
        assert value is molecule
        events.append("remove_stereo")

    def inchi(value: object) -> str:
        assert value is molecule
        events.append("inchi")
        return "inchi"

    def canonicalize(value: object) -> object:
        events.append("tautomer")
        return value

    def kekulize(value: object, clear: bool) -> None:
        assert value is molecule
        assert clear
        events.append("kekulize")

    def isotope_mass(element: str, isotope: int) -> float:
        return 13.5 if (element, isotope) == ("C", 13) else 0.0

    def from_inchi(value: str) -> types.SimpleNamespace:
        del value
        return molecule

    def standard_mass(element: str) -> float:
        return float(len(element))

    def canonical_smiles(value: object) -> str:
        del value
        return "canonical"

    def enumerator(value: object) -> types.SimpleNamespace:
        del value
        return types.SimpleNamespace(Canonicalize=canonicalize)

    def formula(value: object) -> str:
        del value
        return "CH6N+"

    periodic = types.SimpleNamespace(
        GetMostCommonIsotopeMass=standard_mass,
        GetMassForIsotope=isotope_mass,
    )
    chem = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical_smiles,
        MolToInchi=inchi,
        MolFromInchi=from_inchi,
        RemoveStereochemistry=remove_stereo,
        Kekulize=kekulize,
        GetPeriodicTable=lambda: periodic,
    )

    def construct(version: str) -> types.SimpleNamespace:
        assert version == SIMULATOR_RDKIT
        return types.SimpleNamespace(chem=chem)

    monkeypatch.setattr(ice_data, "MolecularContext", construct)
    parameters = types.SimpleNamespace()
    monkeypatch.setattr(ice_data, "CleanupParameters", lambda: parameters)
    monkeypatch.setattr(ice_data, "TautomerEnumerator", enumerator)
    monkeypatch.setattr(ice_data, "CalcMolFormula", formula)
    chemistry = ice_data.IceChemistry(SIMULATOR_RDKIT, directory=tmp_path)
    transforms = tmp_path / "tautomer-transforms.txt"
    assert parameters.tautomerTransformsFile == str(transforms)
    assert transforms.read_text().startswith("// Name SMARTS Bonds Charges\n")
    np.testing.assert_array_equal(
        chemistry.element_masses,
        [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1],
    )
    row = chemistry.row("input")
    assert events == [
        "parse:input",
        "remove_stereo",
        "parse:canonical",
        "inchi",
        "tautomer",
        "parse:canonical",
        "kekulize",
    ]
    assert row["input_smiles"] == "input"
    assert row["smiles"] == "canonical"
    assert row["valid"]
    expected = np.zeros((2, 32))
    expected[0, [0, 22, 29]] = 1
    expected[1, [1, 21, 29]] = 1
    np.testing.assert_array_equal(row["features"], expected)
    assert row["formula"] == [1, 1, 0, 0, 0, 0, 0, 6, *([0] * 10)]
    assert row["bonds"] == [[0, 1]] * 4
    assert row["bond_orders"] == [1, 2, 3, 2]
    assert row["hydrogens"] == [3, 3]
    assert row["atom_masses"] == [13.5, 1.0]
    assert row["hydrogenated_masses"] == [16.5, 4.0]
    assert row["mass"] == 20.5
    rejected = chemistry.row("invalid")
    assert not rejected["valid"]
    assert rejected["smiles"] == ""
    assert np.isnan(rejected["mass"])
    for stage in ("stereo", "inchi", "canonical"):
        calls = (
            iter((molecule, None))
            if stage == "stereo"
            else iter((molecule, molecule, None))
        )
        chem.MolFromSmiles = partial(_next_molecule, calls)
        chem.MolFromInchi = _missing_molecule if stage == "inchi" else from_inchi
        with pytest.raises(ValueError, match="molecule is not None"):
            _ = chemistry.row("input")


def test_batched_writer_retains_invalid_rows_empty_inputs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["invalid", "C", None, "C", "O"]}),
        request,
    )
    monkeypatch.setattr(ice_data, "IceChemistry", _Encoder)
    output = tmp_path / "roots"
    prepare_ice_graphs(request, output=output, version=SIMULATOR_RDKIT, batch_rows=2)
    source = parquet.ParquetFile(output / "ice.parquet")
    assert source.num_row_groups == 2
    assert source.read_row_group(0).column("input_smiles").to_pylist() == [
        "invalid",
        "C",
    ]
    assert source.read_row_group(1).column("input_smiles").to_pylist() == ["O"]
    assert source.read().column("valid").to_pylist() == [False, True, True]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 3
    assert manifest["valid_rows"] == 2
    assert manifest["node_features"] == 32
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["numpy"] == np.__version__
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "ice_data.py",
        "rdkit_backend.py",
    }
    assert (
        manifest["tautomer_transforms_sha256"]
        == hashlib.sha256(
            b"test input data",
        ).hexdigest()
    )
    np.testing.assert_array_equal(
        cast(
            "NDArray[np.generic]",
            np.load(output / "element_masses.npy", allow_pickle=False),
        ),
        np.arange(18)[:, None],
    )
    with pytest.raises(FileExistsError):
        prepare_ice_graphs(request, output=output, version=SIMULATOR_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    empty = tmp_path / "empty"
    prepare_ice_graphs(request, output=empty, version=SIMULATOR_RDKIT)
    assert parquet.read_table(empty / "ice.parquet").to_pylist() == []
    manifest = json_loads((empty / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == manifest["valid_rows"] == 0


def _bond(kind: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        GetBeginAtomIdx=lambda: 0,
        GetEndAtomIdx=lambda: 1,
        GetBondType=lambda: kind,
    )


def _next_molecule(values: Iterator[object], smiles: str) -> object:
    del smiles
    return next(values)


def _missing_molecule(inchi: str) -> None:
    del inchi


class _Encoder:
    def __init__(self, version: str, *, directory: Path) -> None:
        assert version == SIMULATOR_RDKIT
        _ = (directory / "tautomer-transforms.txt").write_text("test input data")
        self.element_masses = np.arange(18, dtype=np.float64)

    def row(self, smiles: str) -> IceRow:
        return {
            "input_smiles": smiles,
            "smiles": smiles,
            "valid": smiles != "invalid",
            "features": [],
            "bonds": [],
            "bond_orders": [],
            "formula": [],
            "hydrogens": [],
            "atom_masses": [],
            "hydrogenated_masses": [],
            "mass": 0.0,
        }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
