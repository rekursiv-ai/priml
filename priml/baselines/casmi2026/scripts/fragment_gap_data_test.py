"""Check the gap writer and the SDK's element-mass fallback boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.custom_types import GapGraph
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import fragment_gap_data


if TYPE_CHECKING:
    from pathlib import Path


class _Graphs:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def graph(self, smiles: str) -> GapGraph | None:
        return (
            None
            if smiles == "invalid"
            else GapGraph(
                weights=np.array([1.0, 2.0]),
                bonds=np.array([[0, 1]], dtype=np.int64),
                pieces=1,
            )
        )


def test_gap_writer_preserves_bounded_groups_duplicates_and_empty_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fragment_gap_data, "GapChemistry", _Graphs)
    source, output = tmp_path / "input.parquet", tmp_path / "output"
    parquet.write_table(Table.from_pydict({"smiles": ["CC", "invalid", "CC"]}), source)
    fragment_gap_data.prepare_fragment_gap(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=2,
    )
    raw = parquet.ParquetFile(output / "graphs.parquet")
    assert raw.num_row_groups == 2
    assert raw.read().to_pylist() == [
        {
            "smiles": "CC",
            "valid": True,
            "weights": [1.0, 2.0],
            "bonds": [[0, 1]],
            "pieces": 1,
        },
        {"smiles": "invalid", "valid": False, "weights": [], "bonds": [], "pieces": 0},
        {
            "smiles": "CC",
            "valid": True,
            "weights": [1.0, 2.0],
            "bonds": [[0, 1]],
            "pieces": 1,
        },
    ]
    header = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert header["format_version"] == 1
    assert header["rdkit"] == IDENTITY_RDKIT
    assert header["rows"] == 3
    assert header["numpy"] == np.__version__
    assert set(header) == {
        "format_version",
        "rdkit",
        "numpy",
        "rows",
        "inputs",
        "generator",
    }
    assert from_plain(header["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert set(from_plain(header["generator"], dict[str, str])) == {
        "fragment_gap_data.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    with pytest.raises(FileExistsError):
        fragment_gap_data.prepare_fragment_gap(
            source,
            output=output,
            version=IDENTITY_RDKIT,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    empty = tmp_path / "empty"
    fragment_gap_data.prepare_fragment_gap(source, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "graphs.parquet").num_rows == 0
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        fragment_gap_data.prepare_fragment_gap(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )


def test_gap_sdk_retains_atom_indices_isotope_ignorance_and_periodic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = (
        types.SimpleNamespace(
            GetIdx=lambda: 1,
            GetSymbol=lambda: "C",
            GetTotalNumHs=lambda: 3,
        ),
        types.SimpleNamespace(
            GetIdx=lambda: 0,
            GetSymbol=lambda: "Cu",
            GetTotalNumHs=lambda: 0,
        ),
    )
    molecule = types.SimpleNamespace(
        GetNumAtoms=lambda: 2,
        GetAtoms=lambda: atoms,
        GetBonds=lambda: (
            types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        ),
    )
    fallback: list[str] = []

    def isotope(element: str) -> float:
        fallback.append(element)
        return 43.0

    def parse(value: str) -> types.SimpleNamespace | None:
        if value == "error":
            raise ValueError("Invalid scientific input.")
        return molecule if value != "invalid" else None

    def components(value: types.SimpleNamespace) -> tuple[tuple[int, ...], ...]:
        assert value is molecule
        return ((0, 1),)

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(
            chem=types.SimpleNamespace(
                MolFromSmiles=parse,
                GetPeriodicTable=lambda: types.SimpleNamespace(
                    GetMostCommonIsotopeMass=isotope,
                ),
                GetMolFrags=components,
            ),
        )

    monkeypatch.setattr(fragment_gap_data, "MolecularContext", context)
    chemistry = fragment_gap_data.GapChemistry(IDENTITY_RDKIT)
    graph = chemistry.graph("isotope")
    assert graph is not None
    np.testing.assert_array_equal(graph.weights, [43.0, 12.0 + 3 * 1.00782503207])
    np.testing.assert_array_equal(graph.bonds, [[1, 0]])
    assert graph.pieces == 1
    assert fallback == ["Cu"]
    assert chemistry.graph("invalid") is None
    assert chemistry.graph("error") is None
    assert chemistry.graph("") is None
    molecule.GetNumAtoms = lambda: 0
    assert chemistry.graph("empty") is None
    with pytest.raises(ValueError, match=r"^Gap graphs require RDKit 2026\.03\.3\.$"):
        _ = fragment_gap_data.GapChemistry(SIMULATOR_RDKIT)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
