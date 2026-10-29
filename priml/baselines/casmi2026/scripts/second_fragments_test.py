"""Checks for the second engine's unrounded neutral-fragment preparation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import json

from pyarrow import Table, parquet

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import ELEMENT_MASSES, IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts import second_fragments
from priml.baselines.casmi2026.scripts.second_fragments import (
    FragmentChemistry,
    fragment_masses,
    prepare_second_fragments,
)


if TYPE_CHECKING:
    from pathlib import Path


_H = ELEMENT_MASSES["H"]
_C = ELEMENT_MASSES["C"]
_O = ELEMENT_MASSES["O"]


def _molecule(
    atoms: list[tuple[str, int]],
    bonds: list[tuple[int, int]],
) -> SimpleNamespace:
    members = [
        SimpleNamespace(
            GetIdx=lambda index=index: index,
            GetSymbol=lambda symbol=symbol: symbol,
            GetTotalNumHs=lambda count=count: count,
        )
        for index, (symbol, count) in enumerate(atoms)
    ]
    links = [
        SimpleNamespace(
            GetBeginAtomIdx=lambda left=left: left,
            GetEndAtomIdx=lambda right=right: right,
        )
        for left, right in bonds
    ]
    return SimpleNamespace(
        GetNumAtoms=lambda: len(members),
        GetAtoms=lambda: members,
        GetBonds=lambda: links,
    )


_MOLECULES = {
    # Ethanol heavy-atom chain with its hydrogens.
    "CCO": _molecule([("C", 3), ("C", 2), ("O", 1)], [(0, 1), (1, 2)]),
    "[Xx]": _molecule([("Xx", 0)], []),
    "C": _molecule([("C", 4)], []),
}


_CHEM = SimpleNamespace(MolFromSmiles=_MOLECULES.get)


class _Context:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.chem = _CHEM


@pytest.fixture(autouse=True)
def fake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(second_fragments, "MolecularContext", _Context)


def test_masses_union_parent_one_cuts_and_two_cuts_in_sorted_order() -> None:
    weights = np.array([1.0, 2.0, 3.0])
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        fragment_masses(weights, bonds=bonds),
        [1.0, 2.0, 3.0, 5.0, 6.0],
    )
    np.testing.assert_array_equal(
        fragment_masses(weights, bonds=bonds, max_breaks=1),
        [1.0, 3.0, 5.0, 6.0],
    )


def test_masses_fall_back_to_the_parent_without_bonds_or_above_the_cap() -> None:
    weights = np.array([1.0, 2.0, 3.0])
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        fragment_masses(weights, bonds=np.empty((0, 2), dtype=np.int64)),
        [6.0],
    )
    np.testing.assert_array_equal(
        fragment_masses(weights, bonds=bonds, max_bonds=1),
        [6.0],
    )


def test_a_wrong_release_is_rejected() -> None:
    with pytest.raises(ValueError, match=rf"require RDKit {IDENTITY_RDKIT}\."):
        _ = FragmentChemistry("0.0.0")


def test_graph_weights_include_attached_hydrogens_and_bonds_keep_order() -> None:
    graph = FragmentChemistry(IDENTITY_RDKIT).graph("CCO")
    assert graph is not None
    weights, bonds = graph
    np.testing.assert_allclose(weights, [_C + 3 * _H, _C + 2 * _H, _O + _H])
    np.testing.assert_array_equal(bonds, [[0, 1], [1, 2]])


def test_graph_without_bonds_has_an_empty_two_column_matrix() -> None:
    graph = FragmentChemistry(IDENTITY_RDKIT).graph("C")
    assert graph is not None
    assert graph[1].shape == (0, 2)


@pytest.mark.parametrize("smiles", ["unparsable", "[Xx]"])
def test_graph_rejects_unparsable_and_unknown_element_structures(smiles: str) -> None:
    assert FragmentChemistry(IDENTITY_RDKIT).graph(smiles) is None


def test_prepare_writes_ragged_masses_graphs_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "pool.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "unparsable", "C"]}),
        source,
    )
    output = tmp_path / "out"
    prepare_second_fragments(source, output=output, version=IDENTITY_RDKIT)
    ethanol = sorted(
        {
            _C + 3 * _H + _C + 2 * _H + _O + _H,
            _C + 3 * _H,
            _C + 2 * _H + _O + _H,
            _C + 3 * _H + _C + 2 * _H,
            _O + _H,
            _C + 2 * _H,
        },
    )
    np.testing.assert_array_equal(
        np.load(output / "frag_off.npy"),
        [0, len(ethanol), len(ethanol), len(ethanol) + 1],
    )
    np.testing.assert_allclose(
        np.load(output / "frag_mass.npy"),
        [*ethanol, _C + 4 * _H],
    )
    rows = parquet.read_table(output / "graphs.parquet").to_pylist()
    assert [(row["smiles"], row["valid"]) for row in rows] == [
        ("CCO", True),
        ("unparsable", False),
        ("C", True),
    ]
    assert rows[0]["bonds"] == [[0, 1], [1, 2]]
    assert rows[1]["weights"] == []
    assert rows[1]["bonds"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["masses"] == len(ethanol) + 1
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["max_breaks"] == 2
    assert manifest["max_bonds"] == 34
    assert set(manifest["generator"]) == {
        "second_fragments.py",
        "rdkit_backend.py",
    }
    assert list(manifest["inputs"]) == [str(source.resolve())]


def test_prepare_refuses_to_overwrite_its_input(tmp_path: Path) -> None:
    source = tmp_path / "pool.parquet"
    parquet.write_table(Table.from_pydict({"smiles": ["C"]}), source)
    with pytest.raises(ValueError, match="aliases protected input artifact"):
        prepare_second_fragments(source, output=source, version=IDENTITY_RDKIT)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
