"""Checks for the atomic_data module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from pyarrow import Table, parquet
from treekle import json_loads

import pytest
import torch

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import atomic_data
from priml.baselines.casmi2026.scripts.atomic_data import prepare_graphs


if TYPE_CHECKING:
    from pathlib import Path

    from rdkit import Chem


def _atom(
    *,
    number: int = 7,
    degree: int = 0,
    hydrogens: int = 3,
    charge: int = 0,
    hybrid: str = "SP3",
    chiral: str = "CHI_UNSPECIFIED",
    partial_charge: str | None = "0.0",
    ring: bool = False,
    ring_size_filled: int = 6,
) -> Chem.Atom:
    def ring_size(size: int) -> bool:
        return ring and size == ring_size_filled

    def property_value(name: str) -> str:
        return _property(partial_charge, name)

    return cast(
        "Chem.Atom",
        SimpleNamespace(
            GetAtomicNum=lambda: number,
            GetTotalDegree=lambda: degree + hydrogens,
            GetTotalNumHs=lambda: hydrogens,
            GetFormalCharge=lambda: charge,
            GetHybridization=lambda: hybrid,
            GetChiralTag=lambda: chiral,
            GetIsAromatic=lambda: ring,
            IsInRing=lambda: ring,
            IsInRingSize=ring_size,
            GetProp=property_value,
        ),
    )


def _property(value: str | None, name: str) -> str:
    if value is None or name != "_GasteigerCharge":
        raise KeyError(name)
    return value


@pytest.mark.parametrize(
    (
        "number",
        "degree",
        "hydrogens",
        "charge",
        "hybrid",
        "chiral",
        "partial_charge",
        "active",
    ),
    [
        (
            6,
            4,
            4,
            0,
            "SP3",
            "CHI_TETRAHEDRAL_CW",
            "0.25",
            [0, 23, 28, 34, 38, 43, 46, 48, 50, 52, 53, 56, 58, 60, 62, 69],
        ),
        (
            999,
            6,
            5,
            3,
            "OTHER",
            "CHI_OTHER",
            "nan",
            [18, 25, 31, 37, 44, 45, 48, 50, 52, 54, 56, 58, 59, 64],
        ),
        (
            7,
            -1,
            -1,
            -3,
            "SP2",
            "CHI_UNSPECIFIED",
            "inf",
            [1, 31, 33, 45, 48, 50, 52, 54, 56, 58, 59, 61, 68],
        ),
        (
            7,
            -1,
            -1,
            -3,
            "SP2",
            "CHI_UNSPECIFIED",
            None,
            [1, 31, 33, 45, 48, 50, 52, 54, 56, 58, 59, 61, 68],
        ),
    ],
)
def test_complete_atom_layout_unknown_overflows_and_nonfinite_charges(
    *,
    number: int,
    degree: int,
    hydrogens: int,
    charge: int,
    hybrid: str,
    chiral: str,
    partial_charge: str | None,
    active: list[int],
) -> None:
    atom = _atom(
        number=number,
        degree=degree,
        hydrogens=hydrogens,
        charge=charge,
        hybrid=hybrid,
        chiral=chiral,
        partial_charge=partial_charge,
        ring=number == 6,
    )
    expected = torch.zeros(74)
    expected[active] = 1
    expected[73] = 0.25 if number == 6 else 0.0
    assert torch.equal(
        torch.tensor(
            atomic_data._node_features(atom, ring_count=2 if number == 6 else 0),
        ),
        expected,
    )


@pytest.mark.parametrize("kind", ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"])
@pytest.mark.parametrize(
    "stereo",
    ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "OTHER"],
)
def test_complete_bond_layout_all_types_stereochemistry_and_fused_rings(
    kind: str,
    stereo: str,
) -> None:
    conjugated, ring = kind == "AROMATIC", stereo == "STEREOZ"

    def ring_size(size: int) -> bool:
        return ring and size == 5

    bond = cast(
        "Chem.Bond",
        SimpleNamespace(
            GetBondType=lambda: kind,
            GetStereo=lambda: stereo,
            GetIsConjugated=lambda: conjugated,
            IsInRing=lambda: ring,
            IsInRingSize=ring_size,
        ),
    )
    expected = [0.0] * 28
    expected[("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER").index(kind)] = 1.0
    expected[5 + int(conjugated)] = 1.0
    expected[7 + int(ring)] = 1.0
    for position, size in enumerate(range(3, 9)):
        expected[9 + 2 * position + int(not (ring and size == 5))] = 1.0
    expected[21 + int(ring)] = 1.0
    expected[
        23 + ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "OTHER").index(stereo)
    ] = 1.0
    assert atomic_data._bond_features(bond, ring_count=2 if ring else 0) == expected


@pytest.mark.parametrize("hybrid", ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNKNOWN"])
@pytest.mark.parametrize(
    "chiral",
    [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "UNKNOWN",
    ],
)
def test_atom_hybridization_and_chirality_use_complete_ordered_vocabularies(
    hybrid: str,
    chiral: str,
) -> None:
    actual = atomic_data._node_features(
        _atom(hybrid=hybrid, chiral=chiral),
        ring_count=1,
    )
    assert actual[32:38] == [
        float(hybrid == value)
        for value in ("SP", "SP2", "SP3", "SP3D", "SP3D2", "UNKNOWN")
    ]
    assert actual[61:65] == [
        float(chiral == value)
        for value in (
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER",
        )
    ]
    assert actual[59:61] == [1.0, 0.0]


def test_atom_integer_channel_boundaries_and_every_element_group() -> None:
    for degree in range(-1, 7):
        actual = atomic_data._node_features(_atom(degree=degree), ring_count=0)
        assert actual[19:26] == [
            *[float(degree == value) for value in range(6)],
            float(degree > 5),
        ]
    for charge in range(-3, 4):
        actual = atomic_data._node_features(_atom(charge=charge), ring_count=0)
        assert actual[26:32] == [
            *[float(charge == value) for value in range(-2, 3)],
            float(abs(charge) > 2),
        ]
    for hydrogens in range(-1, 6):
        actual = atomic_data._node_features(_atom(hydrogens=hydrogens), ring_count=0)
        assert actual[39:45] == [
            *[float(hydrogens == value) for value in range(5)],
            float(hydrogens > 4),
        ]
    elements = (
        (6, 4),
        (7, 3),
        (15, 3),
        (8, 5),
        (16, 5),
        (14, 4),
        (53, 6),
        (1, 0),
        (17, 6),
        (9, 6),
        (35, 6),
        (5, 2),
        (34, 5),
        (26, 7),
        (27, 7),
        (33, 3),
        (11, 1),
        (19, 1),
    )
    assert tuple(atomic_data._elements().items()) == elements
    for position, (number, group) in enumerate(elements):
        actual = atomic_data._node_features(_atom(number=number), ring_count=0)
        assert actual[:19] == [float(index == position) for index in range(19)]
        assert actual[65:73] == [float(index == group) for index in range(8)]


def test_graph_writer_full_rows_manifest_and_glacier_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _atom()
    molecule = SimpleNamespace(
        GetAtoms=lambda: [atom],
        GetBonds=list,
        GetNumAtoms=lambda: 1,
    )

    def parse(value: str) -> SimpleNamespace | None:
        return None if value == "invalid" else molecule

    def canonical(_: object) -> str:
        return "canonical"

    def remove_hydrogens(value: object) -> object:
        return value

    def rings(_: object) -> list[list[int]]:
        return []

    def partial_charges(_: object) -> None:
        return None

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=remove_hydrogens,
        GetSymmSSSR=rings,
    )
    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    monkeypatch.setattr(atomic_data, "ComputeGasteigerCharges", partial_charges)

    def context(version: str) -> SimpleNamespace:
        return SimpleNamespace(version=version, chem=chemistry)

    monkeypatch.setattr(
        atomic_data,
        "MolecularContext",
        context,
    )
    request = tmp_path / "request.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["N", "invalid", None, "N"]}),
        request,
    )
    output = tmp_path / "output"
    prepare_graphs(request, output=output, version=SIMULATOR_RDKIT)
    rows = parquet.read_table(output / "graphs.parquet").to_pylist()
    assert [row["input_smiles"] for row in rows] == ["N", "invalid"]
    assert [row["smiles"] for row in rows] == ["canonical", ""]
    assert [row["valid"] for row in rows] == [True, False]
    assert rows[0]["degrees"] == [0]
    assert rows[0]["bonds"] == []
    assert rows[0]["weights"] == []
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["format_version"] == 1
    assert manifest["raw"] is False
    assert manifest["rows"] == 2
    assert manifest["valid_rows"] == 1
    assert manifest["node_features"] == 74
    assert manifest["bond_features"] == 28
    with pytest.raises(FileExistsError):
        prepare_graphs(request, output=output, version=SIMULATOR_RDKIT)

    parquet.write_table(
        Table.from_pydict({"smiles": ["invalid", None, "N", "N", "O"]}),
        request,
    )
    batched = tmp_path / "batched"
    prepare_graphs(
        request,
        output=batched,
        version=SIMULATOR_RDKIT,
        batch_rows=1,
    )
    source = parquet.ParquetFile(batched / "graphs.parquet")
    assert source.num_row_groups == 3
    assert source.read().to_pylist() == [
        rows[1],
        rows[0],
        {**rows[0], "input_smiles": "O"},
    ]
    assert source.read_row_group(0).to_pylist() == [rows[1]]
    assert source.read_row_group(1).to_pylist() == [rows[0]]
    for batch_rows in (0, -1):
        with pytest.raises(ValueError, match="batch_rows must be positive"):
            prepare_graphs(
                request,
                output=tmp_path / "bad-batch",
                version=SIMULATOR_RDKIT,
                batch_rows=batch_rows,
            )
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    empty = tmp_path / "empty"
    prepare_graphs(request, output=empty, version=SIMULATOR_RDKIT, batch_rows=2)
    assert parquet.read_table(empty / "graphs.parquet").to_pylist() == []
    manifest = json_loads((empty / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 0
    assert manifest["valid_rows"] == 0


def test_actual_bonded_ring_graph_has_complete_features_and_ring_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ring_size(size: int) -> bool:
        return size == 3

    atom = _atom(degree=2, hydrogens=1, ring=True, ring_size_filled=3)
    bonds = [
        SimpleNamespace(
            GetBeginAtomIdx=lambda left=left: left,
            GetEndAtomIdx=lambda right=right: right,
            GetBondTypeAsDouble=lambda: 1.0,
            GetBondType=lambda: "SINGLE",
            GetIsConjugated=lambda: False,
            IsInRing=lambda: True,
            IsInRingSize=ring_size,
            GetStereo=lambda: "STEREONONE",
        )
        for left, right in ((0, 1), (1, 2), (2, 0))
    ]
    molecule = SimpleNamespace(
        GetAtoms=lambda: [atom] * 3,
        GetNumAtoms=lambda: 3,
        GetBonds=lambda: bonds,
    )

    def parse(value: str) -> SimpleNamespace:
        del value
        return molecule

    def canonical(value: object) -> str:
        del value
        return "N1NN1"

    def unchanged(value: SimpleNamespace) -> SimpleNamespace:
        return value

    def rings(value: object) -> list[list[int]]:
        del value
        return [[0, 1, 2]]

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=unchanged,
        GetSymmSSSR=rings,
    )

    def unavailable(value: object) -> None:
        del value
        raise RuntimeError("missing Gasteiger parameters")

    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    monkeypatch.setattr(atomic_data, "ComputeGasteigerCharges", unavailable)
    row = atomic_data._graph_row("requested", raw=False)
    node = [0.0] * 74
    for index in (1, 21, 28, 34, 38, 40, 46, 47, 50, 52, 54, 56, 58, 59, 61, 68):
        node[index] = 1.0
    bond = [0.0] * 28
    for index in (0, 5, 8, 9, 12, 14, 16, 18, 20, 21, 23):
        bond[index] = 1.0
    assert row == {
        "input_smiles": "requested",
        "smiles": "N1NN1",
        "valid": True,
        "features": [node] * 3,
        "bonds": [[0, 1], [1, 2], [2, 0]],
        "bond_features": [bond] * 3,
        "weights": [1.0] * 3,
        "degrees": [2] * 3,
    }


@pytest.mark.parametrize("case", ["unsupported", "empty", "large", "reparse"])
def test_actual_graph_admission_and_failed_canonical_reparse(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _atom(number=999 if case == "unsupported" else 7)
    molecule = SimpleNamespace(
        GetAtoms=lambda: [atom],
        GetNumAtoms=lambda: 0 if case == "empty" else (101 if case == "large" else 2),
    )

    def parse(value: str) -> SimpleNamespace | None:
        return molecule if value == "input" else None

    def canonical(value: object) -> str:
        del value
        return "canonical"

    def unchanged(value: SimpleNamespace) -> SimpleNamespace:
        return value

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=unchanged,
    )
    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    if case == "reparse":
        with pytest.raises(ValueError, match="Expected parsed is not None"):
            atomic_data._graph_row("input", raw=False)
    else:
        assert atomic_data._graph_row("input", raw=False) == {
            "input_smiles": "input",
            "smiles": "",
            "valid": False,
            "features": [],
            "bonds": [],
            "bond_features": [],
            "weights": [],
            "degrees": [],
        }


def test_raw_graphs_keep_unsupported_elements_and_skip_the_canonical_reparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _atom(number=999)
    molecule = SimpleNamespace(
        GetAtoms=lambda: [atom],
        GetBonds=list,
        GetNumAtoms=lambda: 150,
    )

    def parse(value: str) -> SimpleNamespace | None:
        return molecule if value == "input" else None

    def canonical(value: object) -> str:
        del value
        return "canonical"

    def unchanged(value: SimpleNamespace) -> SimpleNamespace:
        return value

    def rings(value: object) -> list[list[int]]:
        del value
        return []

    def partial_charges(value: object) -> None:
        del value

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=unchanged,
        GetSymmSSSR=rings,
    )
    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    monkeypatch.setattr(atomic_data, "ComputeGasteigerCharges", partial_charges)
    row = atomic_data._graph_row("input", raw=True)
    assert row["valid"] is True
    assert row["smiles"] == "canonical"
    assert len(row["features"]) == 1
    assert atomic_data._graph_row("input", raw=False)["valid"] is False


def test_graph_preparation_rejects_the_main_engine_chemistry_version(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"requires RDKit 2025\.03\.6"):
        prepare_graphs(
            tmp_path / "input.parquet",
            output=tmp_path / "output",
            version=IDENTITY_RDKIT,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
