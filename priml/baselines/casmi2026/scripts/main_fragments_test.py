"""Checks for the main_fragments module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import main_fragments
from priml.baselines.casmi2026.scripts.main_fragments import (
    base_fragment_masses,
    fragment_masses,
    fragment_rows,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_chemistry_preserves_atom_order_and_rejects_unknown_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carbon = types.SimpleNamespace(
        GetIdx=lambda: 1,
        GetSymbol=lambda: "C",
        GetTotalNumHs=lambda: 3,
    )
    oxygen = types.SimpleNamespace(
        GetIdx=lambda: 0,
        GetSymbol=lambda: "O",
        GetTotalNumHs=lambda: 1,
    )
    molecule = types.SimpleNamespace(
        GetNumAtoms=lambda: 2,
        GetAtoms=lambda: (carbon, oxygen),
        GetBonds=lambda: (
            types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        ),
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return None if value == "invalid" else molecule

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(chem=types.SimpleNamespace(MolFromSmiles=parse))

    monkeypatch.setattr(main_fragments, "MolecularContext", context)
    chemistry = main_fragments.FragmentChemistry(IDENTITY_RDKIT)
    graph = chemistry.graph("input")
    assert graph is not None
    np.testing.assert_array_equal(
        graph[0],
        [15.9949146196 + 1.00782503207, 12.0 + 3 * 1.00782503207],
    )
    np.testing.assert_array_equal(graph[1], [[1, 0]])
    assert graph[1].dtype == np.int32
    assert chemistry.graph("invalid") is None
    molecule.GetBonds = lambda: ()
    graph = chemistry.graph("no bonds")
    assert graph is not None
    assert graph[1].shape == (0, 2)
    assert graph[1].dtype == np.int32
    oxygen.GetSymbol = lambda: "unknown"
    assert chemistry.graph("unknown") is None
    with pytest.raises(ValueError, match="requires RDKit"):
        _ = main_fragments.FragmentChemistry(SIMULATOR_RDKIT)


def test_writer_retains_all_rows_bounded_batches_and_empty_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_fragments, "FragmentChemistry", _Graphs)
    request, output = tmp_path / "input.parquet", tmp_path / "output"
    values = ["CC"] * 257 + ["invalid"]
    parquet.write_table(Table.from_pydict({"smiles": values}), request)
    main_fragments.prepare_main_fragments(
        request,
        output=output,
        version=IDENTITY_RDKIT,
    )
    source = parquet.ParquetFile(output / "molecules.parquet")
    assert source.num_row_groups == 2
    rows = source.read().to_pylist()
    assert [row["smiles"] for row in rows] == values
    assert [row["valid"] for row in rows] == [True] * 257 + [False]
    assert rows[0]["weights"] == [1.0, 2.0]
    assert rows[0]["bonds"] == [[0, 1]]
    assert rows[-1]["weights"] == []
    assert rows[-1]["bonds"] == []
    for prefix in ("frag", "f1"):
        offsets = cast("NDArray[np.int64]", np.load(output / f"{prefix}_off.npy"))
        masses = cast("NDArray[np.float32]", np.load(output / f"{prefix}_mass.npy"))
        np.testing.assert_array_equal(offsets[:3], [0, 3, 6])
        assert offsets[257] == offsets[258] == 3 * 257
        assert masses.dtype == np.float32
        np.testing.assert_array_equal(masses, [1, 2, 3] * 257)
    base = cast("NDArray[np.float64]", np.load(output / "base_mass.npy"))
    assert base.dtype == np.float64
    np.testing.assert_array_equal(base, [1, 2, 3] * 257)
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 258
    assert manifest["depth_two_masses"] == manifest["depth_one_masses"] == 3 * 257
    assert manifest["base_masses"] == 3 * 257
    assert manifest["capacity"] == 200_000
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["numpy"] == np.__version__
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "main_fragments.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    assert len(from_plain(manifest["inputs"], dict[str, str])) == 1
    with pytest.raises(FileExistsError):
        main_fragments.prepare_main_fragments(
            request,
            output=output,
            version=IDENTITY_RDKIT,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), request)
    empty = tmp_path / "empty"
    main_fragments.prepare_main_fragments(request, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "molecules.parquet").to_pylist() == []
    for prefix in ("frag", "f1"):
        offsets = cast("NDArray[np.int64]", np.load(empty / f"{prefix}_off.npy"))
        masses = cast("NDArray[np.float32]", np.load(empty / f"{prefix}_mass.npy"))
        np.testing.assert_array_equal(offsets, [0])
        assert masses.shape == (0,)
    base = cast(
        "NDArray[np.float64]",
        np.load(empty / "base_mass.npy", allow_pickle=False),
    )
    assert base.shape == (0,)
    assert base.dtype == np.float64


def test_rounded_fragment_rows_and_cap_encounter_order() -> None:
    # 1 + 2**-10 is exact in binary and rounds up to 1.001 Da.
    weights = np.array([1.0009765625, 2.0, 4.0])
    chain = np.array([[0, 1], [1, 2]], dtype=np.int32)
    deep = [1.001, 2.0, 3.001, 4.0, 6.0, 7.001]
    shallow = [1.001, 3.001, 4.0, 6.0, 7.001]
    for depth, capacity, expected in (
        (2, 200_000, deep),
        (1, 200_000, shallow),
        (0, 200_000, shallow),
        (2, 3, [1.001, 6.0, 7.001]),
        (2, 2, [1.001, 7.001]),
        (2, 1, [7.001]),
    ):
        masses = fragment_masses(weights, bonds=chain, depth=depth, capacity=capacity)
        assert masses.tolist() == expected
    ring = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
    assert fragment_masses(weights, bonds=ring, depth=1).tolist() == [7.001]
    assert fragment_masses(weights, bonds=ring).tolist() == [
        1.001,
        2.0,
        3.001,
        4.0,
        5.001,
        6.0,
        7.001,
    ]
    base, two, one = fragment_rows((weights, chain))
    assert base.tolist() == deep
    assert two.dtype == one.dtype == np.float32
    assert two.tolist() == np.array(deep, np.float32).tolist()
    assert one.tolist() == np.array(shallow, np.float32).tolist()
    lone = (np.array([1.0009765625]), np.zeros((0, 2), np.int32))
    base, two, one = fragment_rows(lone)
    assert base.tolist() == two.tolist() == one.tolist() == [1.0009765625]
    assert two is not one
    assert base_fragment_masses(lone).tolist() == [1.0009765625]
    base, two, one = fragment_rows(None)
    assert (base.dtype, two.dtype, one.dtype) == (np.float64, np.float32, np.float32)
    assert base.shape == two.shape == one.shape == (0,)


def test_fragment_graph_preserves_weights_through_nonroot_bond_endpoints() -> None:
    masses = fragment_masses(
        np.array([1.0, 2.0, 4.0, 8.0]),
        bonds=np.array([[3, 2], [2, 1], [0, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(masses, [1, 2, 4, 6, 8, 9, 12, 13, 14, 15])


class _Graphs:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def graph(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]] | None:
        return (
            None
            if smiles == "invalid"
            else (np.array([1.0, 2.0]), np.array([[0, 1]], dtype=np.int32))
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
