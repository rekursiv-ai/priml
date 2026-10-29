"""Checks for the analog_data module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import hashlib

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    AnalogTable,
)
from priml.baselines.casmi2026.scripts import analog_data


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rdkit import Chem


_SCRIPTS_DIR: Final = _CWD


def test_actual_element_core_ring_and_formula_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    chemistry = analog_data.AnalogChemistry(IDENTITY_RDKIT)
    row = chemistry.record("input", formula="C2H5NO")
    hashes = [
        analog_data._identity(prefix + name)
        for prefix, name in (("E:", "core"), ("G:", "generic"), ("R:", "core"))
    ]
    np.testing.assert_array_equal(row.hashes, [*hashes, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row.numbers, [1, 1, 3, 1, 1, 0, 3, 4])
    np.testing.assert_array_equal(row.formula, [2, 5, 1, 1, 0, 0, 0, 0, 0, 0])
    assert (row.core_smiles, row.element_smiles, row.query_smiles) == (
        "core",
        "full",
        "core",
    )
    assert row.graph is not None
    assert row.core is not None
    np.testing.assert_array_equal(row.graph.elements, [6, 7, 8, 6])
    np.testing.assert_array_equal(row.graph.bonds, [[0, 1], [1, 2], [0, 2], [2, 3]])
    np.testing.assert_array_equal(row.core.elements, [6, 7, 8])
    np.testing.assert_array_equal(row.core.bonds, [[0, 1], [1, 2], [0, 2]])
    np.testing.assert_array_equal(row.core_mask, [True, True, True, False])
    chain = chemistry.record("chain", formula=None)
    np.testing.assert_array_equal(chain.hashes, np.zeros(8, dtype=np.int64))
    np.testing.assert_array_equal(chain.numbers, [0, 0, 0, 0, 0, 0, 0, 2])
    np.testing.assert_array_equal(chain.core_mask, [False, False])
    assert chain.core is None
    assert chain.query_smiles == chain.core_smiles == ""
    for value in ("", "invalid", "empty"):
        rejected = chemistry.record(value, formula=None)
        np.testing.assert_array_equal(rejected.hashes, np.zeros(8))
        np.testing.assert_array_equal(rejected.numbers, np.zeros(8))
        np.testing.assert_array_equal(rejected.formula, np.full(10, -1))
        assert rejected.graph is None
        assert rejected.core is None
        assert rejected.core_mask.size == 0
    assert chemistry._numeric("invalid") is None
    assert analog_data._core([[1], [0]]) == []
    assert analog_data._components([[1], [0], []], active={0, 1, 2}) == [[0, 1], [2]]
    with pytest.raises(ValueError, match=r"requires RDKit 2026\.03\.3"):
        analog_data.AnalogChemistry(SIMULATOR_RDKIT)


def test_a_repeated_bond_is_kept_once_in_the_element_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    row = analog_data.AnalogChemistry(IDENTITY_RDKIT).record("repeated", formula=None)
    np.testing.assert_array_equal(row.numbers, [0, 0, 0, 0, 0, 0, 0, 2])


def test_an_unparsable_element_graph_leaves_graphs_and_mask_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    chemistry = analog_data.AnalogChemistry(IDENTITY_RDKIT)

    def unparsable(smiles: str) -> None:
        del smiles

    monkeypatch.setattr(chemistry, "_numeric", unparsable)
    row = chemistry.record("input", formula=None)
    assert row.element_smiles == "full"
    assert row.graph is None
    assert row.core is None
    assert row.core_mask.size == 0


def test_analog_writer_retains_all_rows_numeric_graphs_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "analog"
    values = ["input", "invalid", "input", "chain"]
    formulas = ["C2H5NO", None, "C2H5NO", "CH3N"]
    parquet.write_table(
        Table.from_pydict({"smiles": values, "formula": formulas}),
        request,
    )
    analog_data.prepare_analog(request, output=output, version=IDENTITY_RDKIT)
    table = AnalogTable(output)
    assert [row["smiles"] for row in table.metadata] == values
    assert [row["formula_input"] for row in table.metadata] == formulas
    assert [row["element_smiles"] for row in table.metadata] == [
        "full",
        "",
        "full",
        "chain",
    ]
    assert [row["core_smiles"] for row in table.metadata] == ["core", "", "core", ""]
    for name, expected in (
        ("graph_elements", [6, 7, 8, 6, 6, 7, 8, 6, 6, 7]),
        ("graph_bonds", [[0, 1], [1, 2], [0, 2], [2, 3]] * 2 + [[0, 1]]),
        ("core_elements", [6, 7, 8] * 2),
        ("core_bonds", [[0, 1], [1, 2], [0, 2]] * 2),
        ("core_mask", [True, True, True, False] * 2 + [False, False]),
        ("graph_atom_off", [0, 4, 4, 8, 10]),
        ("graph_bond_off", [0, 4, 4, 8, 9]),
        ("core_atom_off", [0, 3, 3, 6, 6]),
        ("core_bond_off", [0, 3, 3, 6, 6]),
    ):
        actual = cast("NDArray[np.generic]", np.load(output / f"{name}.npy"))
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        table.numbers,
        [
            [1, 1, 3, 1, 1, 0, 3, 4],
            [0] * 8,
            [1, 1, 3, 1, 1, 0, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 2],
        ],
    )
    np.testing.assert_array_equal(
        table.formula,
        [
            [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
            [-1] * 10,
            [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert table.row(1).graph is None
    assert table.row(1).core is None
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {name: manifest[name] for name in ("format_version", "rdkit", "rows")} == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": 4,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "analog_data.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
            _SCRIPTS_DIR.parent / "chemistry.py",
        )
    }
    with pytest.raises(FileExistsError):
        analog_data.prepare_analog(request, output=output, version=IDENTITY_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [], "formula": []}), request)
    analog_data.prepare_analog(
        request,
        output=tmp_path / "empty",
        version=IDENTITY_RDKIT,
    )
    empty = AnalogTable(tmp_path / "empty")
    assert empty.metadata == []
    assert empty.hashes.shape == (0, 8)
    assert empty.graph.elements.shape == (0,)
    assert empty.graph.bonds.shape == (0, 2)


def _analog_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def discard(value: object) -> None:
        del value

    def atom(index: int, number: int) -> SimpleNamespace:
        return SimpleNamespace(
            GetIdx=lambda: index,
            GetAtomicNum=lambda: number,
            SetNoImplicit=discard,
        )

    def molecule(
        numbers: list[int],
        edges: list[tuple[int, int]],
    ) -> SimpleNamespace:
        atoms = [atom(index, number) for index, number in enumerate(numbers)]
        bonds = [
            SimpleNamespace(
                GetBeginAtomIdx=lambda left=left: left,
                GetEndAtomIdx=lambda right=right: right,
                IsInRing=lambda left=left, right=right: (
                    len(numbers) >= 3 and left < 3 and right < 3
                ),
            )
            for left, right in edges
        ]

        def at(index: int) -> SimpleNamespace:
            return atoms[index]

        def update(*, strict: bool) -> None:
            del strict

        return SimpleNamespace(
            numbers=numbers,
            GetAtoms=lambda: atoms,
            GetBonds=lambda: bonds,
            GetAtomWithIdx=at,
            GetNumAtoms=lambda: len(atoms),
            GetNumBonds=lambda: len(bonds),
            UpdatePropertyCache=update,
        )

    structures = {
        "input": molecule([6, 7, 8, 6, 1], [(0, 1), (1, 2), (2, 0), (2, 3), (0, 4)]),
        "chain": molecule([6, 7], [(0, 1)]),
        "repeated": molecule([6, 7], [(0, 1), (1, 0)]),
        "empty": molecule([], []),
    }

    def editable() -> SimpleNamespace:
        numbers: list[int] = []
        edges: list[tuple[int, int]] = []

        def add_atom(value: Chem.Atom) -> int:
            numbers.append(value.GetAtomicNum())
            return len(numbers) - 1

        def add_bond(left: int, right: int, kind: object) -> None:
            del kind
            edges.append((left, right))

        return SimpleNamespace(
            AddAtom=add_atom,
            AddBond=add_bond,
            GetMol=lambda: molecule(numbers, edges),
        )

    def canonical(value: SimpleNamespace, **kwargs: object) -> str:
        del kwargs
        numbers = from_plain(cast(object, value.numbers), list[int])
        name = (
            "full"
            if len(numbers) == 4
            else (
                "chain"
                if len(numbers) == 2
                else ("generic" if numbers == [6, 6, 6] else "core")
            )
        )
        structures[name] = value
        return name

    def parse(value: str, *, sanitize: bool) -> SimpleNamespace | None:
        del sanitize
        return structures.get(value)

    def construct_atom(number: int) -> SimpleNamespace:
        return atom(0, number)

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        Atom=construct_atom,
        RWMol=editable,
        FastFindRings=discard,
        BondType=SimpleNamespace(SINGLE="single"),
    )

    def context(version: str) -> SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return SimpleNamespace(chem=chemistry)

    monkeypatch.setattr(analog_data, "MolecularContext", context)


def test_graph_hashes_are_seeded_blake2b_and_zero_is_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for text, expected in (
        ("E:C1CCCCC1", 4_101_090_148_000_295_725),
        ("G:C1CCCCC1", 2_963_313_213_660_459_189),
        ("R:C1CCCCC1", 2_587_690_919_976_953_160),
        ("E:C1CCNCC1", 1_399_155_810_780_611_077),
        ("E:C1CCC(CCC2CCCCC2)CC1", 1_073_036_906_936_525_605),
        ("E:C1CC1.C1CC1", 807_823_770_527_924_393),
        (
            "E:C1CC1.C1CCC1.C1CCCC1.C1CCCCC1.C1CCCCCC1.C1CN1.C1CO1",
            3_953_499_957_839_148_272,
        ),
    ):
        assert analog_data._identity(text) == expected

    def reserved(data: bytes, *, digest_size: int) -> SimpleNamespace:
        assert data == b"reserved"
        assert digest_size == 8
        return SimpleNamespace(digest=lambda: bytes(8))

    monkeypatch.setattr(
        analog_data,
        "hashlib",
        SimpleNamespace(blake2b=reserved),
    )
    assert analog_data._identity("reserved") == 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
