"""Replay GLACIER physical rows and writer provenance with a fake SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import glacier_data


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import AtomicRow, GlacierRow


_CWD: Final = Path(__file__).resolve().parent


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[AtomicRow, AtomicRow]:
    atoms = [
        types.SimpleNamespace(
            GetSymbol=lambda: "C",
            GetIsotope=lambda: 13,
            GetNumImplicitHs=lambda: 0,
            GetNumExplicitHs=lambda: 2,
        ),
        types.SimpleNamespace(
            GetSymbol=lambda: "N",
            GetIsotope=lambda: 0,
            GetNumImplicitHs=lambda: 3,
            GetNumExplicitHs=lambda: 0,
        ),
    ]
    molecule = types.SimpleNamespace(GetAtoms=lambda: atoms)
    masses = {"C": 12.0, "N": 14.0, "H": 1.0}

    def most_common_mass(element: str) -> float:
        return masses.get(element, 20.0)

    def isotope_mass(element: str, isotope: int) -> float:
        del element, isotope
        return 13.0

    periodic = types.SimpleNamespace(
        GetMostCommonIsotopeMass=most_common_mass,
        GetMassForIsotope=isotope_mass,
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return molecule if value == "canonical" else None

    def kekulize(value: object, clear: bool) -> None:
        del value, clear

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        GetPeriodicTable=lambda: periodic,
        Kekulize=kekulize,
    )

    def context(version: str) -> types.SimpleNamespace:
        assert version == SIMULATOR_RDKIT
        return types.SimpleNamespace(chem=chemistry)

    def formula_string(value: object) -> str:
        del value
        return "CH5N"

    monkeypatch.setattr(glacier_data, "CalcMolFormula", formula_string)
    monkeypatch.setattr(glacier_data, "MolecularContext", context)
    valid: AtomicRow = {
        "input_smiles": "requested",
        "smiles": "canonical",
        "valid": True,
        "features": [],
        "bonds": [],
        "bond_features": [],
        "weights": [],
        "degrees": [],
    }
    invalid: AtomicRow = {
        **valid,
        "input_smiles": "invalid",
        "smiles": "",
        "valid": False,
    }
    return valid, invalid


def _expected() -> tuple[GlacierRow, GlacierRow]:
    formula = [0.0] * 18
    formula[0] = formula[1] = 1.0
    formula[7] = 5.0
    atom_formulas = [[0.0] * 18 for _ in range(2)]
    atom_formulas[0][0] = atom_formulas[1][1] = 1.0
    atom_formulas[0][7], atom_formulas[1][7] = 2.0, 3.0
    return (
        {
            "input_smiles": "requested",
            "smiles": "canonical",
            "valid": True,
            "formula": formula,
            "atom_formulas": atom_formulas,
            "hydrogens": [2, 3],
            "atom_masses": [13.0, 14.0],
            "hydrogenated_masses": [15.0, 17.0],
            "total_hydrogens": 5,
        },
        {
            "input_smiles": "invalid",
            "smiles": "",
            "valid": False,
            "formula": [],
            "atom_formulas": [],
            "hydrogens": [],
            "atom_masses": [],
            "hydrogenated_masses": [],
            "total_hydrogens": 0,
        },
    )


def test_physical_rows_keep_isotopes_hydrogens_and_rejected_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid, invalid = _fixture(monkeypatch)
    expected, rejected = _expected()
    physical = glacier_data.GlacierChemistry(SIMULATOR_RDKIT)
    assert physical.row(valid) == expected
    assert physical.row(invalid) == rejected
    with pytest.raises(ValueError, match="Expected molecule is not None"):
        physical.row({**valid, "smiles": "unparsable"})


def test_writer_streams_groups_masses_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid, invalid = _fixture(monkeypatch)
    expected, rejected = _expected()

    def graphs(
        input_path: Path,
        *,
        output: Path,
        version: str,
        batch_rows: int,
    ) -> None:
        del input_path, version, batch_rows
        output.mkdir()
        source = Table.from_pylist([valid, invalid])
        with parquet.ParquetWriter(output / "graphs.parquet", source.schema) as writer:
            writer.write_table(source.slice(0, 1))
            writer.write_table(source.slice(1, 1))
        _ = (output / "manifest.json").write_text('{"format_version": 1}\n')

    monkeypatch.setattr(glacier_data, "prepare_graphs", graphs)
    request, output = tmp_path / "input.parquet", tmp_path / "glacier"
    parquet.write_table(
        Table.from_pydict({"smiles": ["requested", "invalid"]}),
        request,
    )
    glacier_data.prepare_glacier_graphs(
        request,
        output=output,
        version=SIMULATOR_RDKIT,
        batch_rows=1,
    )
    source = parquet.ParquetFile(output / "physical.parquet")
    assert source.num_row_groups == 2
    assert source.read().to_pylist() == [expected, rejected]
    np.testing.assert_array_equal(
        cast("NDArray[np.float64]", np.load(output / "element_masses.npy")),
        np.array([12.0, 14.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0] + [20.0] * 10)[
            :,
            None,
        ],
    )
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "rows", "valid_rows")
    } == {
        "format_version": 1,
        "rdkit": SIMULATOR_RDKIT,
        "rows": 2,
        "valid_rows": 1,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert (
        manifest["graph_manifest_sha256"]
        == hashlib.sha256((output / "graphs/manifest.json").read_bytes()).hexdigest()
    )
    assert from_plain(manifest["generator"], dict[str, str]) == {
        name: hashlib.sha256((_CWD / name).read_bytes()).hexdigest()
        for name in ("glacier_data.py", "graph_properties.py", "rdkit_backend.py")
    }
    for version, batch_rows in ((IDENTITY_RDKIT, 1), (SIMULATOR_RDKIT, 0)):
        with pytest.raises(ValueError, match=r"requires|positive"):
            glacier_data.prepare_glacier_graphs(
                request,
                output=tmp_path / "bad",
                version=version,
                batch_rows=batch_rows,
            )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
