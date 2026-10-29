"""Check Fragnet atom observation and the ordered stream writer with a fake SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    molecular_record,
)
from priml.baselines.casmi2026.scripts import fragnet_data


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


def _atom(
    symbol: str,
    *,
    hydrogens: int,
    degree: int,
    charge: int = 0,
    aromatic: bool = False,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        GetSymbol=lambda: symbol,
        GetTotalNumHs=lambda: hydrogens,
        GetDegree=lambda: degree,
        GetFormalCharge=lambda: charge,
        GetIsAromatic=lambda: aromatic,
    )


def _molecule(
    *atoms: types.SimpleNamespace,
    bonds: tuple[types.SimpleNamespace, ...] = (),
) -> types.SimpleNamespace:
    return types.SimpleNamespace(GetAtoms=lambda: atoms, GetBonds=lambda: bonds)


def _bond(begin: int, end: int, *, aromatic: bool, ring: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        GetBeginAtomIdx=lambda: begin,
        GetEndAtomIdx=lambda: end,
        GetIsAromatic=lambda: aromatic,
        IsInRing=lambda: ring,
    )


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    molecules: dict[str, types.SimpleNamespace | None] = {
        "CO": _molecule(
            _atom("C", hydrogens=3, degree=1),
            _atom("O", hydrogens=1, degree=1),
            bonds=(_bond(0, 1, aromatic=False, ring=False),),
        ),
        "[Na+]": _molecule(_atom("Na", hydrogens=0, degree=0, charge=1)),
        "unsupported": _molecule(_atom("Xx", hydrogens=0, degree=0)),
        "invalid": None,
    }

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(
            chem=types.SimpleNamespace(MolFromSmiles=molecules.__getitem__),
        )

    monkeypatch.setattr(fragnet_data, "MolecularContext", context)


def test_atom_projection_retains_every_observed_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    chemistry = fragnet_data.FragnetChemistry(IDENTITY_RDKIT)
    assert chemistry.atoms("invalid") is None
    atoms = chemistry.atoms("CO")
    assert atoms is not None
    assert atoms.symbols == ("C", "O")
    assert atoms.hydrogens.tolist() == [3, 1]
    assert atoms.degree.tolist() == [1, 1]
    assert atoms.charge.tolist() == [0, 0]
    assert atoms.aromatic.tolist() == [0, 0]
    assert atoms.bonds.tolist() == [[0, 1]]
    assert atoms.bond_aromatic.tolist() == [False]
    assert atoms.bond_ring.tolist() == [False]
    charged = chemistry.atoms("[Na+]")
    assert charged is not None
    assert charged.charge.tolist() == [1]
    assert charged.bonds.shape == (0, 2)
    assert charged.bond_ring.shape == (0,)
    with pytest.raises(
        ValueError,
        match=r"^Fragnet preparation requires RDKit 2026\.03\.3\.$",
    ):
        _ = fragnet_data.FragnetChemistry(SIMULATOR_RDKIT)


def test_writer_retains_duplicate_invalid_and_unsupported_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "nested" / "output"
    smiles = ["CO", "CO", "invalid", "unsupported", "[Na+]"]
    parquet.write_table(
        Table.from_pydict({"smiles": smiles, "unrelated": ["unused"] * len(smiles)}),
        request,
    )
    fragnet_data.prepare_fragnet(
        request,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    metadata = parquet.ParquetFile(output / "molecules.parquet")
    assert metadata.num_row_groups == 2
    rows = metadata.read().to_pylist()
    assert [row["smiles"] for row in rows] == smiles
    assert [(row["parsed"], row["valid"]) for row in rows] == [
        (True, True),
        (True, True),
        (False, False),
        (True, False),
        (True, True),
    ]
    assert rows[0]["symbols"] == ["C", "O"]
    assert rows[0]["hydrogens"] == [3, 1]
    assert rows[0]["bonds"] == [[0, 1]]
    assert rows[0]["bond_aromatic"] == [False]
    assert rows[2]["symbols"] == []
    assert rows[2]["bonds"] == []
    assert rows[3]["symbols"] == ["Xx"]
    carbon_monoxide = fragnet_data.FragnetChemistry(IDENTITY_RDKIT).atoms("CO")
    record = molecular_record(carbon_monoxide, masses=ELEMENT_MASSES)
    assert record is not None
    sodium = molecular_record(
        fragnet_data.FragnetChemistry(IDENTITY_RDKIT).atoms("[Na+]"),
        masses=ELEMENT_MASSES,
    )
    assert sodium is not None
    mass = cast("NDArray[np.float64]", np.load(output / "mass.npy"))
    np.testing.assert_array_equal(
        mass,
        [record["mass"], record["mass"], np.nan, np.nan, sodium["mass"]],
    )
    assert np.load(output / "atom_off.npy").tolist() == [0, 2, 4, 4, 4, 5]
    assert np.load(output / "bond_off.npy").tolist() == [0, 1, 2, 2, 2, 2]
    for name in ("amass", "kind", "el", "nh", "el10", "bonds", "bring"):
        np.testing.assert_array_equal(
            np.load(output / f"{name}.npy"),
            np.concatenate([record[name], record[name], sodium[name]]),
        )
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {name: manifest[name] for name in ("format_version", "rdkit", "rows")} == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": 5,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _CWD / "fragnet_data.py",
            _CWD / "rdkit_backend.py",
            _CWD.parent / "chemistry.py",
            _CWD.parent / "fragnet.py",
        )
    }


def test_writer_rejects_bad_batches_existing_outputs_and_aliased_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "output"
    parquet.write_table(Table.from_pydict({"smiles": ["CO"]}), request)
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        fragnet_data.prepare_fragnet(
            request,
            output=output,
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    fragnet_data.prepare_fragnet(request, output=output, version=IDENTITY_RDKIT)
    with pytest.raises(FileExistsError):
        fragnet_data.prepare_fragnet(request, output=output, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        fragnet_data.prepare_fragnet(request, output=request, version=IDENTITY_RDKIT)


@pytest.mark.parametrize("values", [[], ["invalid"]])
def test_writer_saves_empty_ragged_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: list[str],
) -> None:
    _install(monkeypatch)
    source, destination = tmp_path / "input.parquet", tmp_path / "empty"
    parquet.write_table(Table.from_pydict({"smiles": values}), source)
    fragnet_data.prepare_fragnet(source, output=destination, version=IDENTITY_RDKIT)
    assert parquet.read_table(destination / "molecules.parquet").num_rows == len(values)
    assert np.load(destination / "bonds.npy").shape == (0, 2)
    assert np.load(destination / "amass.npy").shape == (0,)
    assert np.load(destination / "atom_off.npy").tolist() == [0] * (len(values) + 1)
    assert np.load(destination / "bond_off.npy").tolist() == [0] * (len(values) + 1)
    assert np.load(destination / "mass.npy").shape == (len(values),)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
