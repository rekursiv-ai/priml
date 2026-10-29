"""Checks for the admission module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import admission
from priml.baselines.casmi2026.simulators import SimulatorAdmissionTable


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.casmi2026.custom_types import SimulatorAdmissionRow


class _Decisions:
    def __init__(self, version: str, *, directory: Path) -> None:
        assert version == SIMULATOR_RDKIT
        _ = (directory / "tautomer-transforms.txt").write_text("declared test boundary")

    def row(self, smiles: str) -> SimulatorAdmissionRow:
        accepted = smiles != "invalid"
        return {
            "smiles": smiles,
            "ice_accepted": accepted,
            "ice_reason": "" if accepted else "rdkit parse",
            "glacier_accepted": accepted,
            "glacier_reason": "" if accepted else "rdkit parse",
            "glacier_canonical": "canonical" if accepted else None,
            "glacier_heavy_atoms": 2 if accepted else 0,
        }


def test_admission_writer_keeps_all_duplicate_invalid_and_empty_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admission, "AdmissionChemistry", _Decisions)
    request, output = tmp_path / "request.parquet", tmp_path / "output"
    values = ["valid", "invalid", "valid", "other"]
    parquet.write_table(Table.from_pydict({"smiles": values}), request)
    admission.prepare_admission(
        request,
        output=output,
        version=SIMULATOR_RDKIT,
        batch_rows=3,
    )
    table = SimulatorAdmissionTable(output)
    assert [row["smiles"] for row in table.rows] == values
    assert [row["ice_accepted"] for row in table.rows] == [True, False, True, True]
    assert table.for_smiles("valid")["glacier_heavy_atoms"] == 2
    assert parquet.ParquetFile(output / "admission.parquet").num_row_groups == 2
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["rows"] == 4
    assert manifest["format_version"] == 1
    with pytest.raises(FileExistsError):
        admission.prepare_admission(request, output=output, version=SIMULATOR_RDKIT)
    for version, batch in ((IDENTITY_RDKIT, 3), (SIMULATOR_RDKIT, 0)):
        with pytest.raises(ValueError, match=r"requires|positive"):
            admission.prepare_admission(
                request,
                output=tmp_path / "invalid",
                version=version,
                batch_rows=batch,
            )
    parquet.write_table(Table.from_pydict({"smiles": []}), request)
    admission.prepare_admission(
        request,
        output=tmp_path / "empty",
        version=SIMULATOR_RDKIT,
    )
    assert SimulatorAdmissionTable(tmp_path / "empty").rows == ()


def test_actual_admission_paths_and_canonical_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def molecule(name: str) -> SimpleNamespace:
        symbol = "Xe" if name == "unsupported" else "N"
        size = 101 if name == "large" else (0 if name == "" else 2)
        return SimpleNamespace(
            name=name,
            GetAtoms=lambda: [SimpleNamespace(GetSymbol=lambda: symbol)],
            GetNumAtoms=lambda: size,
            GetNumHeavyAtoms=lambda: size,
        )

    def parse(value: str) -> SimpleNamespace | None:
        if value == "exception":
            raise RuntimeError("scientific failure")
        return None if value in ("invalid", "reparse") else molecule(value)

    def canonical(value: SimpleNamespace) -> str:
        name = from_plain(cast(object, value.name), str)
        return "reparse" if name == "roundtrip" else name

    def inchi(value: str) -> SimpleNamespace | None:
        return None if value == "inchi" else molecule(value)

    def encode_inchi(value: SimpleNamespace) -> str:
        return from_plain(cast(object, value.name), str)

    def unchanged(value: SimpleNamespace) -> SimpleNamespace:
        return value

    def tautomer(value: SimpleNamespace) -> SimpleNamespace:
        name = from_plain(cast(object, value.name), str)
        return molecule("reparse") if name == "tautomer" else value

    def discard(value: object) -> None:
        del value

    chemistry = SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        MolToInchi=encode_inchi,
        MolFromInchi=inchi,
        RemoveStereochemistry=discard,
        RemoveHs=unchanged,
    )

    def context(version: str, *, directory: Path) -> SimpleNamespace:
        del version, directory
        return SimpleNamespace(
            chem=chemistry,
            tautomers=SimpleNamespace(Canonicalize=tautomer),
            elements=("N",),
        )

    monkeypatch.setattr(admission, "IceChemistry", context)
    actual = admission.AdmissionChemistry(SIMULATOR_RDKIT, directory=tmp_path)
    for value, ice, ice_reason, glacier, glacier_reason, heavy in (
        ("accepted", True, "", "accepted", "", 2),
        ("", False, "rdkit parse", None, "empty", 0),
        ("invalid", False, "rdkit parse", None, "rdkit parse", 0),
        (
            "unsupported",
            False,
            "elements ['Xe']",
            None,
            "element outside VALID_ATOM_NUM",
            0,
        ),
        ("large", True, "", None, "> 100 heavy atoms", 0),
        ("roundtrip", False, "inchi round trip", None, "canonical re-parse", 0),
        ("inchi", False, "inchi round trip", "inchi", "", 2),
        ("tautomer", False, "inchi round trip", "tautomer", "", 2),
        (
            "exception",
            False,
            "exception RuntimeError('scientific failure')",
            None,
            "exception RuntimeError('scientific failure')",
            0,
        ),
    ):
        assert actual.row(value) == {
            "smiles": value,
            "ice_accepted": ice,
            "ice_reason": ice_reason,
            "glacier_accepted": glacier is not None,
            "glacier_reason": glacier_reason,
            "glacier_canonical": glacier,
            "glacier_heavy_atoms": heavy,
        }

    def unavailable(value: str) -> None:
        del value
        raise RuntimeError("backend became unavailable")

    chemistry.MolFromSmiles = unavailable
    cached = actual.row("accepted")
    assert cached["ice_accepted"] is False
    assert cached["glacier_canonical"] == "accepted"
    assert cached["glacier_heavy_atoms"] == 2
    assert cached["glacier_reason"] == ""


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
