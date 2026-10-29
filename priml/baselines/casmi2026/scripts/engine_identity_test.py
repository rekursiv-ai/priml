"""Preserve every engine identity through the primitive and prepared boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import engine_identity


if TYPE_CHECKING:
    from pathlib import Path


_KEYS: dict[str, str | None] = {
    "CCO": "LFQSCWFLJHTTHZ",
    "empty": "",
    "invalid": None,
    "parse-error": None,
    "canonical-error": None,
    "key-error": None,
}


class _Molecule:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles
        self.canonical = False


class Primitives:
    """Supply captured SDK outcomes and explicit exceptional operations."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.chem = types.SimpleNamespace(
            MolFromSmiles=self.parse,
            MolToInchiKey=self.inchi,
        )
        self.tautomers = types.SimpleNamespace(Canonicalize=self.canonicalize)

    def parse(self, smiles: str) -> _Molecule | None:
        if smiles == "parse-error":
            raise ValueError("known parse failure")
        return None if smiles == "invalid" else _Molecule(smiles)

    def canonicalize(self, molecule: _Molecule) -> _Molecule:
        if molecule.smiles == "canonical-error":
            raise RuntimeError("known tautomer failure")
        molecule.canonical = True
        return molecule

    def inchi(self, molecule: _Molecule) -> str:
        assert molecule.canonical
        if molecule.smiles == "key-error":
            raise RuntimeError("known InChI failure")
        key = _KEYS[molecule.smiles]
        assert key is not None
        return key + "-UHFFFAOYSA-N" if key else ""


class RecordedKeys:
    """Return the recorded SDK decision for each declared input."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def key(self, smiles: str) -> str | None:
        return _KEYS[smiles]


def test_engine_identity_sdk_keeps_first14_empty_and_known_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_identity, "Chemistry", Primitives)
    chemistry = engine_identity.EngineKeyChemistry(IDENTITY_RDKIT)
    assert {smiles: chemistry.key(smiles) for smiles in _KEYS} == _KEYS
    with pytest.raises(
        ValueError,
        match=r"^Engine identity preparation requires RDKit 2026\.03\.3\.$",
    ):
        engine_identity.EngineKeyChemistry(SIMULATOR_RDKIT)


def test_writer_preserves_all_rows_duplicates_empty_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_identity, "EngineKeyChemistry", RecordedKeys)
    strings = ["CCO", "empty", "invalid", "CCO"]
    source, output = tmp_path / "request.parquet", tmp_path / "prepared"
    parquet.write_table(Table.from_pydict({"smiles": strings}), source)
    engine_identity.prepare_engine_identity(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    assert parquet.read_table(output / "keys.parquet").to_pylist() == [
        {"smiles": value, "key": _KEYS[value]} for value in strings
    ]
    header = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert header["rows"] == len(strings)
    assert header["rdkit"] == IDENTITY_RDKIT
    assert header["format_version"] == 1
    assert from_plain(header["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert set(from_plain(header["generator"], dict[str, str])) == {
        "engine_identity.py",
        "rdkit_backend.py",
    }
    with pytest.raises(FileExistsError):
        engine_identity.prepare_engine_identity(
            source,
            output=output,
            version=IDENTITY_RDKIT,
        )
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        engine_identity.prepare_engine_identity(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    engine_identity.prepare_engine_identity(
        source,
        output=tmp_path / "empty",
        version=IDENTITY_RDKIT,
    )
    assert parquet.read_table(tmp_path / "empty/keys.parquet").num_rows == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
