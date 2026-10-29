"""Checks for the identity_data module."""

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
from priml.baselines.casmi2026.scripts import identity_data


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import IdentityRow


_SCRIPTS: Final = _CWD
_KEY: Final = "ABCDEFGHIJKLMNOPQRS"
_ATOMS: Final = {
    "CCO": ("C", "C", "O", "H"),
    "CCO.[Na+]": ("C", "C", "O", "Na", "H"),
    "flaky": ("C",),
    "no-key": ("C",),
    "tautomer-error": ("C",),
    "properties-error": ("N", "N"),
    "formula-error": ("C",),
}


class _Atom:
    def __init__(self, symbol: str) -> None:
        self.GetSymbol = lambda: symbol


class _Molecule:
    def __init__(self, text: str) -> None:
        self.text = text
        self.stereo = True
        self.GetAtoms = lambda: [_Atom(symbol) for symbol in _ATOMS[text]]
        self.GetNumHeavyAtoms = lambda: sum(s != "H" for s in _ATOMS[text])


class _Primitives:
    """Replay the scientific SDK surface that IdentityChemistry composes."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.parses = dict.fromkeys(_ATOMS, 0)
        self.chemistry = types.SimpleNamespace(
            standardize=self.standardize,
            chem=types.SimpleNamespace(
                MolFromSmiles=self.parse,
                RemoveStereochemistry=self.remove_stereo,
                MolToInchiKey=self.inchi_key,
            ),
            tautomers=types.SimpleNamespace(Canonicalize=self.canonicalize),
        )

    def standardize(self, text: str) -> str | None:
        if text == "standard-error":
            raise OSError("standardization failure")
        if text not in _ATOMS:
            return None
        return text.partition(".")[0]

    def parse(self, text: str) -> _Molecule | None:
        if text not in _ATOMS:
            return None
        self.parses[text] += 1
        # The identity key is requested first; its re-parse for counting fails.
        if text == "flaky" and self.parses[text] == 2:
            return None
        return _Molecule(text)

    def remove_stereo(self, molecule: _Molecule) -> None:
        molecule.stereo = False

    def canonicalize(self, molecule: _Molecule) -> _Molecule:
        assert not molecule.stereo
        if molecule.text == "tautomer-error":
            raise ValueError("canonicalization failure")
        return molecule

    def inchi_key(self, molecule: _Molecule) -> str:
        return "" if molecule.text == "no-key" else _KEY

    def score_key(self, text: str) -> str | None:
        return "key:" + text if text in _ATOMS else None

    def raw_fingerprint(self, text: str) -> NDArray[np.uint8] | None:
        if text == "properties-error":
            return None
        return np.full(2069, len(text), np.uint8)


def _formula(molecule: _Molecule) -> str:
    if molecule.text in {"formula-error", "properties-error"}:
        raise OSError("formula failure")
    return "F:" + molecule.text


def _mass(molecule: _Molecule) -> float:
    return float(len(molecule.text))


def _install(monkeypatch: pytest.MonkeyPatch) -> identity_data.IdentityChemistry:
    monkeypatch.setattr(identity_data, "PubChemChemistry", _Primitives)
    monkeypatch.setattr(identity_data, "CalcMolFormula", _formula)
    monkeypatch.setattr(identity_data, "ExactMolWt", _mass)
    return identity_data.IdentityChemistry(IDENTITY_RDKIT)


def test_scientific_release_is_explicit() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Identity preparation requires RDKit 2026\.03\.3\.$",
    ):
        identity_data.IdentityChemistry(SIMULATOR_RDKIT)


def test_record_keeps_every_decision_for_a_standardized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chemistry = _install(monkeypatch)
    row, fingerprint = chemistry.record("CCO.[Na+]")
    metric = (
        '{"py/tuple":["ABCDEFGHIJKLMN",{"py/tuple":'
        '[{"py/tuple":["C",2]},{"py/tuple":["Na",1]},{"py/tuple":["O",1]}]}]}'
    )
    assert row == {
        "smiles": "CCO.[Na+]",
        "standardized": "CCO",
        "key": "key:CCO.[Na+]",
        "formula": "F:CCO.[Na+]",
        "metric": metric,
        "standard_key": "key:CCO",
        "standard_metric": (
            '{"py/tuple":["ABCDEFGHIJKLMN",{"py/tuple":'
            '[{"py/tuple":["C",2]},{"py/tuple":["O",1]}]}]}'
        ),
        "standard_formula_lookup": "F:CCO",
        "properties_valid": True,
        "standard_formula": "F:CCO",
        "standard_mass": 3.0,
        "standard_heavy": 3,
        "fingerprint_valid": True,
    }
    np.testing.assert_array_equal(fingerprint, np.full(2069, 3, np.uint8))
    assert fingerprint.shape == (2069,)


@pytest.mark.parametrize("text", ["invalid", "standard-error"])
def test_record_without_standardization_has_zero_fingerprint_and_null_metrics(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    chemistry = _install(monkeypatch)
    row, fingerprint = chemistry.record(text)
    assert row["standardized"] is None
    assert row["standard_key"] is None
    assert row["standard_metric"] == "null"
    assert row["metric"] == "null"
    assert row["formula"] == text
    assert row["standard_formula_lookup"] == ""
    assert not row["properties_valid"]
    assert (row["standard_formula"], row["standard_mass"], row["standard_heavy"]) == (
        "",
        0.0,
        0,
    )
    assert not row["fingerprint_valid"]
    assert fingerprint.dtype == np.uint8
    np.testing.assert_array_equal(fingerprint, np.zeros(2069, np.uint8))


def test_properties_and_fingerprint_fail_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chemistry = _install(monkeypatch)
    row, fingerprint = chemistry.record("properties-error")
    assert row["standardized"] == "properties-error"
    assert row["key"] == "key:properties-error"
    assert row["formula"] == "properties-error"
    assert not row["properties_valid"]
    assert not row["fingerprint_valid"]
    np.testing.assert_array_equal(fingerprint, np.zeros(2069, np.uint8))
    assert chemistry.properties("formula-error") is None
    assert chemistry.formula("formula-error") == "formula-error"
    assert chemistry.properties("invalid") is None
    assert chemistry.formula("invalid") == "invalid"
    assert chemistry.properties("CCO") == ("F:CCO", 3.0, 3)
    assert chemistry.formula("CCO") == "F:CCO"


def test_identity_rejects_every_sdk_stop_and_counts_heavy_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chemistry = _install(monkeypatch)
    assert chemistry.identity("CCO") == ("ABCDEFGHIJKLMN", (("C", 2), ("O", 1)))
    assert chemistry.identity("CCO.[Na+]") == (
        "ABCDEFGHIJKLMN",
        (("C", 2), ("Na", 1), ("O", 1)),
    )
    assert chemistry.identity("invalid") is None
    assert chemistry.identity("tautomer-error") is None
    assert chemistry.identity("no-key") is None
    assert chemistry.identity("flaky") is None


def _row(text: str) -> IdentityRow:
    return {
        "smiles": text,
        "standardized": text.upper(),
        "key": "key",
        "formula": "F",
        "metric": "null",
        "standard_key": None,
        "standard_metric": "null",
        "standard_formula_lookup": "",
        "properties_valid": text != "bad",
        "standard_formula": "C",
        "standard_mass": 1.5,
        "standard_heavy": 2,
        "fingerprint_valid": text != "bad",
    }


class _Recorder:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def record(self, smiles: str) -> tuple[IdentityRow, NDArray[np.uint8]]:
        fingerprint = np.full(2069, 0 if smiles == "bad" else len(smiles), np.uint8)
        return _row(smiles), fingerprint


def test_writer_keeps_every_row_duplicate_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity_data, "IdentityChemistry", _Recorder)
    strings = ["a", "bb", "bad", "a", "ccc", "bb", "a"]
    source, output = tmp_path / "input.parquet", tmp_path / "nested" / "prepared"
    parquet.write_table(Table.from_pydict({"smiles": strings}), source)
    identity_data.prepare_identity(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    metadata = parquet.ParquetFile(output / "molecules.parquet")
    assert metadata.num_row_groups == 3
    rows = metadata.read().to_pylist()
    assert [dict(row) for row in rows] == [dict(_row(text)) for text in strings]
    fingerprints = cast("NDArray[np.uint8]", np.load(output / "fingerprints.npy"))
    assert fingerprints.shape == (7, 2069)
    assert fingerprints[:, 0].tolist() == [1, 2, 0, 1, 3, 2, 1]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["raw_bits"] == 16_551
    assert manifest["rows"] == 7
    assert manifest["numpy"] == np.__version__
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS / "identity_data.py",
            _SCRIPTS / "pubchem_data.py",
            _SCRIPTS / "rdkit_backend.py",
        )
    }
    with pytest.raises(FileExistsError):
        identity_data.prepare_identity(source, output=output, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        identity_data.prepare_identity(source, output=source, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        identity_data.prepare_identity(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    empty = tmp_path / "empty"
    identity_data.prepare_identity(source, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "molecules.parquet").num_rows == 0
    assert np.load(empty / "fingerprints.npy").shape == (0, 2069)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
