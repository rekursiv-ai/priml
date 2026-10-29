"""Checks for the pubchem_data module."""

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
from priml.baselines.casmi2026.scripts import pubchem_data
from priml.baselines.casmi2026.scripts.pubchem_data import PubChemChemistry


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray


_SCRIPTS_DIR = _CWD


def test_pubchem_preparation_requires_the_exact_scientific_release() -> None:
    with pytest.raises(
        ValueError,
        match=r"^PubChem preparation requires RDKit 2026\.03\.3\.$",
    ):
        _ = PubChemChemistry(SIMULATOR_RDKIT)


def _ethanol_bits() -> NDArray[np.uint8]:
    """Pack ethanol's raw bits 0, 4096, 16384 and 16550 into 2069 bytes."""
    packed = np.zeros(2069, np.uint8)
    packed[[0, 512, 2048, 2068]] = [128, 128, 128, 2]
    return packed


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_data, "Chemistry", ObservedChemistry)
    monkeypatch.setattr(
        pubchem_data,
        "GenMACCSKeys",
        ObservedChemistry(IDENTITY_RDKIT).maccs_keys,
    )


def test_complete_primitive_records_and_independent_sdk_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    chemistry = PubChemChemistry(IDENTITY_RDKIT)
    parsed, partial, valid, full, key = chemistry.record("CCO")
    assert (parsed, valid, key) == (True, True, "LFQSCWFLJHTTHZ")
    assert partial.dtype == full.dtype == np.uint8
    np.testing.assert_array_equal(partial, _ethanol_bits()[:512])
    np.testing.assert_array_equal(full, _ethanol_bits())
    # The last bit of every generator block lands in that block's last byte.
    _, partial, _, full, key = chemistry.record("C[N+](C)(C)C")
    assert key == "QEMXHQIAXOOASZ"
    assert np.flatnonzero(partial).tolist() == [511]
    assert np.flatnonzero(full).tolist() == [511, 1023, 1279, 1535, 1791, 2047]
    np.testing.assert_array_equal(np.unique(full[full > 0]), [1])
    for value, expected in (
        ("invalid", (False, False, None)),
        ("", (True, True, None)),
    ):
        parsed, partial, valid, full, key = chemistry.record(value)
        assert (parsed, valid, key) == expected
        np.testing.assert_array_equal(partial, np.zeros(512, np.uint8))
        np.testing.assert_array_equal(full, np.zeros(2069, np.uint8))
    parsed, partial, valid, full, key = chemistry.record("raw_error")
    assert (parsed, valid, key) == (True, False, "LFQSCWFLJHTTHZ")
    np.testing.assert_array_equal(partial, _ethanol_bits()[:512])
    np.testing.assert_array_equal(full, np.zeros(2069, np.uint8))
    parsed, partial, valid, full, key = chemistry.record("key_error")
    assert (parsed, valid, key) == (True, True, None)
    np.testing.assert_array_equal(full, _ethanol_bits())


def test_writer_retains_duplicate_rows_full_layouts_and_empty_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    source, destination = tmp_path / "input.parquet", tmp_path / "nested" / "prepared"
    smiles = ["CCO"] * 3 + ["invalid", "", "raw_error", "key_error"]
    parquet.write_table(
        Table.from_pydict(
            {"metadata": [42] * len(smiles), "normalized_smiles": smiles},
        ),
        source,
    )
    pubchem_data.prepare_pubchem(
        source,
        output=destination,
        version=IDENTITY_RDKIT,
        batch_rows=4,
    )
    metadata = parquet.ParquetFile(destination / "molecules.parquet")
    assert metadata.num_row_groups == 2
    rows = metadata.read().to_pylist()
    assert [row["smiles"] for row in rows] == smiles
    assert [row["parsed"] for row in rows] == [True] * 3 + [False] + [True] * 3
    assert [row["raw_valid"] for row in rows] == (
        [True] * 3 + [False, True, False, True]
    )
    assert [row["key"] for row in rows] == (
        ["LFQSCWFLJHTTHZ"] * 3 + [None, None, "LFQSCWFLJHTTHZ", None]
    )
    ecfp4 = np.load(destination / "ecfp4.npy")
    raw_fp = np.load(destination / "raw_fp.npy")
    assert ecfp4.shape == (7, 512)
    assert raw_fp.shape == (7, 2069)
    for index in (0, 1, 2, 6):
        np.testing.assert_array_equal(raw_fp[index], _ethanol_bits())
    for index in (0, 1, 2, 5, 6):
        np.testing.assert_array_equal(ecfp4[index], _ethanol_bits()[:512])
    for index in (3, 4, 5):
        np.testing.assert_array_equal(raw_fp[index], np.zeros(2069, np.uint8))
    manifest = json_loads(
        (destination / "manifest.json").read_text(),
        dict[str, object],
    )
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "rows", "raw_bits", "ecfp4_bits")
    } == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": len(smiles),
        "raw_bits": 16_551,
        "ecfp4_bits": 4096,
    }
    assert manifest["numpy"] == np.__version__
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "pubchem_data.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
        )
    }
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        pubchem_data.prepare_pubchem(
            source,
            output=tmp_path / "none",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    with pytest.raises(FileExistsError):
        pubchem_data.prepare_pubchem(source, output=destination, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        pubchem_data.prepare_pubchem(source, output=source, version=IDENTITY_RDKIT)
    for index, values in enumerate((list[str](), ["invalid"])):
        parquet.write_table(Table.from_pydict({"smiles": values}), source)
        output = tmp_path / f"empty-{index}"
        pubchem_data.prepare_pubchem(source, output=output, version=IDENTITY_RDKIT)
        assert parquet.read_table(output / "molecules.parquet").num_rows == len(values)
        assert np.load(output / "ecfp4.npy").shape == (len(values), 512)
        assert np.load(output / "raw_fp.npy").shape == (len(values), 2069)


@pytest.mark.parametrize("field", ["partial", "key"])
def test_partial_radius_and_full_upstream_key_are_independent_contracts(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = ObservedChemistry(IDENTITY_RDKIT)
    observed.bits["CCO"][4097] = 1

    def chemistry(version: str) -> ObservedChemistry:
        assert version == IDENTITY_RDKIT
        return observed

    def upstream_key(value: str) -> str:
        assert value == "CCO"
        return "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    observed.chem.MolToInchiKey = upstream_key
    monkeypatch.setattr(pubchem_data, "Chemistry", chemistry)
    monkeypatch.setattr(pubchem_data, "GenMACCSKeys", observed.maccs_keys)
    prepared = PubChemChemistry(IDENTITY_RDKIT)
    _, partial, _, full, key = prepared.record("CCO")
    if field == "partial":
        np.testing.assert_array_equal(partial, _ethanol_bits()[:512])
        assert full[512] == 128 + 64
    else:
        assert key == "LFQSCWFLJHTTHZ"


class ObservedChemistry:
    """Replay literal parser, fingerprint and key outcomes through the SDK adapter."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.bits = {
            value: np.zeros(16_551, np.uint8)
            for value in ("CCO", "C[N+](C)(C)C", "", "raw_error", "key_error")
        }
        for value, positions in (
            ("CCO", [0, 4096, 16_384, 16_550]),
            ("C[N+](C)(C)C", [4095, 8191, 10_239, 12_287, 14_335, 16_383]),
            ("raw_error", [0]),
            ("key_error", [0, 4096, 16_384, 16_550]),
        ):
            self.bits[value][positions] = 1
        self.keys = {
            "CCO": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "raw_error": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "C[N+](C)(C)C": "QEMXHQIAXOOASZ-UHFFFAOYSA-N",
            "": "",
        }
        self.chem = types.SimpleNamespace(
            MolFromSmiles=self.parse,
            MolToInchiKey=self.key,
        )
        self.tautomers = types.SimpleNamespace(Canonicalize=self.canonicalize)
        self.generators = tuple(
            types.SimpleNamespace(
                GetFingerprintAsNumPy=ObservedFingerprint(
                    self,
                    index=index,
                ).fingerprint,
            )
            for index in range(6)
        )

    def parse(self, value: str) -> str | None:
        return value if value in self.bits else None

    def canonicalize(self, value: str) -> str:
        return value

    def key(self, value: str) -> str:
        if value == "key_error":
            raise OSError("independent key error")
        return self.keys[value]

    def maccs_keys(self, value: str) -> list[int]:
        if value == "raw_error":
            raise OSError("independent raw-fingerprint error")
        return from_plain(
            cast(object, self.bits[value][16_384:16_551].tolist()),
            list[int],
        )


class ObservedFingerprint:
    def __init__(self, owner: ObservedChemistry, *, index: int) -> None:
        self.owner, self.index = owner, index

    def fingerprint(self, value: str) -> NDArray[np.uint8]:
        bounds = (0, 4096, 8192, 10_240, 12_288, 14_336, 16_384)
        return self.owner.bits[value][bounds[self.index] : bounds[self.index + 1]]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
