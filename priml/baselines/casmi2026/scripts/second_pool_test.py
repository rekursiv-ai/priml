"""Checks for the second engine's fingerprint cache and candidate-pool assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import json
import math
import pickle

from pyarrow import Table, parquet
from treekle import json_dumps

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts import second_pool
from priml.baselines.casmi2026.scripts.second_pool import (
    prepare_second_fingerprints,
    prepare_second_pool,
)


# Four-block layout: Morgan2 4096, Morgan3 4096, RDK6 2048, MACCS 167.
_BLOCKS = {0: (0, 4096), 1: (4096, 4096), 3: (8192, 2048)}
_MACCS_START = 10_240
_SELECTED = np.array([0, 5, 4096, 8193, 10_243], dtype=np.int64)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Molecule:
    bits: frozenset[int]
    mass: float
    fails: bool = False


_MOLECULES = {
    "A": _Molecule(bits=frozenset({0, 4096}), mass=100.5),
    "B": _Molecule(bits=frozenset({5, 8193, 10_243}), mass=200.25),
    "boom": _Molecule(bits=frozenset(), mass=0.0, fails=True),
}


def _generator(start: int, size: int) -> SimpleNamespace:
    def fingerprint(molecule: _Molecule) -> np.ndarray:
        if molecule.fails:
            raise ValueError("Fingerprint failed.")
        vector = np.zeros(size, dtype=np.int64)
        for bit in molecule.bits:
            if start <= bit < start + size:
                vector[bit - start] = 1
        return vector

    return SimpleNamespace(GetFingerprintAsNumPy=fingerprint)


_CHEM = SimpleNamespace(MolFromSmiles=_MOLECULES.get)


class _Chemistry:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.chem = _CHEM
        self.generators = {index: _generator(*_BLOCKS[index]) for index in _BLOCKS} | {
            2: _generator(0, 0),
            4: _generator(0, 0),
            5: _generator(0, 0),
        }


def _maccs(molecule: _Molecule) -> list[int]:
    return [int(_MACCS_START + index in molecule.bits) for index in range(167)]


def _exact_mass(molecule: _Molecule) -> float:
    return molecule.mass


@pytest.fixture(autouse=True)
def fake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(second_pool, "Chemistry", _Chemistry)
    monkeypatch.setattr(second_pool, "GenMACCSKeys", _maccs)
    monkeypatch.setattr(second_pool, "ExactMolWt", _exact_mass)


def _bits_file(tmp_path: Path, selected: np.ndarray) -> Path:
    path = tmp_path / "bits.npy"
    np.save(path, selected)
    return path


def _smiles_file(tmp_path: Path, column: str, values: list[str | None]) -> Path:
    path = tmp_path / "requested.parquet"
    parquet.write_table(Table.from_pydict({column: values}), path)
    return path


@pytest.mark.parametrize("column", ["smiles", "normalized_smiles"])
def test_fingerprints_pack_selected_bits_and_mark_failed_rows_invalid(
    tmp_path: Path,
    column: str,
) -> None:
    source = _smiles_file(
        tmp_path,
        column,
        ["A", None, "bad", "B", "A", "boom"],
    )
    bits = _bits_file(tmp_path, _SELECTED)
    output = tmp_path / "out"
    prepare_second_fingerprints(
        source,
        bits=bits,
        output=output,
        version=IDENTITY_RDKIT,
    )
    np.testing.assert_array_equal(
        np.load(output / "fingerprints.npy"),
        [[0b10100000], [0], [0b01011000], [0]],
    )
    np.testing.assert_array_equal(np.load(output / "fp_bits.npy"), _SELECTED)
    rows = parquet.read_table(output / "fingerprints.parquet").to_pylist()
    table = {name: [row[name] for row in rows] for name in ("smiles", "valid", "mass")}
    assert table["smiles"] == ["A", "bad", "B", "boom"]
    assert table["valid"] == [True, False, True, False]
    mass = np.array([row["mass"] for row in rows], dtype=np.float64)
    np.testing.assert_array_equal(mass, [100.5, math.nan, 200.25, math.nan])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows"] == 4
    assert manifest["selected_bits"] == 5
    assert manifest["raw_bits"] == 10_407
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert set(manifest["generator"]) == {
        "second_pool.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    assert set(manifest["inputs"]) == {str(source.resolve()), str(bits.resolve())}


def test_fingerprints_require_the_isolated_release(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=rf"require RDKit {IDENTITY_RDKIT}\."):
        prepare_second_fingerprints(
            _smiles_file(tmp_path, "smiles", ["A"]),
            bits=_bits_file(tmp_path, _SELECTED),
            output=tmp_path / "out",
            version="0.0.0",
        )


@pytest.mark.parametrize(
    "selected",
    [
        np.zeros((2, 3), dtype=np.int64),
        np.array([1, 2], dtype=np.int32),
        np.array([], dtype=np.int64),
        np.array([-1, 2], dtype=np.int64),
        np.array([1, 10_407], dtype=np.int64),
    ],
)
def test_fingerprints_reject_selected_bits_outside_the_layout(
    tmp_path: Path,
    selected: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"^Selected bits require an int64 vector"):
        prepare_second_fingerprints(
            _smiles_file(tmp_path, "smiles", ["A"]),
            bits=_bits_file(tmp_path, selected),
            output=tmp_path / "out",
            version=IDENTITY_RDKIT,
        )
    assert not (tmp_path / "out").exists()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Staged:
    training: Path
    coconut: Path
    bio: Path
    fingerprints: Path


def _stage(tmp_path: Path) -> _Staged:
    training = tmp_path / "train.parquet"
    parquet.write_table(
        Table.from_pydict(
            {
                "inchikey14": ["K1", None, "T1", "T2", "T3", "T1", "T4", "T5"],
                "normalized_smiles": ["x", "y", "t1", "t2", "t3", "t1", "t4", None],
            },
        ),
        training,
    )
    for name, keys, smiles, mass, packed in (
        ("coco", ["K1", "K2"], ["s1", "s2"], [300.0, 100.0], [[1], [2]]),
        ("bio", ["K2", "K3"], ["s2b", "s3"], [100.0, 200.0], [[3], [4]]),
    ):
        directory = tmp_path / name
        directory.mkdir()
        _ = (directory / f"{name}_meta.pkl").write_bytes(
            pickle.dumps({"keys": keys, "smiles": smiles}),
        )
        np.save(directory / f"{name}_mass.npy", np.array(mass))
        np.save(directory / f"{name}_fp.npy", np.array(packed, dtype=np.uint8))
    cache = tmp_path / "fp"
    cache.mkdir()
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["t1", "t2", "t3", "t4"],
                "valid": [True, False, True, True],
                "mass": [150.0, math.nan, 50.0, math.nan],
            },
        ),
        cache / "fingerprints.parquet",
    )
    np.save(
        cache / "fingerprints.npy",
        np.array([[8], [9], [16], [32]], dtype=np.uint8),
    )
    np.save(cache / "fp_bits.npy", _SELECTED)
    _ = (cache / "manifest.json").write_text(
        json_dumps({"format_version": 1, "rdkit": IDENTITY_RDKIT}),
    )
    return _Staged(
        training=training,
        coconut=tmp_path / "coco",
        bio=tmp_path / "bio",
        fingerprints=cache,
    )


def _assemble(staged: _Staged, output: Path) -> None:
    prepare_second_pool(
        staged.training,
        coconut=staged.coconut,
        bio=staged.bio,
        fingerprints=staged.fingerprints,
        output=output,
    )


def test_pool_merges_sources_skips_known_and_invalid_and_sorts_by_mass(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path)
    output = tmp_path / "out"
    _assemble(staged, output)
    np.testing.assert_array_equal(
        np.load(output / "pool_mass.npy"),
        [50.0, 100.0, 150.0, 200.0, 300.0],
    )
    np.testing.assert_array_equal(
        np.load(output / "pool_fp.npy"),
        [[16], [2], [8], [4], [1]],
    )
    np.testing.assert_array_equal(np.load(output / "fp_bits.npy"), _SELECTED)
    rows = parquet.read_table(output / "pool_meta.parquet").to_pylist()
    meta = {name: [row[name] for row in rows] for name in ("mass", "key", "smiles")}
    assert meta["key"] == ["T3", "K2", "T1", "K3", "K1"]
    assert meta["smiles"] == ["t3", "s2", "t1", "s3", "s1"]
    assert meta["mass"] == [50.0, 100.0, 150.0, 200.0, 300.0]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows"] == 5
    assert manifest["selected_bits"] == 5
    assert manifest["raw_bits"] == 10_407
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert len(manifest["inputs"]) == 11
    assert set(manifest["generator"]) == {
        "second_pool.py",
        "rdkit_backend.py",
        "chemistry.py",
    }


@pytest.mark.parametrize(
    "header",
    [
        {"format_version": 2, "rdkit": IDENTITY_RDKIT},
        {"format_version": 1, "rdkit": "0.0.0"},
    ],
)
def test_pool_rejects_a_cache_of_another_format_or_release(
    tmp_path: Path,
    header: dict[str, object],
) -> None:
    staged = _stage(tmp_path)
    _ = (staged.fingerprints / "manifest.json").write_text(json_dumps(header))
    with pytest.raises(ValueError, match=r"^Fingerprint cache requires format 1"):
        _assemble(staged, tmp_path / "out")


@pytest.mark.parametrize(
    "packed",
    [
        # prepare_second_pool packs the five _SELECTED bits into a width of 1 byte.
        np.zeros((3, 1), dtype=np.uint8),
        np.zeros((4, 2), dtype=np.uint8),
        np.zeros((4, 1), dtype=np.int16),
    ],
)
def test_pool_rejects_a_fingerprint_cache_whose_shape_differs(
    tmp_path: Path,
    packed: np.ndarray,
) -> None:
    staged = _stage(tmp_path)
    np.save(staged.fingerprints / "fingerprints.npy", packed)
    with pytest.raises(ValueError, match=r"^Fingerprint cache rows or packed width"):
        _assemble(staged, tmp_path / "out")


@pytest.mark.parametrize("name", ["coco", "bio"])
def test_pool_rejects_provided_fingerprints_whose_shape_differs(
    tmp_path: Path,
    name: str,
) -> None:
    staged = _stage(tmp_path)
    np.save(tmp_path / name / f"{name}_fp.npy", np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"^Provided fingerprint rows or packed width"):
        _assemble(staged, tmp_path / "out")


@pytest.mark.parametrize("name", ["coco", "bio"])
def test_pool_rejects_provided_masses_whose_count_differs(
    tmp_path: Path,
    name: str,
) -> None:
    staged = _stage(tmp_path)
    np.save(tmp_path / name / f"{name}_mass.npy", np.zeros(3))
    with pytest.raises(ValueError, match=r"^Provided mass rows differ from metadata.$"):
        _assemble(staged, tmp_path / "out")


def test_pool_refuses_to_overwrite_an_input(tmp_path: Path) -> None:
    staged = _stage(tmp_path)
    with pytest.raises(ValueError, match="aliases protected input artifact"):
        _assemble(staged, staged.training)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
