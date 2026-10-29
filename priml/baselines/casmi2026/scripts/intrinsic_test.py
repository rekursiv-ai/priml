"""Checks for the intrinsic module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import sys
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    record_fields,
)
from priml.baselines.casmi2026.scripts import intrinsic
from priml.baselines.casmi2026.scripts.intrinsic import (
    capability_rules,
    descriptor_rules,
)


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray


_SCRIPTS: Final = _CWD


def test_capability_vocabulary_is_the_released_one() -> None:
    rules = capability_rules()
    assert [name for name, _ in rules] == [
        "oh_aliph",
        "cooh",
        "phenol",
        "nh2",
        "carbonyl",
        "ester",
        "acetyl",
        "ome",
        "ome_ar",
        "nme",
        "hex",
        "dhex",
        "pent",
        "glca",
        "cglyc",
        "malonyl",
        "coum",
        "caff",
        "ferul",
        "galloyl",
        "sulf",
        "phos",
    ]
    text = "\n".join(f"{name}\t{smarts}" for name, smarts in rules)
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == "9116486b2bde483380295eaa13049fe841361fd737ae4cdb7dbf6f21e36ac5d4"
    )
    assert {name for name, _ in descriptor_rules()} <= set(record_fields())


def test_records_capability_caps_and_failure_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chemistry, record, capabilities = _boundary(monkeypatch)
    np.testing.assert_array_equal(chemistry.descriptors("valid"), record)
    np.testing.assert_array_equal(chemistry.capabilities("valid"), capabilities)
    for value in ("invalid", "exception"):
        np.testing.assert_array_equal(chemistry.descriptors(value), np.zeros(34))
        np.testing.assert_array_equal(chemistry.capabilities(value), np.zeros(22))
    for value in ("empty-scaffold", "missing-scaffold", "failed-scaffold"):
        empty = record.copy()
        for prefix, offset in (("S:", 0), ("G:", 2)):
            hashed = int.from_bytes(
                hashlib.blake2b(prefix.encode(), digest_size=6).digest(),
                "little",
            )
            empty[offset : offset + 2] = [hashed >> 24, hashed & 0xFF_FFFF]
        empty[6:8] = 0
        np.testing.assert_array_equal(chemistry.descriptors(value), empty)
    with pytest.raises(ValueError, match=r"requires RDKit 2026\.03\.3"):
        intrinsic.IntrinsicChemistry(SIMULATOR_RDKIT)


def test_writer_keeps_complete_arrays_vocabularies_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, record, capabilities = _boundary(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "intrinsic"
    parquet.write_table(
        Table.from_pydict({"smiles": ["valid", "invalid", "valid", None]}),
        request,
    )
    intrinsic.prepare_intrinsic(request, output=output, version=IDENTITY_RDKIT)
    assert parquet.read_table(output / "molecules.parquet").column(
        "smiles",
    ).to_pylist() == ["valid", "invalid"]
    np.testing.assert_array_equal(
        cast("NDArray[np.float32]", np.load(output / "records.npy")),
        np.stack([record, np.zeros(34)]),
    )
    np.testing.assert_array_equal(
        cast("NDArray[np.uint8]", np.load(output / "capabilities.npy")),
        np.stack([capabilities, np.zeros(22)]),
    )
    assert (output / "record_names.txt").read_text().split() == list(record_fields())
    assert (output / "capability_names.txt").read_text().split() == [
        name for name, _ in capability_rules()
    ]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {name: manifest[name] for name in ("format_version", "rdkit", "rows")} == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": 2,
    }
    assert manifest["numpy"] == np.__version__
    assert from_plain(manifest["record_fields"], list[str]) == list(record_fields())
    assert from_plain(manifest["capability_fields"], list[str]) == [
        name for name, _ in capability_rules()
    ]
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS / "intrinsic.py",
            _SCRIPTS / "rdkit_backend.py",
            _SCRIPTS.parent / "fragnet.py",
        )
    }
    with pytest.raises(FileExistsError):
        intrinsic.prepare_intrinsic(request, output=output, version=IDENTITY_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    empty = tmp_path / "empty"
    intrinsic.prepare_intrinsic(request, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "molecules.parquet").num_rows == 0
    assert np.load(empty / "records.npy").shape == (0, 34)
    assert np.load(empty / "capabilities.npy").shape == (0, 22)


def _boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[intrinsic.IntrinsicChemistry, NDArray[np.float32], NDArray[np.uint8]]:
    descriptor_counts = {
        query: index + 1 for index, (_, query) in enumerate(descriptor_rules())
    }

    def matches(pattern: str, **kwargs: int) -> tuple[tuple[int, ...], ...]:
        return ((0,),) * (
            200 if kwargs.get("maxMatches") == 255 else descriptor_counts[pattern]
        )

    def parse(value: str) -> types.SimpleNamespace | None:
        if value == "exception":
            raise RuntimeError("scientific failure")
        if value == "invalid":
            return None
        return types.SimpleNamespace(
            name=value,
            GetNumHeavyAtoms=lambda: 4,
            GetAtoms=lambda: [
                types.SimpleNamespace(GetSymbol=lambda: "N", GetFormalCharge=lambda: 1),
                types.SimpleNamespace(
                    GetSymbol=lambda: "O",
                    GetFormalCharge=lambda: -1,
                ),
                types.SimpleNamespace(GetSymbol=lambda: "C", GetFormalCharge=lambda: 0),
                types.SimpleNamespace(GetSymbol=lambda: "C", GetFormalCharge=lambda: 0),
            ],
            GetRingInfo=lambda: types.SimpleNamespace(NumRings=lambda: 1),
            GetSubstructMatches=matches,
        )

    def scaffold(molecule: types.SimpleNamespace) -> types.SimpleNamespace | None:
        name = from_plain(cast(object, molecule.name), str)
        if name == "failed-scaffold":
            raise RuntimeError("scaffold failure")
        if name == "missing-scaffold":
            return None
        size = 0 if name == "empty-scaffold" else 3
        return types.SimpleNamespace(
            name="scaffold" if size else "",
            GetNumAtoms=lambda: size,
            GetNumHeavyAtoms=lambda: size,
        )

    def canonical(molecule: types.SimpleNamespace) -> str:
        return from_plain(cast(object, molecule.name), str)

    def descriptor(value: object, *, result: float) -> float:
        del value
        return result

    def unchanged(value: str) -> str:
        return value

    def generic(value: object) -> types.SimpleNamespace:
        del value
        return types.SimpleNamespace(name="generic")

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolFromSmarts=unchanged,
        MolToSmiles=canonical,
    )

    def context(version: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(version=version, chem=chemistry)

    monkeypatch.setattr(intrinsic, "MolecularContext", context)
    modules: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "rdkit.Chem.rdMolDescriptors",
            {
                "CalcNumAromaticRings": partial(descriptor, result=1),
                "CalcTPSA": partial(descriptor, result=float("inf")),
                "CalcNumHBD": partial(descriptor, result=2),
                "CalcNumHBA": partial(descriptor, result=3),
                "CalcNumRotatableBonds": partial(descriptor, result=4),
                "CalcFractionCSP3": partial(descriptor, result=0.5),
            },
        ),
        ("rdkit.Chem.Crippen", {"MolLogP": partial(descriptor, result=float("nan"))}),
        (
            "rdkit.Chem.Scaffolds.MurckoScaffold",
            {
                "GetScaffoldForMol": scaffold,
                "MakeScaffoldGeneric": generic,
            },
        ),
    )
    for name, methods in modules:
        module = types.ModuleType(name)
        vars(module).update(methods)
        monkeypatch.setitem(sys.modules, name, module)
    expected = np.zeros(34, dtype=np.float32)
    for prefix, offset in (("S:scaffold", 0), ("G:generic", 2)):
        hashed = int.from_bytes(
            hashlib.blake2b(prefix.encode(), digest_size=6).digest(),
            "little",
        )
        expected[offset : offset + 2] = [hashed >> 24, hashed & 0xFF_FFFF]
    expected[4:10] = [1, 1, 3, 3 / 4, 1, 1]
    expected[10:19] = np.arange(1, 10)
    expected[19:30] = [1, 1, 0, 0, 0, 0, 0, 2, 3, 4, 0.5]
    expected[30:34] = [10, 11, 12, 4]
    capabilities = np.full(22, 200, dtype=np.uint8)
    capabilities[[10, 12, 14]] = 255
    return intrinsic.IntrinsicChemistry(IDENTITY_RDKIT), expected, capabilities


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
