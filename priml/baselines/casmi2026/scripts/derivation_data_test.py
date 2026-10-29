"""Replay prepared terminal paths, site environments and identities with a fake SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import types

from pyarrow import Table, parquet
from treekle import from_plain, json_loads

import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts import derivation_data


if TYPE_CHECKING:
    from priml.baselines.casmi2026.scripts.derivation_chemistry import (
        DerivationChemistry,
    )


_CWD: Final = Path(__file__).resolve().parent


def _fake_chemistry(version: str) -> DerivationChemistry:
    assert version == IDENTITY_RDKIT

    def canonical(value: str) -> str:
        if value == "unserializable":
            raise RuntimeError("invalid terminal")
        return value

    def products(value: str, *, rule: str, limit: int) -> list[tuple[str, int]]:
        assert limit == {("root", "first"): 32, ("branch", "second"): 16}[(value, rule)]
        return {
            ("root", "first"): [("branch", 3), ("unused", 9)],
            ("branch", "second"): [
                ("leaf", 4),
                ("rejected", 8),
                ("unserializable", 13),
            ],
        }[(value, rule)]

    def environments(value: str) -> dict[int, tuple[int, int, int]]:
        return {3: (1, 2, 3)} if value == "root" else {4: (4, 5, 6)}

    def parse(value: str) -> str | None:
        return None if value == "invalid" else value

    def standardize(value: str) -> str | None:
        return None if value == "rejected" else "normalized-" + value

    def identity(value: str) -> tuple[str, str]:
        return "key-" + value, "skeleton-" + value

    return cast(
        "DerivationChemistry",
        types.SimpleNamespace(
            chem=types.SimpleNamespace(MolFromSmiles=parse, MolToSmiles=canonical),
            rules={"first": "first", "second": "second"},
            products=products,
            environments=environments,
            standard=types.SimpleNamespace(standardize=standardize),
            identity=identity,
        ),
    )


@pytest.mark.parametrize("extra_identities", [False, True])
def test_prepared_paths_keep_order_failures_limits_and_complete_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_identities: bool,
) -> None:
    monkeypatch.setattr(derivation_data, "DerivationChemistry", _fake_chemistry)
    request = tmp_path / "requests.parquet"
    parquet.write_table(
        Table.from_pylist(
            [
                {"parent_smiles": "root", "combo": "first+second", "max_mols": 1},
                {"parent_smiles": "invalid", "combo": "first", "max_mols": None},
            ],
        ),
        request,
    )
    identities = tmp_path / "identities.parquet"
    parquet.write_table(Table.from_pydict({"smiles": ["extra", "leaf"]}), identities)
    output = tmp_path / "prepared"
    derivation_data.prepare_derivations(
        request,
        identities=identities if extra_identities else None,
        output=output,
        version=IDENTITY_RDKIT,
    )
    assert parquet.read_table(output / "requests.parquet").to_pylist() == [
        {
            "parent_smiles": "root",
            "combo": ["first", "second"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 1,
            "terminal_count": 3,
            "applications": "root\tfirst\t32\t2\nbranch\tsecond\t16\t3",
        },
        {
            "parent_smiles": "invalid",
            "combo": ["first"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 0,
            "applications": "",
        },
    ]
    assert parquet.read_table(output / "paths.parquet").to_pylist() == [
        {
            "request": 0,
            "position": 0,
            "raw": "leaf",
            "standardized": "normalized-leaf",
            "key": "key-normalized-leaf",
            "skeleton": "skeleton-normalized-leaf",
            "environments": [[1, 2, 3], [4, 5, 6]],
        },
        {
            "request": 0,
            "position": 1,
            "raw": "rejected",
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": [[1, 2, 3], [-1, -1, -1]],
        },
        {
            "request": 0,
            "position": 2,
            "raw": None,
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": [[1, 2, 3], [-1, -1, -1]],
        },
    ]
    wanted = ["root", "leaf", "normalized-leaf", "rejected", "invalid"]
    if extra_identities:
        wanted.append("extra")
    assert parquet.read_table(output / "identities.parquet").to_pylist() == [
        {"smiles": value, "key": "key-" + value, "skeleton": "skeleton-" + value}
        for value in wanted
    ]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "requests", "paths", "identities")
    } == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "requests": 2,
        "paths": 3,
        "identities": len(wanted),
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ([request, identities] if extra_identities else [request])
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        Path(name).name: hashlib.sha256((_CWD / name).read_bytes()).hexdigest()
        for name in (
            "derivation_data.py",
            "derivation_chemistry.py",
            "rdkit_backend.py",
            "../chemistry.py",
            "../v6_reactions.tsv",
        )
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
