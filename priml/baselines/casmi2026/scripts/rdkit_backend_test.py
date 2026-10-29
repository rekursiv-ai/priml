"""Checks for the rdkit_backend module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts import rdkit_backend
from priml.baselines.casmi2026.scripts.rdkit_backend import (
    Chemistry,
    MolecularContext,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


def _atom(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(GetSymbol=lambda: symbol)


def _atoms() -> list[SimpleNamespace]:
    return [_atom(symbol) for symbol in ("O", "C", "H", "C")]


def _mol(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, GetAtoms=_atoms, GetNumHeavyAtoms=lambda: 3)


class _Bits:
    def __init__(self, index: int) -> None:
        self.bits = np.array([index % 2, index // 2 % 2], dtype=np.uint8)

    def fingerprint(self, mol: SimpleNamespace) -> NDArray[np.uint8]:
        del mol
        return self.bits


def _fingerprint(index: int) -> SimpleNamespace:
    return SimpleNamespace(GetFingerprintAsNumPy=_Bits(index).fingerprint)


class _Library:
    """Record every call of the scientific library the backend makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.features = object()
        self.rdkit = SimpleNamespace(
            __version__=IDENTITY_RDKIT,
            MolFromSmiles=self.parse,
            MolToSmiles=self.to_smiles,
            MolToInchiKey=self.key,
            Mol=self.copy,
            RemoveStereochemistry=self.remove_stereo,
            GetFormalCharge=self.charge,
        )

    def parse(self, text: str) -> SimpleNamespace | None:
        self.calls.append(("parse", text))
        if text == "error":
            raise ValueError("Bad chemistry input.")
        return None if text == "invalid" else _mol(text)

    def to_smiles(self, mol: SimpleNamespace) -> str:
        self.calls.append(("smiles", mol.name))
        return f"{mol.name}-canonical"

    def key(self, mol: SimpleNamespace) -> str:
        self.calls.append(("key", mol.name))
        return "" if mol.name == "empty" else f"ABCDEFGHIJKLMN-{mol.name}"

    def copy(self, mol: SimpleNamespace) -> SimpleNamespace:
        self.calls.append(("copy", mol.name))
        return _mol(mol.name + "-copy")

    def remove_stereo(self, mol: SimpleNamespace) -> None:
        self.calls.append(("stereo", mol.name))

    def charge(self, mol: SimpleNamespace) -> int:
        return int(mol.name == "charged")

    def tautomers(self) -> SimpleNamespace:
        return SimpleNamespace(Canonicalize=self.canonicalize)

    def canonicalize(self, mol: SimpleNamespace) -> SimpleNamespace:
        self.calls.append(("tautomer", mol.name))
        return mol

    def largest(self) -> SimpleNamespace:
        return SimpleNamespace(choose=self.choose)

    def choose(self, mol: SimpleNamespace) -> SimpleNamespace:
        self.calls.append(("largest", mol.name))
        return mol

    def uncharger(self) -> SimpleNamespace:
        return SimpleNamespace(uncharge=self.uncharge)

    def uncharge(self, mol: SimpleNamespace) -> SimpleNamespace:
        self.calls.append(("uncharge", mol.name))
        return mol

    def feature_invariants(self) -> object:
        return self.features

    def morgan(self, **kwargs: object) -> SimpleNamespace:
        radius = cast(int, kwargs["radius"])
        featured = kwargs.get("atomInvariantsGenerator") is self.features
        self.calls.append(("morgan", (radius, kwargs["fpSize"], featured)))
        return _fingerprint(radius - 2 + 2 * featured)

    def rdk(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("rdk", (kwargs["fpSize"], kwargs["maxPath"])))
        return _fingerprint(3)

    def pair(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("pair", kwargs["fpSize"]))
        return _fingerprint(4)

    def torsion(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("torsion", kwargs["fpSize"]))
        return _fingerprint(5)


def _maccs(mol: SimpleNamespace) -> list[int]:
    del mol
    return [1, 0, 1]


def _mass(mol: SimpleNamespace) -> float:
    del mol
    return 46.041864812


def _formula(mol: SimpleNamespace) -> str:
    del mol
    return "C2H6O"


def _install(monkeypatch: pytest.MonkeyPatch, library: _Library) -> None:
    """Replace every lazily imported scientific name of the backend by one fake."""
    for name, value in (
        ("TautomerEnumerator", library.tautomers),
        ("LargestFragmentChooser", library.largest),
        ("Uncharger", library.uncharger),
        ("GetMorganGenerator", library.morgan),
        ("GetMorganFeatureAtomInvGen", library.feature_invariants),
        ("GetRDKitFPGenerator", library.rdk),
        ("GetAtomPairGenerator", library.pair),
        ("GetTopologicalTorsionGenerator", library.torsion),
        ("GenMACCSKeys", _maccs),
        ("ExactMolWt", _mass),
        ("CalcMolFormula", _formula),
        ("rdkit", library.rdkit),
        ("Chem", library.rdkit),
    ):
        monkeypatch.setattr(rdkit_backend, name, value)


def test_complete_structure_preparation_and_primitive_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _Library()
    _install(monkeypatch, library)
    chemistry = Chemistry(IDENTITY_RDKIT)
    assert library.calls == [
        ("morgan", (2, 4096, False)),
        ("morgan", (3, 4096, False)),
        ("morgan", (2, 2048, True)),
        ("rdk", (2048, 6)),
        ("pair", 2048),
        ("torsion", 2048),
    ]
    record = chemistry.record("CCO")
    assert record is not None
    assert (record.smiles, record.formula, record.mass, record.heavy_atoms) == (
        "CCO",
        "C2H6O",
        46.041864812,
        3,
    )
    assert (record.key, record.raw_key, record.metric_key) == ("ABCDEFGHIJKLMN",) * 3
    assert record.composition == (("C", 2), ("O", 1))
    packed: NDArray[np.uint8] = record.packed
    np.testing.assert_array_equal(
        packed,
        np.packbits([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]),
    )
    assert library.calls[-8:] == [
        ("parse", "CCO"),
        ("key", "CCO"),
        ("tautomer", "CCO"),
        ("key", "CCO"),
        ("copy", "CCO"),
        ("stereo", "CCO-copy"),
        ("tautomer", "CCO-copy"),
        ("key", "CCO-copy"),
    ]
    assert chemistry.standardize("CCO") == "CCO-canonical"
    assert library.calls[-5:] == [
        ("parse", "CCO"),
        ("largest", "CCO"),
        ("uncharge", "CCO"),
        ("stereo", "CCO"),
        ("smiles", "CCO"),
    ]
    assert chemistry.standardize("charged") is None
    assert chemistry.standardize("charged", keep_charged=True) == "charged-canonical"
    for value in ("invalid", "error"):
        assert chemistry.record(value) is None
        assert chemistry.standardize(value) is None
    assert chemistry.record("empty") is None


def test_library_version_is_checked_before_any_primitive_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _Library()
    _install(monkeypatch, library)
    with pytest.raises(
        ValueError,
        match=rf"^Preparation requires RDKit {SIMULATOR_RDKIT}; got {IDENTITY_RDKIT}\.$",
    ):
        _ = Chemistry(SIMULATOR_RDKIT)
    assert library.calls == []


def test_reaction_context_builds_only_molecular_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _Library()
    _install(monkeypatch, library)
    context = MolecularContext(IDENTITY_RDKIT)
    assert context.version == IDENTITY_RDKIT
    assert context.chem is library.rdkit
    assert library.calls == []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
