"""Checks for the derivation_chemistry module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, Reaction
from priml.baselines.casmi2026.scripts import derivation_chemistry
from priml.baselines.casmi2026.scripts.derivation_chemistry import (
    DerivationChemistry,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from rdkit import Chem


def _extra() -> SimpleNamespace:
    """Hold the bit information a Morgan generator reports."""

    def allocate() -> None:
        return None

    def bit_info() -> dict[int, list[tuple[int, int]]]:
        return {
            10: [(0, 0), (1, 0)],
            11: [(0, 1)],
            12: [(0, 3)],
            13: [(1, 2)],
        }

    return SimpleNamespace(AllocateBitInfoMap=allocate, GetBitInfoMap=bit_info)


def _morgan(**options: object) -> SimpleNamespace:
    del options

    def fingerprint(molecule: object, **outputs: object) -> None:
        del molecule, outputs

    return SimpleNamespace(GetSparseCountFingerprint=fingerprint)


def _atom(**properties: int) -> SimpleNamespace:
    def has(name: str) -> bool:
        return name in properties

    def get(name: str) -> int:
        return properties[name]

    return SimpleNamespace(HasProp=has, GetIntProp=get)


def _product(
    name: str,
    *,
    atoms: int = 3,
    site: int | None = None,
) -> SimpleNamespace:
    mapped = (
        [_atom(), _atom(old_mapno=1, react_atom_idx=site)]
        if site is not None
        else [_atom(old_mapno=2, react_atom_idx=9)]
    )
    return SimpleNamespace(
        name=name,
        GetNumAtoms=lambda: atoms,
        GetAtoms=lambda: mapped,
    )


def _reaction(groups: list[tuple[SimpleNamespace, ...]] | None) -> SimpleNamespace:
    def run(reactants: tuple[object, ...]) -> list[tuple[SimpleNamespace, ...]]:
        del reactants
        if groups is None:
            raise RuntimeError("reaction failed")
        return groups

    return SimpleNamespace(RunReactants=run)


def _recorder(label: str, edits: list[tuple[str, object]]) -> SimpleNamespace:
    """Log every property edit as a labelled pair."""

    def log(name: str) -> Callable[[object], None]:
        def edit(value: object) -> None:
            edits.append((label + name, value))

        return edit

    return SimpleNamespace(
        SetBondType=log("type"),
        SetIsAromatic=log("aromatic"),
        SetFormalCharge=log("charge"),
        SetNumExplicitHs=log("hs"),
        SetNoImplicit=log("implicit"),
        SetChiralTag=log("chiral"),
    )


def _chem(edited: list[tuple[str, object]]) -> SimpleNamespace:
    def sanitize(product: SimpleNamespace) -> None:
        if product.name == "unsanitary":
            raise ValueError(product.name)

    def to_smiles(molecule: SimpleNamespace, **options: bool) -> str:
        if hasattr(molecule, "name"):
            return str(molecule.name)
        return "isomeric" if options.get("isomericSmiles", True) else "skeleton"

    def from_smiles(smiles: str) -> SimpleNamespace | None:
        if smiles == "invalid":
            return None
        if smiles == "broken":
            raise RuntimeError("parse failed")
        return SimpleNamespace(smiles=smiles)

    def inchi_key(molecule: object) -> str:
        del molecule
        return "ABCDEFGHIJKLMN-OPQRSTUVWX-N"

    def editable(molecule: object) -> SimpleNamespace:
        del molecule
        return SimpleNamespace(
            GetBonds=lambda: [_recorder("bond ", edited)],
            GetAtoms=lambda: [_recorder("atom ", edited)],
        )

    return SimpleNamespace(
        BondType=SimpleNamespace(SINGLE="single"),
        ChiralType=SimpleNamespace(CHI_UNSPECIFIED="unspecified"),
        SanitizeMol=sanitize,
        MolToSmiles=to_smiles,
        MolFromSmiles=from_smiles,
        MolToInchiKey=inchi_key,
        RWMol=editable,
    )


def _standard(version: str, edited: list[tuple[str, object]]) -> SimpleNamespace:
    assert version == IDENTITY_RDKIT

    def canonicalize(molecule: SimpleNamespace) -> SimpleNamespace:
        if molecule.smiles == "tautomer-error":
            raise RuntimeError("tautomer failed")
        return molecule

    return SimpleNamespace(
        chem=_chem(edited),
        tautomers=SimpleNamespace(Canonicalize=canonicalize),
    )


@pytest.fixture
def edits() -> list[tuple[str, object]]:
    return []


@pytest.fixture
def chemistry(
    monkeypatch: pytest.MonkeyPatch,
    edits: list[tuple[str, object]],
) -> DerivationChemistry:
    def standard(version: str) -> SimpleNamespace:
        return _standard(version, edits)

    monkeypatch.setattr(derivation_chemistry, "Chemistry", standard)
    monkeypatch.setattr(derivation_chemistry, "AdditionalOutput", _extra)
    monkeypatch.setattr(derivation_chemistry, "GetMorganGenerator", _morgan)
    return DerivationChemistry(IDENTITY_RDKIT)


def _rule() -> Reaction:
    return Reaction(name="rule", smarts="[C:1]>>[C:1]", delta=(("C", 1), ("H", 2)))


def _parent() -> Chem.Mol:
    return cast("Chem.Mol", SimpleNamespace(GetNumHeavyAtoms=lambda: 2))


def test_a_wrong_scientific_release_is_rejected() -> None:
    with pytest.raises(ValueError, match=rf"requires RDKit {IDENTITY_RDKIT}"):
        _ = DerivationChemistry("0.0.0")


def test_the_rule_table_adds_the_declared_dehexose_to_the_v6_vocabulary(
    chemistry: DerivationChemistry,
) -> None:
    assert "dehexose2" in chemistry.rules
    assert chemistry.rules["dehexose2"].name == "dehexose2"
    assert len(chemistry.rules) > 1


def test_environments_back_off_to_the_nearest_smaller_radius_and_to_missing_sites(
    chemistry: DerivationChemistry,
) -> None:
    molecule = SimpleNamespace(GetNumAtoms=lambda: 3)
    assert chemistry.environments(cast("Chem.Mol", molecule)) == {
        0: (11, 11, 12),
        1: (10, 13, 13),
        2: (-1, -1, -1),
    }


def test_products_filter_atoms_deduplicate_reject_and_report_the_mapped_site(
    chemistry: DerivationChemistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _product("first", site=1)
    groups: list[tuple[SimpleNamespace, ...]] = [
        (kept, _product("ignored second member")),
        (_product("wrong size", atoms=9),),
        (_product("first"),),
        (_product("unsanitary"),),
        (_product("second"),),
    ]
    compiled: list[str] = []

    def compile_reaction(smarts: str) -> SimpleNamespace:
        compiled.append(smarts)
        return _reaction(groups)

    monkeypatch.setattr(derivation_chemistry, "ReactionFromSmarts", compile_reaction)
    result = chemistry.products(_parent(), rule=_rule(), limit=10)
    assert [(cast("SimpleNamespace", item).name, site) for item, site in result] == [
        ("first", 1),
        ("second", -1),
    ]
    assert result[0][0] is kept
    result = chemistry.products(_parent(), rule=_rule(), limit=0)
    assert [cast("SimpleNamespace", item).name for item, _ in result] == ["first"]
    assert compiled == ["[C:1]>>[C:1]"]


def test_a_failing_reaction_yields_no_products(
    chemistry: DerivationChemistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(smarts: str) -> SimpleNamespace:
        del smarts
        return _reaction(None)

    monkeypatch.setattr(derivation_chemistry, "ReactionFromSmarts", reject)
    assert chemistry.products(_parent(), rule=_rule(), limit=3) == []


def test_identity_keeps_the_key_prefix_and_resets_every_graph_property(
    chemistry: DerivationChemistry,
    edits: list[tuple[str, object]],
) -> None:
    assert chemistry.identity("CCO") == ("ABCDEFGHIJKLMN", "skeleton")
    assert edits == [
        ("bond type", "single"),
        ("bond aromatic", False),
        ("atom aromatic", False),
        ("atom charge", 0),
        ("atom hs", 0),
        ("atom implicit", True),
        ("atom chiral", "unspecified"),
    ]


@pytest.mark.parametrize("smiles", ["invalid", "broken", "tautomer-error"])
def test_unparsable_or_failing_structures_have_explicit_missing_identities(
    chemistry: DerivationChemistry,
    smiles: str,
) -> None:
    expected_skeleton = "skeleton" if smiles == "tautomer-error" else None
    assert chemistry.identity(smiles) == (None, expected_skeleton)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
