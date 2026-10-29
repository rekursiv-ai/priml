"""Reaction sanitization, encounter order, provenance and bounded enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import types

import pytest

from priml.baselines.casmi2026.isolated import Reaction, reaction_rules
from priml.baselines.casmi2026.scripts import derivatives
from priml.baselines.casmi2026.scripts.derivatives import (
    Derivative,
    DerivativeGenerator,
)


if TYPE_CHECKING:
    from rdkit import Chem


@dataclass(frozen=True, slots=True, kw_only=True)
class Structure:
    name: str


class Primitives:
    """Fake scientific library exposing its names through a namespace."""

    def __init__(self) -> None:
        self.compiled: list[str] = []
        self.applied: list[str] = []
        self.chem = types.SimpleNamespace(
            MolFromSmiles=self.parse,
            MolToSmiles=self.to_smiles,
            SanitizeMol=self.sanitize,
        )

    def parse(self, smiles: str) -> Structure | None:
        return None if smiles == "invalid" else Structure(name=smiles)

    def to_smiles(self, molecule: Chem.Mol) -> str:
        return cast("Structure", molecule).name.removeprefix("duplicate:")

    def sanitize(self, molecule: Structure) -> None:
        if molecule.name == "bad":
            raise ValueError("Invalid reaction product.")

    def reaction(self, smarts: str) -> types.SimpleNamespace:
        self.compiled.append(smarts)
        return types.SimpleNamespace(RunReactants=self.run)

    def run(
        self,
        molecules: tuple[Structure],
    ) -> tuple[tuple[Structure, ...], ...]:
        molecule = molecules[0]
        self.applied.append(molecule.name)
        if molecule.name == "error":
            raise RuntimeError("Reaction rejected input.")
        if molecule.name == "empty":
            return ()
        return tuple(
            (Structure(name=name), Structure(name="unused coproduct"))
            for name in (
                "bad",
                molecule.name,
                "duplicate:" + molecule.name,
                molecule.name + "A",
                molecule.name + "B",
            )
        )


def install(monkeypatch: pytest.MonkeyPatch, library: Primitives) -> None:
    """Replace the lazily imported scientific names of the generator by one fake."""
    monkeypatch.setattr(derivatives, "Chem", library.chem)
    monkeypatch.setattr(derivatives, "ReactionFromSmarts", library.reaction)


def derive(
    generator: DerivativeGenerator,
    smiles: str,
    *,
    delta: float,
    tolerance: float = 0.003,
    max_reactions: int = 2,
    max_products: int = 60,
    allow_isomerization: bool = True,
    application_budget: int = 800,
    stage_parents: int = 24,
    first_products: int = 32,
    later_products: int = 16,
) -> list[Derivative]:
    """Derive with the released budgets unless a case overrides one."""
    return generator.derive(
        smiles,
        delta=delta,
        tolerance=tolerance,
        max_reactions=max_reactions,
        max_products=max_products,
        allow_isomerization=allow_isomerization,
        application_budget=application_budget,
        stage_parents=stage_parents,
        first_products=first_products,
        later_products=later_products,
    )


def apply(
    generator: DerivativeGenerator,
    item: Chem.Mol,
    *,
    rule: Reaction,
    max_products: int = 64,
) -> list[Chem.Mol]:
    """Apply one rule with the released cap unless a case overrides it."""
    return generator.apply(item, rule=rule, max_products=max_products)


def molecule(name: str) -> Chem.Mol:
    """Stand in for a parsed scientific molecule."""
    return cast("Chem.Mol", Structure(name=name))


def test_apply_sanitizes_deduplicates_caps_and_reuses_compiled_reactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = Primitives()
    install(monkeypatch, library)
    generator = DerivativeGenerator()
    rule = reaction_rules()[0]
    actual = apply(generator, molecule("parent"), rule=rule)
    assert [library.to_smiles(item) for item in actual] == [
        "parent",
        "parentA",
        "parentB",
    ]
    assert [
        library.to_smiles(item)
        for item in apply(generator, molecule("parent"), rule=rule, max_products=0)
    ] == ["parent"]
    assert apply(generator, molecule("error"), rule=rule) == []
    assert apply(generator, molecule("empty"), rule=rule) == []
    assert library.compiled == [rule.smarts]


def test_derive_retains_first_plan_and_exact_parent_string_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = Primitives()
    install(monkeypatch, library)
    generator = DerivativeGenerator()
    products = derive(generator, "parent", delta=15.9949146196, max_reactions=1)
    assert products == [
        Derivative(smiles="parentA", plan=(0,)),
        Derivative(smiles="parentB", plan=(0,)),
    ]
    assert library.applied == ["parent"] * 3
    assert derive(generator, "invalid", delta=0) == []
    assert derive(generator, "parent", delta=0, allow_isomerization=False) == []
    assert derive(generator, "parent", delta=1000) == []
    assert derive(generator, "parent", delta=0, max_products=0) == []
    assert derive(
        generator,
        "duplicate:parent",
        delta=15.9949146196,
        max_reactions=1,
        max_products=1,
    ) == [Derivative(smiles="parent", plan=(0,))]


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_application_budget_keeps_partial_depth_products_and_first_provenance(
    budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = Primitives()
    install(monkeypatch, library)
    generator = DerivativeGenerator()
    generator.rules = reaction_rules()[:3]
    products = derive(
        generator,
        "parent",
        delta=2 * 15.9949146196,
        application_budget=budget,
        stage_parents=1,
        first_products=2,
        later_products=2,
    )
    assert products == [Derivative(smiles="parentA", plan=(0, 0))]
    assert len(library.applied) == budget


def test_empty_stage_stops_the_current_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    library = Primitives()
    install(monkeypatch, library)
    generator = DerivativeGenerator()
    assert derive(generator, "empty", delta=2 * 15.9949146196) == []
    assert library.applied == ["empty"] * 7


def test_explicit_vocabulary_and_application_provenance_keep_each_actual_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = Primitives()
    install(monkeypatch, library)
    calls: list[tuple[str, str, int, int]] = []

    def observe(item: Chem.Mol, rule: Reaction, limit: int, count: int) -> None:
        calls.append((library.to_smiles(item), rule.name, limit, count))

    generator = DerivativeGenerator(
        rules=reaction_rules()[:1],
        on_application=observe,
    )
    assert derive(generator, "parent", delta=15.9949146196, max_reactions=1) == [
        Derivative(smiles="parentA", plan=(0,)),
        Derivative(smiles="parentB", plan=(0,)),
    ]
    assert calls == [("parent", "hydroxyl_arom", 32, 3)]
    for name in ("error", "empty"):
        assert apply(generator, molecule(name), rule=generator.rules[0]) == []
    assert calls[1:] == [
        ("error", "hydroxyl_arom", 64, 0),
        ("empty", "hydroxyl_arom", 64, 0),
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
