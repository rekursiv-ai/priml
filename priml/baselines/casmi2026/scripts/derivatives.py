"""Bounded chemical derivative enumeration in the isolated preparation process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from priml.baselines.casmi2026.isolated import matching_plans, reaction_rules
from priml.baselines.casmi2026.scripts.products import unique_products


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rdkit import Chem
    from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

    from priml.baselines.casmi2026.isolated import Reaction
else:
    from wrapt import lazy_import

    Chem = lazy_import("rdkit.Chem")
    ReactionFromSmarts = lazy_import("rdkit.Chem.rdChemReactions", "ReactionFromSmarts")


@dataclass(frozen=True, slots=True, kw_only=True)
class Derivative:
    """A canonical product and its first successful reaction plan."""

    smiles: str
    plan: tuple[int, ...]


class DerivativeGenerator:
    """Execute mass-selected reaction plans with explicit resource budgets."""

    def __init__(
        self,
        *,
        rules: Sequence[Reaction] | None = None,
        on_application: Callable[[Chem.Mol, Reaction, int, int], None] | None = None,
    ) -> None:
        self.rules = reaction_rules() if rules is None else tuple(rules)
        self.on_application = on_application
        self._compiled: dict[str, ChemicalReaction] = {}

    def apply(
        self,
        molecule: Chem.Mol,
        *,
        rule: Reaction,
        max_products: int,
    ) -> list[Chem.Mol]:
        """Return unique sanitized first products in scientific-library order.

        Args:
          molecule: The parent in its current reaction stage.
          rule: Scientific SMARTS and stable reaction identity.
          max_products: Stop after this many unique products; zero retains the first.

        Returns:
          products: Sanitized first products, with canonical duplicates removed.

        """
        if rule.name not in self._compiled:
            self._compiled[rule.name] = ReactionFromSmarts(rule.smarts)
        try:
            products = self._compiled[rule.name].RunReactants((molecule,))
        except Exception:  # noqa: BLE001 -- The scientific extension can raise arbitrary reaction errors.
            products = ()
        unique = unique_products(
            products,
            canonical=_canonical,
            limit=max_products,
        )
        if self.on_application is not None:
            self.on_application(molecule, rule, max_products, len(unique))
        return [product for product, _ in unique]

    def derive(
        self,
        smiles: str,
        *,
        delta: float,
        tolerance: float,
        max_reactions: int,
        max_products: int,
        allow_isomerization: bool,
        application_budget: int,
        stage_parents: int,
        first_products: int,
        later_products: int,
    ) -> list[Derivative]:
        """Return ordered products and first-plan provenance within all budgets.

        Args:
          smiles: Parent string; only this exact string is excluded from products.
          delta: Requested neutral-mass difference.
          tolerance: Inclusive plan mass tolerance.
          max_reactions: Maximum transformations per plan, at most two.
          max_products: Final per-parent output cap in encounter order.
          allow_isomerization: Admit neutral remove/add plans near zero delta.
          application_budget: Total reaction applications, including empty outcomes.
          stage_parents: Maximum intermediate molecules expanded per stage.
          first_products: Unique-product cap for each first-stage application.
          later_products: Unique-product cap for each later-stage application.

        Returns:
          products: Canonical products with their first matching plan. Exhausting the
            application budget preserves the current partial stage for the mass gate.

        """
        parent = Chem.MolFromSmiles(smiles)
        if parent is None or (abs(delta) < tolerance and not allow_isomerization):
            return []
        plans = matching_plans(
            self.rules,
            delta,
            tolerance=tolerance,
            max_reactions=max_reactions,
        )
        discovered: dict[str, Derivative] = {}
        remaining = application_budget
        for plan in plans:
            molecules = [parent]
            for stage, index in enumerate(plan):
                next_stage: list[Chem.Mol] = []
                for molecule in molecules[:stage_parents]:
                    next_stage.extend(
                        self.apply(
                            molecule,
                            rule=self.rules[index],
                            max_products=first_products
                            if stage == 0
                            else later_products,
                        ),
                    )
                    remaining -= 1
                    if remaining <= 0:
                        break
                molecules = next_stage
                if not molecules or remaining <= 0:
                    break
            for molecule in molecules:
                canonical = Chem.MolToSmiles(molecule)
                if canonical != smiles and canonical not in discovered:
                    discovered[canonical] = Derivative(smiles=canonical, plan=plan)
            if len(discovered) >= max_products or remaining <= 0:
                break
        return list(discovered.values())[:max_products]


def _canonical(product: Chem.Mol) -> str:
    _ = Chem.SanitizeMol(product)
    return Chem.MolToSmiles(product)
