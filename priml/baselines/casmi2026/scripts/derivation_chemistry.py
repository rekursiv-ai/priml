"""Observe reaction sites and chemistry identities through standard SDK primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    reaction_rules,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.scripts.products import unique_products
from priml.baselines.casmi2026.scripts.rdkit_backend import Chemistry


if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
    from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator

    from priml.baselines.casmi2026.isolated import Reaction
else:
    from wrapt import lazy_import

    AdditionalOutput = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "AdditionalOutput",
    )
    GetMorganGenerator = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetMorganGenerator",
    )
    ReactionFromSmarts = lazy_import("rdkit.Chem.rdChemReactions", "ReactionFromSmarts")


class DerivationChemistry:
    """Own only isolated scientific operations, never prior probabilities."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Derivation preparation requires RDKit {IDENTITY_RDKIT}.")
        self.standard = Chemistry(version)
        self.chem = self.standard.chem
        self.morgan = GetMorganGenerator(radius=3)
        self.rules = {rule.name: rule for rule in v6_reaction_rules()}
        self.rules.update(
            (rule.name, rule) for rule in reaction_rules() if rule.name == "dehexose2"
        )
        self._compiled: dict[str, ChemicalReaction] = {}

    def environments(
        self,
        molecule: Chem.Mol,
    ) -> dict[int, tuple[int, int, int]]:
        """Return every atom's complete unfolded radius-one/two/three environment.

        Args:
          molecule: Current reaction-stage object, preserving its atom order.

        Returns:
          environments: All atom indices with smaller-radius and missing-site backoff.

        """
        extra = AdditionalOutput()
        extra.AllocateBitInfoMap()
        _ = self.morgan.GetSparseCountFingerprint(molecule, additionalOutput=extra)
        observed: dict[tuple[int, int], int] = {}
        for bit, entries in extra.GetBitInfoMap().items():
            observed.update(((atom, radius), int(bit)) for atom, radius in entries)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.
        result: dict[int, tuple[int, int, int]] = {}
        for index in range(molecule.GetNumAtoms()):
            previous = observed.get((index, 0), -1)
            values: list[int] = []
            for radius in (1, 2, 3):
                previous = observed.get((index, radius), previous)
                values.append(previous)
            result[index] = (values[0], values[1], values[2])
        return result

    def products(
        self,
        molecule: Chem.Mol,
        *,
        rule: Reaction,
        limit: int,
    ) -> list[tuple[Chem.Mol, int]]:
        """Retain distinct complete products and their actual parent atom indices.

        Args:
          molecule: Actual current-stage object; canonical strings never replace it.
          rule: Declared scientific transformation.
          limit: Unique-product cap; zero still admits the first product.

        Returns:
          products: Sanitized first products in reaction encounter order and their sites.

        """
        if rule.name not in self._compiled:
            self._compiled[rule.name] = ReactionFromSmarts(rule.smarts)
        try:
            groups = self._compiled[rule.name].RunReactants((molecule,))
        except Exception:  # noqa: BLE001 -- The scientific extension can raise arbitrary reaction errors.
            return []
        atoms = molecule.GetNumHeavyAtoms() + sum(
            count for element, count in rule.delta if element != "H"
        )
        unique = unique_products(
            groups,
            canonical=self._canonical,
            limit=limit,
            keep=lambda product: product.GetNumAtoms() == atoms,
        )
        return [
            (
                product,
                next(
                    (
                        atom.GetIntProp("react_atom_idx")
                        for atom in product.GetAtoms()
                        if atom.HasProp("old_mapno")
                        and atom.GetIntProp("old_mapno") == 1
                    ),
                    -1,
                ),
            )
            for product, _ in unique
        ]

    def _canonical(self, product: Chem.Mol) -> str:
        _ = self.chem.SanitizeMol(product)
        return self.chem.MolToSmiles(product)

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        """Compute both independent identities, retaining present-empty skeletons.

        Args:
          smiles: Candidate, proposal or derivation-log structure as supplied.

        Returns:
          key: Tautomer-canonical InChIKey14 or an explicit invalid identity.
          skeleton: Complete element/connectivity identity or explicit failure.

        """
        key = None
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical = self.standard.tautomers.Canonicalize(molecule)
                encoded = self.standard.chem.MolToInchiKey(canonical)
                key = encoded[:14] if encoded else None
        except Exception:  # noqa: BLE001 -- Scientific tautomer/InChI failures carry an explicit invalid key.
            key = None
        skeleton = None
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is not None:
                editable = self.chem.RWMol(molecule)
                for bond in editable.GetBonds():
                    bond.SetBondType(self.chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)
                for atom in editable.GetAtoms():
                    atom.SetIsAromatic(False)
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(True)
                    atom.SetChiralTag(self.chem.ChiralType.CHI_UNSPECIFIED)
                skeleton = self.chem.MolToSmiles(editable, isomericSmiles=False)
        except Exception:  # noqa: BLE001 -- Invalid scientific graph identities remain explicit failures.
            skeleton = None
        return key, skeleton
