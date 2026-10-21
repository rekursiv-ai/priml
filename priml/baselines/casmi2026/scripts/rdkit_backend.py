"""Typed standard chemistry primitives for an isolated preparation interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import NDArray

import numpy as np


if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.MACCSkeys import GenMACCSKeys
    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        LargestFragmentChooser,
        TautomerEnumerator,
        Uncharger,
    )
    from rdkit.Chem.rdFingerprintGenerator import (
        GetAtomPairGenerator,
        GetMorganFeatureAtomInvGen,
        GetMorganGenerator,
        GetRDKitFPGenerator,
        GetTopologicalTorsionGenerator,
    )
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    import rdkit
else:
    from wrapt import lazy_import

    rdkit = lazy_import("rdkit")
    Chem = lazy_import("rdkit.Chem")
    ExactMolWt = lazy_import("rdkit.Chem.Descriptors", "ExactMolWt")
    GenMACCSKeys = lazy_import("rdkit.Chem.MACCSkeys", "GenMACCSKeys")
    LargestFragmentChooser = lazy_import(
        "rdkit.Chem.MolStandardize.rdMolStandardize",
        "LargestFragmentChooser",
    )
    TautomerEnumerator = lazy_import(
        "rdkit.Chem.MolStandardize.rdMolStandardize",
        "TautomerEnumerator",
    )
    Uncharger = lazy_import("rdkit.Chem.MolStandardize.rdMolStandardize", "Uncharger")
    GetAtomPairGenerator = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetAtomPairGenerator",
    )
    GetMorganFeatureAtomInvGen = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetMorganFeatureAtomInvGen",
    )
    GetMorganGenerator = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetMorganGenerator",
    )
    GetRDKitFPGenerator = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetRDKitFPGenerator",
    )
    GetTopologicalTorsionGenerator = lazy_import(
        "rdkit.Chem.rdFingerprintGenerator",
        "GetTopologicalTorsionGenerator",
    )
    CalcMolFormula = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcMolFormula")


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class StructureRecord:
    """Complete reusable identity, properties and seven-block packed fingerprint."""

    smiles: str
    formula: str
    mass: float
    heavy_atoms: int
    key: str
    raw_key: str
    metric_key: str
    composition: tuple[tuple[str, int], ...]
    packed: NDArray[np.uint8]


class MolecularContext:
    """Own only molecular primitives for reaction-only preparation jobs."""

    def __init__(self, version: str) -> None:
        if rdkit.__version__ != version:
            raise ValueError(
                f"Preparation requires RDKit {version}; got {rdkit.__version__}.",
            )
        self.version = version
        self.chem = Chem


class Chemistry(MolecularContext):
    """Own standard RDKit objects solely inside the data-generation process."""

    def __init__(self, version: str) -> None:
        super().__init__(version)
        self.tautomers = TautomerEnumerator()
        self.largest = LargestFragmentChooser()
        self.uncharger = Uncharger()
        self.generators = (
            GetMorganGenerator(radius=2, fpSize=4096),
            GetMorganGenerator(radius=3, fpSize=4096),
            GetMorganGenerator(
                radius=2,
                fpSize=2048,
                atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
            ),
            GetRDKitFPGenerator(fpSize=2048, maxPath=6),
            GetAtomPairGenerator(fpSize=2048),
            GetTopologicalTorsionGenerator(fpSize=2048),
        )

    def standardize(self, smiles: str, *, keep_charged: bool = False) -> str | None:
        """Choose the largest fragment, uncharge and remove stereochemistry.

        Args:
          smiles: Raw structure string, before standardization.
          keep_charged: Admit a molecule whose formal charge remains nonzero.

        Returns:
          standardized: Canonical stereo-free SMILES, or None for rejected inputs.

        """
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            molecule = self.uncharger.uncharge(self.largest.choose(molecule))
            if not keep_charged and self.chem.GetFormalCharge(molecule) != 0:
                return None
            self.chem.RemoveStereochemistry(molecule)
            return self.chem.MolToSmiles(molecule)
        except (ValueError, RuntimeError, TypeError):
            return None

    def record(self, smiles: str) -> StructureRecord | None:
        """Compute complete properties, both engine keys and official identity.

        Args:
          smiles: Structure string; standardization is a separate explicit operation.

        Returns:
          record: Complete numerical fingerprint and identities, or None if invalid.

        """
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            parts = [
                generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
                for generator in self.generators
            ]
            parts.append(np.array(list(GenMACCSKeys(molecule)), np.uint8))
            raw_key = self.chem.MolToInchiKey(molecule)[:14]
            key = self.chem.MolToInchiKey(self.tautomers.Canonicalize(molecule))[:14]
            stereo_free = self.chem.Mol(molecule)
            self.chem.RemoveStereochemistry(stereo_free)
            metric_key = self.chem.MolToInchiKey(
                self.tautomers.Canonicalize(stereo_free),
            )[:14]
            if not raw_key or not key or not metric_key:
                return None
            composition: dict[str, int] = {}
            for atom in stereo_free.GetAtoms():
                element = atom.GetSymbol()
                if element != "H":
                    composition[element] = composition.get(element, 0) + 1
            return StructureRecord(
                smiles=smiles,
                formula=CalcMolFormula(molecule),
                mass=ExactMolWt(molecule),
                heavy_atoms=molecule.GetNumHeavyAtoms(),
                key=key,
                raw_key=raw_key,
                metric_key=metric_key,
                composition=tuple(sorted(composition.items())),
                packed=np.packbits(np.concatenate(parts)),
            )
        except (ValueError, RuntimeError, TypeError):
            return None
