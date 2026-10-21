"""Extract shared atom properties from isolated scientific-library primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import re

from numpy.typing import NDArray

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AtomProperties:
    """Complete ordered formula, hydrogen and isotope-aware mass arrays."""

    formula: NDArray[np.float64]
    atom_formulas: NDArray[np.float64]
    elements: NDArray[np.int64]
    hydrogens: NDArray[np.int64]
    masses: NDArray[np.float64]
    hydrogenated_masses: NDArray[np.float64]
    mass: float
    total_hydrogens: int


def atom_properties(
    atoms: Sequence[Chem.Atom],
    *,
    formula: str,
    elements: Sequence[str],
    masses: NDArray[np.float64],
    periodic_table: Chem.PeriodicTable,
) -> AtomProperties:
    """Retain the full physical atom records used by both forward simulators.

    Args:
      atoms: Complete ordered atoms after family-specific canonicalization.
      formula: Molecular formula before Kekulization.
      elements: Complete ordered physical element vocabulary.
      masses: Monoisotopic masses aligned with that vocabulary.
      periodic_table: Isotope mass lookup in the isolated interpreter.

    Returns:
      properties: All per-atom and molecular physical arrays without truncation.

    """
    composition = np.zeros(len(elements))
    for match in re.finditer(r"([A-Z][a-z]*)([0-9]*)", formula):
        symbol, count = match.groups()
        composition[elements.index(symbol)] += int(count) if count else 1
    indices = np.fromiter(
        (elements.index(atom.GetSymbol()) for atom in atoms),
        dtype=np.int64,
    )
    hydrogens = np.fromiter(
        (atom.GetNumImplicitHs() + atom.GetNumExplicitHs() for atom in atoms),
        dtype=np.int64,
    )
    atom_masses = np.array(
        [
            periodic_table.GetMassForIsotope(atom.GetSymbol(), atom.GetIsotope())
            if atom.GetIsotope()
            else masses[elements.index(atom.GetSymbol())]
            for atom in atoms
        ],
    )
    hydrogen = elements.index("H")
    hydrogenated = cast(
        "NDArray[np.float64]",
        hydrogens * masses[hydrogen] + atom_masses,
    )
    atom_formulas = np.eye(len(elements))[indices]
    atom_formulas[:, hydrogen] += hydrogens
    return AtomProperties(
        formula=composition,
        atom_formulas=atom_formulas,
        elements=indices,
        hydrogens=hydrogens,
        masses=atom_masses,
        hydrogenated_masses=hydrogenated,
        mass=float(hydrogenated.sum()),
        total_hydrogens=int(hydrogens.sum()),
    )
