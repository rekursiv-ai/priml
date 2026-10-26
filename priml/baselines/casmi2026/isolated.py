"""Scientific preparation surface shared with the isolated RDKit interpreter.

Imports only the standard library, NumPy, Arrow and RDKit plus the base record types: no
torch, scipy, pandas or configgle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast, override

import csv
import pickle
import re

from numpy.typing import NDArray
from treekle import from_plain, json_dumps, json_loads

import numpy as np

from priml.baselines.casmi2026.custom_types import (
    AnalogMetadata,
    FragnetRecord,
    PreparationHeader,
    ReactionHeader,
    ReactionProduct,
    ReactionRequest,
)
from priml.data.arrays import load_array


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pyarrow import parquet

else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


_THIS: Final = Path(__file__).resolve()

ELEMENT_MASSES: Final = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "P": 30.97376163,
    "S": 31.97207100,
    "F": 18.99840322,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "Na": 22.9897692809,
    "K": 38.96370668,
    "Si": 27.9769265325,
    "B": 11.0093054,
    "Se": 79.9165213,
    "Li": 7.01600455,
    "Mg": 23.985041700,
    "Ca": 39.96259098,
    "Fe": 55.9349375,
    "Cu": 62.9295975,
    "Zn": 63.9291422,
    "As": 74.9215965,
    "Al": 26.98153863,
    "Co": 58.933195,
    "Mn": 54.9380451,
    "Ni": 57.9353429,
    "Cr": 51.9405075,
    "Sn": 119.9021947,
    "Hg": 201.970643,
    "Pt": 194.9647911,
    "Au": 196.9665687,
    "Ag": 106.905097,
    "Ti": 47.9479463,
    "V": 50.9439595,
    "Ge": 73.9211778,
    "Te": 129.9062244,
    "Sb": 120.9038157,
    "Bi": 208.9803987,
    "Ba": 137.9052472,
    "Sr": 87.9056121,
    "Rb": 84.911789738,
    "Cs": 132.905451933,
    "Be": 9.0121822,
    "Ga": 68.9255736,
    "W": 183.9509312,
    "Mo": 97.9054082,
    "Pb": 207.9766521,
    "Tl": 204.9744275,
    "Cd": 113.9033585,
    "Zr": 89.9047044,
    "Pd": 105.903486,
    "Rh": 102.905504,
    "Ru": 101.9043493,
    "Ir": 192.9629264,
    "Os": 191.9614807,
    "Re": 186.9557531,
    "Nb": 92.9063781,
    "Y": 88.9058483,
    "Sc": 44.9559119,
    "La": 138.9063533,
    "Ce": 139.9054387,
    "Nd": 141.9077233,
    "Eu": 152.9212303,
    "Gd": 157.9241039,
    "Tb": 158.9253468,
    "Dy": 163.9291748,
    "Ho": 164.9303221,
    "Er": 165.9302931,
    "Tm": 168.9342133,
    "Yb": 173.9388621,
    "Lu": 174.9407718,
    "Th": 232.0380553,
    "U": 238.0507882,
    "Hf": 179.94655,
    "Ta": 180.9479958,
    "In": 114.903878,
    "Xe": 131.9041535,
    "Kr": 83.911507,
    "Ar": 39.9623831225,
    "Ne": 19.9924401754,
    "He": 4.00260325415,
    "D": 2.0141017778,
}
"""Monoisotopic mass in daltons of every element in the pinned scientific table."""


HYDROGEN_MASS: Final = ELEMENT_MASSES["H"]
"""Monoisotopic hydrogen atom mass, the unit of every hydrogen-shift ladder."""


ELECTRON_MASS: Final = 0.00054857990
"""Electron mass in daltons used by adduct conversion and gap carriers."""


SIMULATOR_ELECTRON_MASS: Final = 0.00054858
"""Electron mass as rounded by the released ICE and GLACIER recipes."""


SIMULATOR_SYMBOLS: Final = (
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "Na",
    "K",
)
"""Released simulator element vocabulary in its model input order."""


SIMULATOR_NUMBERS: Final = (
    6,
    7,
    15,
    8,
    16,
    14,
    53,
    1,
    17,
    9,
    35,
    5,
    34,
    26,
    27,
    33,
    11,
    19,
)
"""Atomic numbers of ``SIMULATOR_SYMBOLS``."""


SIMULATOR_GROUPS: Final = (4, 3, 3, 5, 5, 4, 6, 0, 6, 6, 6, 2, 5, 7, 7, 3, 1, 1)
"""Chemical group of each element in ``SIMULATOR_SYMBOLS``."""


FORMULA_ELEMENTS: Final = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
"""Ten-element formula vocabulary of the pool, structural and subformula tables."""


SIMULATOR_HYDROGEN_MASS: Final = 1.007825032
"""Hydrogen mass as rounded by the released GLACIER hydrogen-shift ladder."""


_MASSES: Final = {
    **{
        element: ELEMENT_MASSES[element]
        for element in ("H", "C", "N", "O", "Na", "K", "Li", "Ca", "Cl", "Br", "F")
    },
    "water": 2 * HYDROGEN_MASS + ELEMENT_MASSES["O"],
    "ammonia": ELEMENT_MASSES["N"] + 3 * HYDROGEN_MASS,
    "methyl": ELEMENT_MASSES["C"] + 3 * HYDROGEN_MASS,
    "electron": ELECTRON_MASS,
    "proton": HYDROGEN_MASS - ELECTRON_MASS,
}


_TRANSFERS: Final = {
    "+H": (("H", 1),),
    "+NH4": (("N", 1), ("H", 4)),
    "+Na": (("Na", 1),),
    "+K": (("K", 1),),
    "+Li": (("Li", 1),),
    "-H2O+H": (("H", 1), ("water", -1)),
    "-2H2O+H": (("H", 1), ("water", -2)),
    "-H2O": (("water", -1),),
    "": (("H", 0),),
    "+2H": (("H", 2),),
    "+3H": (("H", 3),),
    "+H+Na": (("H", 1), ("Na", 1)),
    "+HNa": (("H", 1), ("Na", 1)),
    "+2Na": (("Na", 2),),
    "+2Na-H": (("Na", 2), ("H", -1)),
    "+Ca": (("Ca", 1),),
    "+CH3OH+H": (("H", 1), ("C", 1), ("H", 4), ("O", 1)),
    "+CH3CN+H": (("H", 1), ("C", 2), ("H", 3), ("N", 1)),
    "+CH3CN+Na": (("Na", 1), ("C", 2), ("H", 3), ("N", 1)),
    "+C2H4N": (("C", 2), ("H", 4), ("N", 1)),
    "+CH5O": (("C", 1), ("H", 5), ("O", 1)),
    "-H2N": (("N", -1), ("H", -2)),
    "-NH3+H": (("H", 1), ("ammonia", -1)),
    "-H5O3": (("H", -5), ("O", -3)),
    "-CH3": (("methyl", -1),),
    "+H-CH3": (("H", 1), ("methyl", -1)),
    "-H": (("H", -1),),
    "-2H": (("H", -2),),
    "-H2": (("H", -2),),
    "-H2O-H": (("H", -1), ("water", -1)),
    "+CH2O2-H": (("C", 1), ("H", 2), ("O", 2), ("H", -1)),
    "+HCOO": (("C", 1), ("H", 2), ("O", 2), ("H", -1)),
    "+C2H4O2-H": (("C", 2), ("H", 4), ("O", 2), ("H", -1)),
    "+CH3COO": (("C", 2), ("H", 4), ("O", 2), ("H", -1)),
    "+Cl": (("Cl", 1),),
    "+Br": (("Br", 1),),
    "+Na-2H": (("Na", 1), ("H", -2)),
    "+K-2H": (("K", 1), ("H", -2)),
    "+Na-H2": (("Na", 1), ("H", -2)),
    "+K-H2": (("K", 1), ("H", -2)),
    "-C3H6NO2": (("C", -3), ("H", -6), ("N", -1), ("O", -2)),
    "+C2F3O2": (("C", 2), ("F", 3), ("O", 2)),
    "+CF3COO": (("C", 2), ("F", 3), ("O", 2)),
}


_NAMES: Final = (
    "[M+H]+",
    "[M+NH4]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+Cl]-",
    "[M+Li]+",
    "[M-H2O]+",
    "[M]+",
    "[M+2H]2+",
    "[M+3H]3+",
    "[M+H+Na]2+",
    "[M+HNa]2+",
    "[M+2Na]2+",
    "[M+2Na-H]+",
    "[M+Ca]2+",
    "[M+CH3OH+H]+",
    "[M+CH3CN+H]+",
    "[M+CH3CN+Na]+",
    "[M+C2H4N]+",
    "[M+CH5O]+",
    "[M-H2N]+",
    "[M-NH3+H]+",
    "[M-H5O3]+",
    "[M-CH3]+",
    "[M+H-CH3]+",
    "[M-2H]-",
    "[M-2H]2-",
    "[M-H2]2-",
    "[M]-",
    "[M+HCOO]-",
    "[M+C2H4O2-H]-",
    "[M+CH3COO]-",
    "[M+Br]-",
    "[M+Na-2H]-",
    "[M+K-2H]-",
    "[M+Na-H2]-",
    "[M+K-H2]-",
    "[M-CH3]-",
    "[M-C3H6NO2]-",
    "[M+C2F3O2]-",
    "[M+CF3COO]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M+K]+",
    "[2M+NH4]+",
    "[2M+Ca]2+",
    "[2M-H]-",
    "[2M+CH2O2-H]-",
    "[2M+C2H4O2-H]-",
    "[2M+Na-2H]-",
    "[2M+Cl]-",
    "[3M+H]+",
    "[3M+Na]+",
    "[3M-H]-",
)


SECOND_NAMES: Final = (
    "[M+H]+",
    "[M+NH4]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M+2H]2+",
    "[M]+",
    "[M-H2O]+",
    "[M+CH3OH+H]+",
    "[M+CH3CN+H]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+C2H4O2-H]-",
    "[M+Cl]-",
    "[M]-",
    "[M-2H]-",
    "[M+Na-2H]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M+NH4]+",
    "[2M+K]+",
    "[2M-H]-",
    "[2M+CH2O2-H]-",
    "[2M+C2H4O2-H]-",
    "[2M+Na-2H]-",
    "[3M+H]+",
    "[3M-H]-",
)
"""Second-engine precursor mass table vocabulary, in table order."""


_SECOND_TRANSFERS: Final = {
    "+H": (("proton", 1),),
    "+NH4": (("N", 1), ("H", 4), ("electron", -1)),
    "+Na": (("Na", 1), ("electron", -1)),
    "+K": (("K", 1), ("electron", -1)),
    "-H2O+H": (("proton", 1), ("water", -1)),
    "-2H2O+H": (("proton", 1), ("water", -2)),
    "+2H": (("proton", 2),),
    "-H2O": (("electron", -1), ("water", -1)),
    "+CH3OH+H": (("proton", 1), ("C", 1), ("H", 4), ("O", 1)),
    "+CH3CN+H": (("proton", 1), ("C", 2), ("H", 3), ("N", 1)),
    "-H": (("proton", -1),),
    "-H2O-H": (("proton", -1), ("water", -1)),
    "+CH2O2-H": (("C", 1), ("H", 2), ("O", 2), ("proton", -1)),
    "+C2H4O2-H": (("C", 2), ("H", 4), ("O", 2), ("proton", -1)),
    "+Cl": (("Cl", 1), ("electron", 1)),
    "-2H": (("proton", -2),),
    "+Na-2H": (("Na", 1), ("proton", -2)),
}


def adduct_names() -> tuple[str, ...]:
    """Return every recognized adduct in checkpoint vocabulary order."""
    return _NAMES


def adduct_index(name: str | None) -> int:
    """Return the checkpoint token, using its final token for unknown names."""
    return _NAMES.index(name) if name in _NAMES else len(_NAMES)


def adduct_parameters(name: str) -> tuple[int, int, float, int]:
    """Decode molecule count, absolute charge, mass delta and polarity.

    Args:
      name: Recognized bracketed adduct name.

    Returns:
      molecules: Number of neutral molecules in the ion.
      charge: Absolute ion charge.
      delta: Atom-transfer mass including the electron correction.
      polarity: Positive one or negative one.

    """
    body, suffix = name[1:].split("]")
    coefficient, transfer = body.split("M")
    molecules = int(coefficient or "1")
    charge = int(suffix[:-1] or "1")
    polarity = 1 if suffix[-1] == "+" else -1
    terms = _TRANSFERS[transfer]
    element, count = terms[0]
    delta = _MASSES[element] * count
    for element, count in terms[1:]:
        delta += _MASSES[element] * count
    return molecules, charge, delta - polarity * charge * ELECTRON_MASS, polarity


def polarity_sign(mode: int | str) -> int:
    """Return plus one for positive-mode queries and minus one otherwise."""
    return 1 if mode in (1, "positive") else -1


def main_ion_deltas(adduct: str) -> tuple[float, ...]:
    """Return every source main-engine fragment carrier, including unknown-adduct rules.

    Args:
      adduct: Original query adduct label.

    Returns:
      deltas: Proton/deprotonation followed by applicable metal or counterion masses.

    """
    positive = adduct.endswith("+")
    deltas = [adduct_parameters("[M+H]+" if positive else "[M-H]-")[2]]
    for marker, label in (("Na", "[M+Na]+"), ("K", "[M+K]+"), ("NH4", "[M+NH4]+")):
        if positive and marker in adduct:
            deltas.append(adduct_parameters(label)[2])
    if not positive:
        if "Cl" in adduct:
            deltas.append(adduct_parameters("[M+Cl]-")[2])
        if "CH2O2" in adduct or "HCOO" in adduct:
            deltas.append(
                ELEMENT_MASSES["C"]
                + 2 * HYDROGEN_MASS
                + 2 * ELEMENT_MASSES["O"]
                - adduct_parameters("[M+H]+")[2],
            )
    return tuple(deltas)


def neutral_mass(
    mz: NDArray[np.float64],
    adducts: list[str | None],
) -> NDArray[np.float64]:
    """Convert measured precursor m/z, retaining NaN for unknown adducts.

    Args:
      mz: Aligned measured precursor values.
      adducts: Bracketed adduct names in the same order.

    Returns:
      masses: Neutral monoisotopic masses in double precision.

    """
    return _convert(mz, adducts=adducts, names=_NAMES, parameters=adduct_parameters)


def precursor_mass(mass: float, *, adduct: str) -> float:
    """Return precursor m/z for a recognized adduct and neutral mass."""
    molecules, charge, delta, _ = adduct_parameters(adduct)
    return (molecules * mass + delta) / charge


def second_adduct_parameters(name: str) -> tuple[int, int, float]:
    """Decode the second engine's molecule count, charge and ion mass offset.

    Args:
      name: A recognized second-engine bracketed adduct name.

    Returns:
      molecules: Neutral molecules per ion.
      charge: Public table's absolute charge, including its [M-2H]- entry.
      delta: Offset retaining proton-first arithmetic and its electron corrections.

    """
    body, suffix = name[1:].split("]")
    coefficient, transfer = body.split("M")
    charge = 2 if name == "[M-2H]-" else int(suffix[:-1] or "1")
    if transfer:
        terms = _SECOND_TRANSFERS[transfer]
        element, count = terms[0]
        delta = _MASSES[element] * count
        for element, count in terms[1:]:
            delta += _MASSES[element] * count
    else:
        delta = -_MASSES["electron"] if suffix[-1] == "+" else _MASSES["electron"]
    return int(coefficient or "1"), charge, delta


def second_neutral_mass(
    mz: NDArray[np.float64],
    adducts: list[str | None],
) -> NDArray[np.float64]:
    """Convert the second engine's precursors with its complete distinct mass table.

    Args:
      mz: Measured precursor mass-to-charge values.
      adducts: Aligned bracketed names from the second engine's mass vocabulary.

    Returns:
      masses: Neutral masses, with NaN for unknown adduct names.

    """
    return _convert(
        mz,
        adducts=adducts,
        names=SECOND_NAMES,
        parameters=second_adduct_parameters,
    )


IDENTITY_RDKIT: Final = "2026.03.3"
"""Release for molecular identities, fingerprints, fragments and reactions."""


SIMULATOR_RDKIT: Final = "2025.03.6"
"""Release for simulator graphs and admission."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ElementGraph:
    """Complete element-only atoms and ordered undirected single bonds."""

    elements: NDArray[np.int16]
    bonds: NDArray[np.int32]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AnalogRecord:
    """Complete prepared intrinsic graph identities, counts and containment inputs."""

    smiles: str
    formula_input: str | None
    hashes: NDArray[np.int64]
    numbers: NDArray[np.int16]
    formula: NDArray[np.int16]
    core_smiles: str
    element_smiles: str
    query_smiles: str
    graph: ElementGraph | None
    core: ElementGraph | None
    core_mask: NDArray[np.bool_]


class AnalogTable:
    """Retain every input row and numeric graph for native structural scoring."""

    def __init__(self, directory: Path) -> None:
        manifest = json_loads(
            (directory / "manifest.json").read_text(),
            PreparationHeader,
        )
        if manifest["format_version"] != 1 or manifest["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Analog data requires format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.metadata = from_plain(
            parquet.read_table(directory / "molecules.parquet").to_pylist(),
            list[AnalogMetadata],
        )
        self.hashes = _load_array(
            directory / "hashes.npy",
            dtype=np.int64,
            dimensions=2,
        )
        self.numbers = _load_array(
            directory / "numbers.npy",
            dtype=np.int16,
            dimensions=2,
        )
        self.formula = _load_array(
            directory / "formula.npy",
            dtype=np.int16,
            dimensions=2,
        )
        count = len(self.metadata)
        if (
            self.hashes.shape != (count, 8)
            or self.numbers.shape != (count, 8)
            or self.formula.shape != (count, 10)
        ):
            raise ValueError(
                "Analog fixed arrays differ from the complete declared layout.",
            )
        self.graph = _GraphStore.load(directory, prefix="graph", rows=count)
        self.core = _GraphStore.load(directory, prefix="core", rows=count)
        self.core_mask = _load_array(
            directory / "core_mask.npy",
            dtype=np.bool_,
            dimensions=1,
        )
        if self.core_mask.shape != self.graph.elements.shape:
            raise ValueError(
                "Analog core mask differs from the full graph atom layout.",
            )
        self._positions: dict[str, int] = {}
        for index, row in enumerate(self.metadata):
            self._positions.setdefault(row["smiles"], index)

    def row(self, index: int) -> AnalogRecord:
        """Return every intrinsic field and complete graph in input row order.

        Args:
          index: Original prepared row, retaining duplicated structure positions.

        Returns:
          record: All identities, counts and numeric containment inputs for that row.

        """
        metadata = self.metadata[index]
        begin, end = (
            self.graph.atom_offsets.item(index),
            self.graph.atom_offsets.item(index + 1),
        )
        return AnalogRecord(
            **metadata,  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            hashes=cast("NDArray[np.int64]", self.hashes[index]),
            numbers=cast("NDArray[np.int16]", self.numbers[index]),
            formula=cast("NDArray[np.int16]", self.formula[index]),
            graph=self.graph.row(index) if metadata["element_smiles"] else None,
            core=self.core.row(index) if metadata["query_smiles"] else None,
            core_mask=self.core_mask[begin:end],
        )

    def for_smiles(self, smiles: str) -> AnalogRecord:
        """Require an explicitly prepared row for the requested structure."""
        return self.row(self._positions[smiles])


def formula_vector(formula: str | None) -> NDArray[np.int16]:
    """Parse all ten element counts, retaining a negative row for rejected formulas.

    Args:
      formula: Original molecular formula, retaining null and charge-only strings.

    Returns:
      counts: C/H/N/O/P/S/F/Cl/Br/I counts, or a complete negative row on failure.

    """
    elements = FORMULA_ELEMENTS
    invalid = np.full(10, -1, dtype=np.int16)
    if formula is None or not formula:
        return invalid
    text = formula.replace("+", "").replace("-", "")
    counts = [0] * 10
    position = 0
    for token in re.finditer(r"([A-Z][a-z]?)(\d*)", text):
        element, count = token.group(1, 2)
        if token.start() != position or element not in elements:
            return invalid
        position = token.end()
        counts[elements.index(element)] += int(count) if count else 1
    return np.array(counts, dtype=np.int16) if position == len(text) else invalid


@dataclass(frozen=True, slots=True, kw_only=True)
class Reaction:
    """A standard reaction SMARTS and its ordered elemental mass difference."""

    name: str
    smarts: str
    delta: tuple[tuple[str, int], ...]

    @property
    def mass(self) -> float:
        """Ordered elemental exact-mass difference of this reaction."""
        return sum(ELEMENT_MASSES[element] * count for element, count in self.delta)


def reaction_rules() -> tuple[Reaction, ...]:
    """Return the complete ordered public-best reaction vocabulary.

    Returns:
      rules: All addition and removal reactions, including distinct duplicate rules.

    """
    definitions = (
        ("hydroxyl_arom", "[c;H1:1]>>[c:1]O", (("O", 1),)),
        ("hydroxyl_aliph", "[C;H2,H3;!$(C=O):1]>>[C:1]O", (("O", 1),)),
        ("hydroxyl_ch", "[C;H1;!$(C=O):1]>>[C:1]O", (("O", 1),)),
        ("methyl_O", "[O;H1;!$(OC=O):1]>>[O:1]C", (("C", 1), ("H", 2))),
        ("methyl_N", "[N;H1,H2:1]>>[N:1]C", (("C", 1), ("H", 2))),
        ("methyl_c", "[c;H1:1]>>[c:1]C", (("C", 1), ("H", 2))),
        ("methoxy_arom", "[c;H1:1]>>[c:1]OC", (("C", 1), ("H", 2), ("O", 1))),
        ("acetyl_O", "[O;H1;!$(OC=O):1]>>[O:1]C(C)=O", (("C", 2), ("H", 2), ("O", 1))),
        (
            "acetyl_N",
            "[N;H1,H2;!$(NC=O):1]>>[N:1]C(C)=O",
            (("C", 2), ("H", 2), ("O", 1)),
        ),
        (
            "malonyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)CC(=O)O",
            (("C", 3), ("H", 2), ("O", 3)),
        ),
        (
            "hexose_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C1OC(CO)C(O)C(O)C1O",
            (("C", 6), ("H", 10), ("O", 5)),
        ),
        (
            "pentose_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C1OCC(O)C(O)C1O",
            (("C", 5), ("H", 8), ("O", 4)),
        ),
        (
            "deoxyhexose_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C1OC(C)C(O)C(O)C1O",
            (("C", 6), ("H", 10), ("O", 4)),
        ),
        (
            "glucuronyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O",
            (("C", 6), ("H", 8), ("O", 6)),
        ),
        (
            "hexose_C",
            "[c;H1:1]>>[c:1]C1OC(CO)C(O)C(O)C1O",
            (("C", 6), ("H", 10), ("O", 5)),
        ),
        ("sulfate_O", "[O;H1;!$(OC=O):1]>>[O:1]S(=O)(=O)O", (("S", 1), ("O", 3))),
        ("prenyl_c", "[c;H1:1]>>[c:1]CC=C(C)C", (("C", 5), ("H", 8))),
        ("prenyl_O", "[O;H1;!$(OC=O):1]>>[O:1]CC=C(C)C", (("C", 5), ("H", 8))),
        ("ethyl_O", "[O;H1;!$(OC=O):1]>>[O:1]CC", (("C", 2), ("H", 4))),
        ("hydrogenate_CC", "[C:1]=[C:2]>>[C:1][C:2]", (("H", 2),)),
        ("hydrogenate_CO", "[C:1]=[O:2]>>[C:1][O:2]", (("H", 2),)),
        ("dehydrogenate", "[C;H1,H2:1][C;H1,H2:2]>>[C:1]=[C:2]", (("H", -2),)),
        ("oxidize_CHOH", "[C;H1:1][O;H1:2]>>[C:1]=[O:2]", (("H", -2),)),
        ("carboxyl_c", "[c;H1:1]>>[c:1]C(=O)O", (("C", 1), ("O", 2))),
        ("carboxyl_CH3", "[C;H3:1]>>[C:1](=O)O", (("O", 2),)),
        ("amine_c", "[c;H1:1]>>[c:1]N", (("N", 1), ("H", 1))),
        ("chloro_c", "[c;H1:1]>>[c:1]Cl", (("Cl", 1), ("H", -1))),
        (
            "coumaroyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)cc1",
            (("C", 9), ("H", 6), ("O", 2)),
        ),
        (
            "feruloyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1",
            (("C", 10), ("H", 8), ("O", 3)),
        ),
        (
            "caffeoyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1",
            (("C", 9), ("H", 6), ("O", 3)),
        ),
        (
            "galloyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1",
            (("C", 7), ("H", 4), ("O", 4)),
        ),
        (
            "benzoyl_O",
            "[O;H1;!$(OC=O):1]>>[O:1]C(=O)c1ccccc1",
            (("C", 7), ("H", 4), ("O", 1)),
        ),
        ("demethyl_O", "[O:1][CH3]>>[O:1]", (("C", -1), ("H", -2))),
        ("demethyl_N", "[N:1][CH3]>>[N:1]", (("C", -1), ("H", -2))),
        ("dehydroxyl_arom", "[c:1][OH]>>[c:1]", (("O", -1),)),
        ("dehydroxyl_aliph", "[C;!$(C=O):1][OH]>>[C:1]", (("O", -1),)),
        ("deacetyl_O", "[O:1]C(=O)[CH3]>>[O:1]", (("C", -2), ("H", -2), ("O", -1))),
        (
            "dehexose",
            "[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]",
            (("C", -6), ("H", -10), ("O", -5)),
        ),
        (
            "dehexose2",
            "[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]",
            (("C", -6), ("H", -10), ("O", -5)),
        ),
        ("depentose", "[O:1]C1OCC(O)C(O)C1O>>[O:1]", (("C", -5), ("H", -8), ("O", -4))),
        (
            "dedeoxyhexose",
            "[O:1]C1OC(C)C(O)C(O)C1O>>[O:1]",
            (("C", -6), ("H", -10), ("O", -4)),
        ),
        ("demalonyl", "[O:1]C(=O)CC(=O)O>>[O:1]", (("C", -3), ("H", -2), ("O", -3))),
        ("desulfate", "[O:1]S(=O)(=O)O>>[O:1]", (("S", -1), ("O", -3))),
        ("deprenyl", "[c:1]CC=C(C)C>>[c:1]", (("C", -5), ("H", -8))),
        ("decarboxyl", "[C,c:1]C(=O)[OH]>>[C,c:1]", (("C", -1), ("O", -2))),
        ("demethoxy", "[c:1]O[CH3]>>[c:1]", (("C", -1), ("H", -2), ("O", -1))),
    )
    return tuple(
        Reaction(name=name, smarts=smarts, delta=delta)
        for name, smarts, delta in definitions
    )


def matching_plans(
    rules: Sequence[Reaction],
    delta: float,
    *,
    tolerance: float,
    max_reactions: int,
) -> list[tuple[int, ...]]:
    """Enumerate every eligible one/two-reaction plan in original encounter order.

    Args:
      rules: Complete ordered reaction vocabulary.
      delta: Requested neutral-mass difference from the parent.
      tolerance: Inclusive absolute mass tolerance.
      max_reactions: Most transformations per plan, including repeated ones.

    Returns:
      plans: All eligible single plans followed by pairs in combination order.

    """
    selected: list[tuple[int, ...]] = [
        (index,)
        for index, rule in enumerate(rules)
        if max_reactions >= 1 and abs(rule.mass - delta) <= tolerance
    ]
    if max_reactions >= 2:
        for first, second in combinations_with_replacement(range(len(rules)), 2):
            atoms = dict(rules[first].delta)
            for element, count in rules[second].delta:
                atoms[element] = atoms.get(element, 0) + count
            mass = sum(
                ELEMENT_MASSES[element] * count for element, count in atoms.items()
            )
            if abs(mass - delta) <= tolerance:
                selected.append((first, second))
    return selected


def v6_reaction_rules() -> tuple[Reaction, ...]:
    """Return every ordered rule of the distinct 71-transform vocabulary.

    Returns:
      rules: Scientific reaction declarations in their mass-plan encounter order.

    """
    with _THIS.with_name("v6_reactions.tsv").open(newline="") as handle:
        return tuple(
            Reaction(
                name=row["name"],
                smarts=row["smarts"],
                delta=tuple(
                    (element, int(count))
                    for item in row["delta"].split(";")
                    for element, count in [item.split("=")]
                ),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        )


class PoolMetadata(TypedDict):
    """The two ordered structure columns in both released candidate metadata files."""

    keys: list[str]
    smiles: list[str]


def read_pool_metadata(path: Path) -> PoolMetadata:
    """Read complete candidate keys and SMILES without importing asset classes.

    Args:
      path: Provided Coco or Bio metadata pickle, containing only scientific data.

    Returns:
      metadata: Both complete string columns in their original encounter order.

    """
    with path.open("rb") as handle:
        values = from_plain(
            cast(object, ScientificDataUnpickler(handle).load()),
            dict[str, object],
        )
    columns: dict[str, list[str]] = {}
    for name in ("keys", "smiles"):
        value = values[name]
        columns[name] = from_plain(
            cast(object, value.tolist()) if isinstance(value, np.ndarray) else value,
            list[str],
        )
    if len(columns["keys"]) != len(columns["smiles"]):
        raise ValueError("Candidate metadata columns have different lengths.")
    return {"keys": columns["keys"], "smiles": columns["smiles"]}


class ScientificDataUnpickler(pickle.Unpickler):
    """Restrict reconstruction to standard NumPy array and dtype data primitives."""

    @override
    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("_codecs", "encode"):
            return _latin1
        if (module, name) in {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        }:
            return cast(object, super().find_class(module, name))
        raise pickle.UnpicklingError(
            f"Candidate metadata requires forbidden global {module}.{name}.",
        )


DATASET_ROOT: Final = "/opt/scratch/datasets/casmi2026"
"""Resource root of every provided input and prepared asset."""


ADMISSION_DIR: Final = "native-chemistry/simulator-admission"
"""Prepared cheap simulator decisions, beneath the dataset root."""


ICE_ROOT_DIR: Final = "native-chemistry/ice-roots"
"""Prepared ICE roots, beneath the dataset root."""


GLACIER_ROOT_DIR: Final = "native-chemistry/glacier-roots"
"""Prepared GLACIER roots, beneath the dataset root."""


FRAGMENT_GAP_DIR: Final = "native-chemistry/fragment-gap"
"""Prepared fragment-gap graphs, beneath the dataset root."""


IDENTITY_DIR: Final = "native-chemistry/identities"
"""Prepared structure identities, beneath the dataset root."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class IntrinsicTable:
    """Complete ordered descriptor and loss-capability records for each input."""

    smiles: tuple[str, ...]
    records: NDArray[np.float32]
    capabilities: NDArray[np.uint8]
    record_names: tuple[str, ...]
    capability_names: tuple[str, ...]
    _positions: dict[str, int] = field(repr=False)

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read complete intrinsic records and their declared feature vocabularies.

        Args:
          directory: Generated molecules.parquet, numerical arrays and vocabulary files.

        Returns:
          table: Every input row and its complete declared molecular feature vectors.

        """
        smiles = tuple(
            from_plain(
                cast(
                    object,
                    parquet.read_table(directory / "molecules.parquet")
                    .column("smiles")
                    .to_pylist(),
                ),
                list[str],
            ),
        )
        records = cast("NDArray[np.float32]", np.load(directory / "records.npy"))
        capabilities = cast(
            "NDArray[np.uint8]",
            np.load(directory / "capabilities.npy"),
        )
        names = tuple((directory / "record_names.txt").read_text().splitlines())
        cap_names = tuple((directory / "capability_names.txt").read_text().splitlines())
        if names != record_fields() or records.shape != (len(smiles), len(names)):
            raise ValueError(
                "Intrinsic record layout differs from its declared vocabulary.",
            )
        if len(cap_names) != 22 or capabilities.shape != (len(smiles), len(cap_names)):
            raise ValueError("Capability layout differs from its declared vocabulary.")
        if records.dtype != np.float32 or capabilities.dtype != np.uint8:
            raise ValueError("Intrinsic data types differ from the prepared schema.")
        return cls(
            smiles=smiles,
            records=records,
            capabilities=capabilities,
            record_names=names,
            capability_names=cap_names,
            _positions={value: index for index, value in enumerate(smiles)},
        )

    def select(
        self,
        smiles: Sequence[str],
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """Select molecular records in the caller's exact order, keeping duplicates.

        Args:
          smiles: Exact structure strings already present in the generated cache.

        Returns:
          records: All34 intrinsic molecular descriptors in caller order.
          capabilities: All22 diagnostic-loss substructure counts in caller order.

        """
        rows = [self._positions[value] for value in smiles]
        return self.records[rows], self.capabilities[rows]

    def scaffold_keys(self, *, generic: bool = False) -> NDArray[np.int64]:
        """Reassemble the exact 48-bit scaffold identities from their float32 halves."""
        position = 2 if generic else 0
        return (self.records[:, position].astype(np.int64) << 24) | self.records[
            :,
            position + 1,
        ].astype(np.int64)


def record_fields() -> tuple[str, ...]:
    """Return the complete ordered intrinsic feature vocabulary.

    Returns:
      fields: Every descriptor column in the released ranker's declared order.

    """
    return (
        "scaf_hi",
        "scaf_lo",
        "gscaf_hi",
        "gscaf_lo",
        "n_ring",
        "n_arom_ring",
        "n_scaf_heavy",
        "frac_scaf",
        "charge_pos",
        "charge_neg",
        "quat_n",
        "basic_amine",
        "amide_n",
        "arom_n",
        "cooh",
        "so3h",
        "po3h",
        "phenol_oh",
        "alc_oh",
        "n_N",
        "n_O",
        "n_S",
        "n_P",
        "n_hal",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "n_rotb",
        "fsp3",
        "glycoside",
        "ester",
        "methoxy",
        "n_heavy",
    )


_VALENCE: Final = {"N": 3, "O": 2, "S": 2}


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FragnetAtoms:
    """Complete ordered standard-library primitives before mobile-H normalization."""

    symbols: tuple[str, ...]
    hydrogens: NDArray[np.int64]
    degree: NDArray[np.int64]
    charge: NDArray[np.int64]
    aromatic: NDArray[np.int64]
    bonds: NDArray[np.int64]
    bond_aromatic: NDArray[np.bool_]
    bond_ring: NDArray[np.bool_]


def molecular_record(
    atoms: FragnetAtoms | None,
    *,
    masses: Mapping[str, float],
) -> FragnetRecord | None:
    """Compute canonical hydrogens, masses, atom kinds and complete graph fields.

    Args:
      atoms: Complete standard molecular primitives, or explicit parse failure.
      masses: Complete pinned fixed-mass vocabulary, independent of isotope labels.

    Returns:
      record: Every source molecular field, or a declared unsupported/empty molecule.

    """
    if (
        atoms is None
        or not atoms.symbols
        or any(symbol not in masses for symbol in atoms.symbols)
    ):
        return None
    frame = _properties(atoms)
    vocabulary = {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "P": 4,
        "F": 5,
        "Cl": 5,
        "Br": 5,
        "I": 5,
    }
    codes = [vocabulary.get(symbol, 6) for symbol in frame.symbols]
    hydrogens = _hydrogens(frame, codes=codes)
    amass = cast(
        "NDArray[np.float64]",
        np.array([masses[symbol] for symbol in frame.symbols])
        + np.array(hydrogens, dtype=np.int64) * masses["H"],
    )
    extended = {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "P": 4,
        "F": 5,
        "Cl": 6,
        "Br": 7,
        "I": 8,
    }
    return {
        "amass": amass,
        "kind": np.array(_kinds(frame, hydrogens=hydrogens), dtype=np.int8),
        "el": np.array(codes, dtype=np.int8),
        "nh": np.array(hydrogens, dtype=np.int8),
        "bonds": atoms.bonds.astype(np.int32),
        "bring": np.where(
            atoms.bond_ring,
            np.where(atoms.bond_aromatic, 2, 1),
            0,
        ).astype(np.int8),
        "mass": float(amass.sum()),
        "el10": np.array(
            [extended.get(symbol, 9) for symbol in frame.symbols],
            dtype=np.uint8,
        ),
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class _Properties:
    symbols: tuple[str, ...]
    hydrogens: tuple[int, ...]
    degree: tuple[int, ...]
    charge: tuple[int, ...]
    aromatic: tuple[int, ...]
    neighbors: tuple[tuple[int, ...], ...]
    bonds: tuple[tuple[int, int], ...]
    aromatic_bonds: tuple[bool, ...]
    unsaturated: tuple[bool, ...]
    mobile: tuple[bool, ...]


def normalize_request(
    request: ReactionRequest | ReactionHeader,
    *,
    default_products: int = 60,
) -> ReactionHeader:
    """Materialize every request default before lookup or data publication.

    Args:
      request: Complete or partial explicit scientific request.
      default_products: Final derivation cap for this vocabulary.

    Returns:
      header: Every request field, including all enumeration budgets.

    """
    return {
        "smiles": request["smiles"],
        "delta": request["delta"],
        "operation": request.get("operation", "derive"),
        "rule": request.get("rule", -1),
        "tolerance": request.get("tolerance", 0.003),
        "max_steps": request.get("max_steps", 2),
        "max_products": request.get(
            "max_products",
            64 if request.get("operation", "derive") == "apply" else default_products,
        ),
        "allow_isomerization": request.get("allow_isomerization", True),
        "application_budget": request.get("application_budget", 800),
        "stage_parents": request.get("stage_parents", 24),
        "first_products": request.get("first_products", 32),
        "later_products": request.get("later_products", 16),
    }


def reaction_key(request: ReactionHeader) -> str:
    """Identify the complete request without rounding its floating mass geometry."""
    fields: dict[str, object] = dict(request)
    fields["delta"] = float(request["delta"]).hex()
    fields["tolerance"] = float(request["tolerance"]).hex()
    return json_dumps(fields, sort_keys=True, separators=(",", ":"))


class ReactionTable:
    """Keep zero-product headers distinct from missing requested computations."""

    def __init__(self, directory: Path) -> None:
        header = json_loads((directory / "manifest.json").read_text(), _Header)
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Reaction data requires format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.default_products = header["default_products"]
        self.requests = tuple(
            from_plain(
                parquet.read_table(directory / "requests.parquet").to_pylist(),
                list[ReactionHeader],
            ),
        )
        self.responses = tuple(
            from_plain(
                parquet.read_table(directory / "results.parquet").to_pylist(),
                list[_Response],
            ),
        )
        self.rules = tuple(
            from_plain(
                parquet.read_table(directory / "rules.parquet")
                .column("name")
                .to_pylist(),
                list[str],
            ),
        )
        self._products: dict[str, tuple[ReactionProduct, ...]] = {}
        if len(self.requests) != len(self.responses):
            raise ValueError("Reaction results differ from their request headers.")
        for position, request in enumerate(self.requests):
            response = self.responses[position]
            if (
                request["smiles"] != response["input_smiles"]
                or len(response["products"]) != len(response["plans"])
                or any(
                    index < 0 or index >= len(self.rules)
                    for plan in response["plans"] + response["matches"]
                    for index in plan
                )
            ):
                raise ValueError("Reaction results differ from their request headers.")
            products: tuple[ReactionProduct, ...] = tuple(
                {
                    "raw": raw,
                    "combo": tuple(self.rules[index] for index in plan) or None,
                }
                for raw, plan in (
                    (raw, response["plans"][index])
                    for index, raw in enumerate(response["products"])
                )
            )
            self._products.setdefault(reaction_key(request), products)

    def products(
        self,
        request: ReactionRequest | ReactionHeader,
    ) -> tuple[ReactionProduct, ...]:
        """Require the complete request and return every product in encounter order."""
        return self._products[
            reaction_key(
                normalize_request(request, default_products=self.default_products),
            )
        ]


class _Header(PreparationHeader):
    default_products: int


class _Response(TypedDict):
    input_smiles: str
    products: list[str]
    plans: list[list[int]]
    matches: list[list[int]]
    applications: str


def _convert(
    mz: NDArray[np.float64],
    *,
    adducts: Sequence[str | None],
    names: tuple[str, ...],
    parameters: Callable[[str], tuple[int, int, float] | tuple[int, int, float, int]],
) -> NDArray[np.float64]:
    labels = np.asarray(adducts, dtype=object)
    result = np.full(len(mz), np.nan)
    for name in names:
        molecules, charge, delta = parameters(name)[:3]
        selected = np.equal(labels, name)
        result[selected] = (mz[selected] * charge - delta) / molecules
    return result


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _GraphStore:
    elements: NDArray[np.int16]
    bonds: NDArray[np.int32]
    atom_offsets: NDArray[np.int64]
    bond_offsets: NDArray[np.int64]

    @classmethod
    def load(cls, directory: Path, *, prefix: str, rows: int) -> _GraphStore:
        elements = _load_array(
            directory / f"{prefix}_elements.npy",
            dtype=np.int16,
            dimensions=1,
        )
        bonds = _load_array(
            directory / f"{prefix}_bonds.npy",
            dtype=np.int32,
            dimensions=2,
        )
        atom_offsets = _load_array(
            directory / f"{prefix}_atom_off.npy",
            dtype=np.int64,
            dimensions=1,
        )
        bond_offsets = _load_array(
            directory / f"{prefix}_bond_off.npy",
            dtype=np.int64,
            dimensions=1,
        )
        if bonds.shape[1] != 2:
            raise ValueError(
                "Analog bonds require two atom indices per undirected edge.",
            )
        for offsets, count in (
            (atom_offsets, len(elements)),
            (bond_offsets, len(bonds)),
        ):
            if (
                offsets.shape != (rows + 1,)
                or offsets.item(0) != 0
                or offsets.item(-1) != count
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError(
                    "Analog ragged offsets differ from the complete graph layout.",
                )
        return cls(
            elements=elements,
            bonds=bonds,
            atom_offsets=atom_offsets,
            bond_offsets=bond_offsets,
        )

    def row(self, index: int) -> ElementGraph:
        atom_begin, atom_end = (
            self.atom_offsets.item(index),
            self.atom_offsets.item(index + 1),
        )
        bond_begin, bond_end = (
            self.bond_offsets.item(index),
            self.bond_offsets.item(index + 1),
        )
        return ElementGraph(
            elements=self.elements[atom_begin:atom_end],
            bonds=self.bonds[bond_begin:bond_end],
        )


def _load_array[S: np.generic](
    path: Path,
    *,
    dtype: type[S],
    dimensions: int,
) -> NDArray[S]:
    value = load_array(path, dtype, mmap=True)
    if value.ndim != dimensions:
        raise ValueError(f"Analog array {path.name} has an invalid rank.")
    return value


def _latin1(value: str, encoding: str) -> bytes:
    if encoding != "latin1":
        raise pickle.UnpicklingError("Legacy NumPy bytes require latin1 encoding.")
    return value.encode("latin1")


def _properties(atoms: FragnetAtoms) -> _Properties:
    hydrogen = from_plain(cast(object, atoms.hydrogens.tolist()), list[int])
    degree = from_plain(cast(object, atoms.degree.tolist()), list[int])
    charge = from_plain(cast(object, atoms.charge.tolist()), list[int])
    aromatic = from_plain(cast(object, atoms.aromatic.tolist()), tuple[int, ...])
    bonds = from_plain(cast(object, atoms.bonds.tolist()), tuple[tuple[int, int], ...])
    neighbors: list[list[int]] = [[] for _ in atoms.symbols]
    for left, right in bonds:
        neighbors[left].append(right)
        neighbors[right].append(left)
    for index, symbol in enumerate(atoms.symbols):
        if (
            charge[index] == -1
            and symbol in _VALENCE
            and hydrogen[index] + degree[index] < _VALENCE[symbol]
        ):
            hydrogen[index] += 1
            charge[index] = 0
    return _Properties(
        symbols=atoms.symbols,
        hydrogens=tuple(hydrogen),
        degree=tuple(degree),
        charge=tuple(charge),
        aromatic=aromatic,
        neighbors=tuple(tuple(values) for values in neighbors),
        bonds=bonds,
        aromatic_bonds=from_plain(
            cast(object, atoms.bond_aromatic.tolist()),
            tuple[bool, ...],
        ),
        unsaturated=tuple(
            symbol == "C"
            and 4 - degree[index] - hydrogen[index] - abs(charge[index]) >= 1
            for index, symbol in enumerate(atoms.symbols)
        ),
        mobile=tuple(
            symbol in _VALENCE and charge[index] == 0
            for index, symbol in enumerate(atoms.symbols)
        ),
    )


def _hydrogens(frame: _Properties, *, codes: Sequence[int]) -> list[int]:
    hydrogen = list(frame.hydrogens)
    groups = [group for group in _mobile_groups(frame) if len(group) > 1]
    if not groups:
        return hydrogen
    labels = _labels(frame, codes=codes)
    priority = ("N", "S", "O")
    for group in groups:
        capacity = {
            index: max(0, _VALENCE[frame.symbols[index]] - frame.degree[index])
            for index in group
        }
        order = sorted(
            group,
            key=lambda index: (
                priority.index(frame.symbols[index]),
                -capacity[index],
                not any(
                    frame.aromatic[neighbor] and frame.symbols[neighbor] == "C"
                    for neighbor in frame.neighbors[index]
                ),
                labels[index],
            ),
        )
        remaining = sum(frame.hydrogens[index] for index in group)
        for index in order:
            hydrogen[index] = min(capacity[index], remaining)
            remaining -= hydrogen[index]
        if remaining > 0:
            for index in group:
                hydrogen[index] = frame.hydrogens[index]
    return hydrogen


def _labels(frame: _Properties, *, codes: Sequence[int]) -> list[int]:
    """Retain every integer-tuple label after all five refinement iterations."""
    labels = [
        hash(
            (
                codes[index],
                frame.hydrogens[index] if frame.symbols[index] == "C" else 0,
                frame.aromatic[index],
                len(frame.neighbors[index]),
            ),
        )
        for index in range(len(codes))
    ]
    for _ in range(5):
        labels = [
            hash(
                (
                    labels[index],
                    tuple(
                        sorted(labels[neighbor] for neighbor in frame.neighbors[index]),
                    ),
                ),
            )
            for index in range(len(labels))
        ]
    return labels


def _mobile_groups(frame: _Properties) -> list[list[int]]:
    links: list[set[int]] = [set() for _ in frame.symbols]
    aromatic: list[set[int]] = [set() for _ in frame.symbols]
    for index, unsaturated in enumerate(frame.unsaturated):
        if unsaturated:
            _connect(
                links,
                members=[
                    neighbor
                    for neighbor in frame.neighbors[index]
                    if frame.mobile[neighbor]
                ],
            )
    for index, (left, right) in enumerate(frame.bonds):
        if (
            frame.mobile[left]
            and frame.mobile[right]
            and frame.aromatic[left]
            and frame.aromatic[right]
        ):
            _connect(links, members=[left, right])
        if frame.aromatic_bonds[index]:
            _connect(aromatic, members=[left, right])
    for group in _records_components(aromatic, members=set(range(len(frame.symbols)))):
        _connect(
            links,
            members=[
                index
                for index in group
                if frame.mobile[index]
                and frame.aromatic[index]
                and frame.symbols[index] == "N"
            ],
        )
    for index, mobile in enumerate(frame.mobile):
        if not mobile:
            continue
        for first in frame.neighbors[index]:
            if not frame.unsaturated[first]:
                continue
            for middle in frame.neighbors[first]:
                if not frame.unsaturated[middle]:
                    continue
                for last in frame.neighbors[middle]:
                    if last != first and frame.unsaturated[last]:
                        _connect(
                            links,
                            members=[
                                index,
                                *(
                                    neighbor
                                    for neighbor in frame.neighbors[last]
                                    if neighbor != index and frame.mobile[neighbor]
                                ),
                            ],
                        )
    return _records_components(
        links,
        members={index for index, mobile in enumerate(frame.mobile) if mobile},
    )


def _connect(neighbors: list[set[int]], *, members: Sequence[int]) -> None:
    for index in range(1, len(members)):
        left, right = members[0], members[index]
        neighbors[left].add(right)
        neighbors[right].add(left)


def _records_components(
    neighbors: Sequence[set[int]],
    *,
    members: set[int],
) -> list[list[int]]:
    pending = set(members)
    result: list[list[int]] = []
    while pending:
        stack = [min(pending)]
        pending.remove(stack[0])
        group: list[int] = []
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in neighbors[index]:
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        result.append(sorted(group))
    return result


def _kinds(frame: _Properties, *, hydrogens: Sequence[int]) -> list[int]:
    terminal = [
        symbol in ("O", "S")
        and frame.degree[index] == 1
        and hydrogens[index] == 0
        and frame.charge[index] == 0
        for index, symbol in enumerate(frame.symbols)
    ]
    kinds: list[int] = []
    for index, symbol in enumerate(frame.symbols):
        if symbol == "C":
            value = (
                4
                if any(terminal[neighbor] for neighbor in frame.neighbors[index])
                else 3
                if frame.aromatic[index]
                else 2
                if frame.unsaturated[index]
                else 1
                if sum(
                    frame.symbols[neighbor] == "O"
                    for neighbor in frame.neighbors[index]
                )
                >= 2
                else 0
            )
        elif symbol == "N":
            value = 5 if frame.aromatic[index] else 6
        elif symbol == "O":
            value = 8 if frame.aromatic[index] else 9 if frame.degree[index] <= 1 else 7
        else:
            value = {"S": 10, "P": 11}.get(symbol, 12)
        kinds.append(value)
    return kinds
