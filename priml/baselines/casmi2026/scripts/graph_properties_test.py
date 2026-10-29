"""Checks for the graph_properties module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from priml.baselines.casmi2026.scripts.graph_properties import atom_properties


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rdkit import Chem


class _Atom:
    def __init__(
        self,
        symbol: str,
        *,
        isotope: int,
        implicit: int,
        explicit: int,
    ) -> None:
        self.GetSymbol: Callable[[], str] = lambda: symbol
        self.GetIsotope: Callable[[], int] = lambda: isotope
        self.GetNumImplicitHs: Callable[[], int] = lambda: implicit
        self.GetNumExplicitHs: Callable[[], int] = lambda: explicit


class _Table:
    def __init__(self, mass: float) -> None:
        self.mass = mass
        self.calls: list[tuple[str, int]] = []
        self.GetMassForIsotope: Callable[[str, int], float] = self._isotope_mass

    def _isotope_mass(self, symbol: str, isotope: int) -> float:
        self.calls.append((symbol, isotope))
        return self.mass


def _atoms(*atoms: _Atom) -> Sequence[Chem.Atom]:
    return cast("Sequence[Chem.Atom]", atoms)


def test_isotope_formula_hydrogen_and_mass_properties() -> None:
    elements = ("C", "N", "H", "O")
    masses = np.array([12.0, 14.0, 1.0, 16.0])
    table = _Table(13.5)
    actual = atom_properties(
        _atoms(
            _Atom("C", isotope=13, implicit=0, explicit=3),
            _Atom("N", isotope=0, implicit=3, explicit=0),
        ),
        formula="CH6N+",
        elements=elements,
        masses=masses,
        periodic_table=cast("Chem.PeriodicTable", table),
    )
    # The formula omits the count of C and N; H6 is parsed as six.
    np.testing.assert_array_equal(actual.formula, [1.0, 1.0, 6.0, 0.0])
    np.testing.assert_array_equal(actual.elements, [0, 1])
    np.testing.assert_array_equal(actual.hydrogens, [3, 3])
    # Only the isotopically labelled carbon consults the periodic table.
    np.testing.assert_array_equal(actual.masses, [13.5, 14.0])
    np.testing.assert_array_equal(actual.hydrogenated_masses, [16.5, 17.0])
    np.testing.assert_array_equal(
        actual.atom_formulas,
        [[1.0, 0.0, 3.0, 0.0], [0.0, 1.0, 3.0, 0.0]],
    )
    assert actual.mass == 33.5
    assert actual.total_hydrogens == 6
    assert table.calls == [("C", 13)]
    assert actual.elements.dtype == actual.hydrogens.dtype == np.int64


def test_repeated_elements_accumulate_and_a_bare_hydrogen_atom_counts() -> None:
    actual = atom_properties(
        _atoms(
            _Atom("C", isotope=0, implicit=3, explicit=0),
            _Atom("C", isotope=0, implicit=3, explicit=0),
            _Atom("H", isotope=0, implicit=0, explicit=0),
        ),
        formula="C2H7",
        elements=("C", "H"),
        masses=np.array([12.0, 1.0]),
        periodic_table=cast("Chem.PeriodicTable", _Table(99.0)),
    )
    np.testing.assert_array_equal(actual.formula, [2.0, 7.0])
    np.testing.assert_array_equal(actual.atom_formulas, [[1, 3], [1, 3], [0, 1]])
    np.testing.assert_array_equal(actual.elements, [0, 0, 1])
    np.testing.assert_array_equal(actual.masses, [12.0, 12.0, 1.0])
    np.testing.assert_array_equal(actual.hydrogenated_masses, [15.0, 15.0, 1.0])
    assert actual.mass == 31.0
    assert actual.total_hydrogens == 6


def test_zero_atoms_keep_every_element_axis() -> None:
    actual = atom_properties(
        _atoms(),
        formula="",
        elements=("C", "H"),
        masses=np.array([12.0, 1.0]),
        periodic_table=cast("Chem.PeriodicTable", _Table(13.0)),
    )
    assert actual.formula.tolist() == [0.0, 0.0]
    assert actual.atom_formulas.shape == (0, 2)
    assert actual.elements.shape == actual.hydrogens.shape == (0,)
    assert actual.masses.shape == actual.hydrogenated_masses.shape == (0,)
    assert actual.mass == 0.0
    assert actual.total_hydrogens == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
