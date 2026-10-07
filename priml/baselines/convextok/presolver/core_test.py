"""Kernels the presolve pipeline cannot drive into every branch.

``_sorted_find`` is PSLP's search within a sorted row; its one caller always
searches for an entry the row holds, so its not-found results are tested here.
"""

import numpy as np
import pytest

from priml.baselines.convextok.presolver import core


@pytest.mark.parametrize("length", [5, 20], ids=["linear", "binary"])
@pytest.mark.parametrize("target", [4, 9, 10, 18, 100])
def test_sorted_find_matches_a_linear_search(length: int, target: int) -> None:
    values = list(range(0, 3 * (length + 2), 3))
    first = 2
    window = values[first : first + length]
    expected = window.index(target) if target in window else -1
    arr = np.array(values, np.int32)
    assert core._sorted_find(arr, first, length, target) == expected


def test_new_core_retains_input_arrays_and_builds_state() -> None:
    coefficients = np.array([2.0, 3.0], np.float64)
    columns = np.array([0, 1], np.int32)
    starts = np.array([0, 2], np.int32)
    state = core.new_core(
        (coefficients, columns, starts),
        2,
        (np.array([-np.inf]), np.array([4.0])),
        (np.array([0.0, -1.0]), np.array([1.0, 2.0])),
        np.array([0.25, -0.5]),
    )
    core.attach_transpose(
        state,
        np.array([0, 1], np.int32),
        np.array([0, 1, 2], np.int32),
        np.array([1, 2, 2], np.int32),
        2,
    )
    fields: dict[str, object] = dict(
        zip(core.SNAPSHOT_FIELDS, core.snapshot(state), strict=True),
    )

    assert (fields["m"], fields["n"]) == (1, 2)
    expected_arrays = (
        ("A_x", coefficients),
        ("A_i", columns),
        ("A_start", starts),
        ("A_end", np.array([2, 2], np.int32)),
        ("lhs", np.array([-np.inf])),
        ("rhs", np.array([4.0])),
        ("lb", np.array([0.0, -1.0])),
        ("ub", np.array([1.0, 2.0])),
        ("c", np.array([0.25, -0.5])),
    )
    for name, expected in expected_arrays:
        actual = fields[name]
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(actual, expected), name


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
