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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
