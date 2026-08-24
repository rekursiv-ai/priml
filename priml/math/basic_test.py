from __future__ import annotations

from typing import cast

import math

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.math.basic import (
    argsort,
    broadcast_sequences,
    ceil_div,
    ceil_multiple,
    factors,
    floor_multiple,
)
from priml.math.custom_types import Tensorable


def test_factors_small_numbers() -> None:
    assert factors(1) == (1,)
    assert factors(6) == (1, 2, 3, 6)
    assert factors(12) == (1, 2, 3, 4, 6, 12)


def test_factors_rejects_non_positive() -> None:
    """factors(0) and factors(-n) are undefined; they must raise, not return ()."""
    with pytest.raises(ValueError, match="positive integer"):
        factors(0)
    with pytest.raises(ValueError, match="positive integer"):
        factors(-6)


def test_multiple_avoids_float_drift() -> None:
    """An integer multiple must land on the exact grid despite float error.

    0.1 * 30 is 2.9999998 in float32; ``multiple * ceil(x / multiple)`` with
    ``multiple=1`` gives ceil(2.9999998)=3, not the floored-low 2 the integer
    ``(x + m - 1) // m`` trick would produce.
    """
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32) * 30
    torch.testing.assert_close(ceil_multiple(x, 1), torch.tensor([3, 6, 9]))
    torch.testing.assert_close(floor_multiple(x, 1), torch.tensor([3, 6, 9]))


def test_multiple_is_idempotent_on_exact_multiples() -> None:
    """A value already on the grid is returned unchanged (not bumped up)."""
    assert ceil_multiple(8.0, 4) == 8
    assert floor_multiple(8.0, 4) == 8
    assert ceil_multiple(4.0, 4) == 4


def test_ceil_multiple_fractional_rounds_up() -> None:
    """ceil_multiple is a true ceiling: 4.4 -> 8 (smallest multiple of 4 >= 4.4)."""
    assert ceil_multiple(4.4, 4) == 8
    assert floor_multiple(4.4, 4) == 4


def test_multiple_handles_negatives() -> None:
    """Ceiling/floor toward +inf/-inf, not toward zero."""
    assert ceil_multiple(-3.0, 4) == 0
    assert floor_multiple(-3.0, 4) == -4


def test_factors_prime() -> None:
    assert factors(7) == (1, 7)
    assert factors(13) == (1, 13)


def test_factors_large() -> None:
    assert factors(100) == (1, 2, 4, 5, 10, 20, 25, 50, 100)


def test_ceil_multiple_none() -> None:
    assert ceil_multiple(5, None) == 5
    assert ceil_multiple(5.5, None) == 5.5


def test_ceil_multiple_int() -> None:
    assert ceil_multiple(5, 3) == 6
    assert ceil_multiple(6, 3) == 6
    assert ceil_multiple(7, 3) == 9


def test_ceil_multiple_float() -> None:
    result = ceil_multiple(5.5, 2.0)
    assert result == 6.0
    assert isinstance(result, float)


def test_ceil_multiple_numpy() -> None:
    x = np.array([5, 7, 10])
    result = ceil_multiple(x, 3)
    np.testing.assert_array_equal(result, np.array([6, 9, 12]))


def test_ceil_multiple_torch() -> None:
    x = torch.tensor([5, 7, 10])
    result = ceil_multiple(x, 3)
    torch.testing.assert_close(result, torch.tensor([6, 9, 12]))


def test_floor_multiple_none() -> None:
    assert floor_multiple(5, None) == 5
    assert floor_multiple(5.5, None) == 5.5


def test_floor_multiple_int() -> None:
    assert floor_multiple(5, 3) == 3
    assert floor_multiple(6, 3) == 6
    assert floor_multiple(8, 3) == 6


def test_floor_multiple_float() -> None:
    result = floor_multiple(5.5, 2.0)
    assert result == 4.0
    assert isinstance(result, float)


def test_floor_multiple_numpy() -> None:
    x = np.array([5, 7, 10])
    result = floor_multiple(x, 3)
    np.testing.assert_array_equal(result, np.array([3, 6, 9]))


def test_floor_multiple_torch() -> None:
    x = torch.tensor([5, 7, 10])
    result = floor_multiple(x, 3)
    torch.testing.assert_close(result, torch.tensor([3, 6, 9]))


def test_ceil_div_matches_the_true_ceiling_for_every_sign() -> None:
    """The docstring promises a ceiling, so it must hold for negative divisors.

    ``(x + y - 1) // y`` silently disagrees there: 4 instead of 3 for
    ``(-5, -2)``, and -1 instead of -2 for ``(5, -2)``.
    """
    for x in range(-12, 13):
        for y in (-7, -3, -2, -1, 1, 2, 3, 7):
            assert ceil_div(x, y) == math.ceil(x / y), f"{x=} {y=}"


def test_argsort_is_stable_and_returns_indices() -> None:
    """Ties keep input order, which is what "stable" buys a caller."""
    assert argsort([3, 1, 2]) == [1, 2, 0]
    assert argsort([3, 1, 2], descending=True) == [0, 2, 1]
    # Equal values keep their original relative order, ascending or not.
    assert argsort([1, 0, 1, 0]) == [1, 3, 0, 2]
    assert argsort([]) == []


def test_broadcast_sequences_pairs_scalars_against_sequences() -> None:
    """Every RoPE axis count is paired with a base through this.

    A scalar is stretched to the other argument's length; equal lengths pass
    through; a length-1 sequence broadcasts like a scalar.
    """
    assert broadcast_sequences(1, [10, 20]) == ([1, 1], [10, 20])
    assert broadcast_sequences([1, 2], 10) == ([1, 2], [10, 10])
    assert broadcast_sequences([1], [10, 20]) == ([1, 1], [10, 20])
    assert broadcast_sequences(1, 10) == ([1], [10])
    # Three arguments broadcast together, which mesh_arange relies on.
    assert broadcast_sequences(0, [2, 3], 1) == ([0, 0], [2, 3], [1, 1])


def test_broadcast_sequences_rejects_incompatible_lengths() -> None:
    """Mismatched lengths are a caller error, not a truncation."""
    with pytest.raises(ValueError, match="Incompatible lengths"):
        _ = broadcast_sequences([1, 2], [10, 20, 30])


def test_multiple_keeps_the_tensor_dtype_it_was_given() -> None:
    """A float ``multiple`` must not pin the result to float32.

    The Tensor branch hardcoded it, so a float64 input came back one ulp short
    of a value already on the grid -- the same defect
    ``test_multiple_is_exact_beyond_float53`` pinned for the scalar path.
    """
    big = torch.tensor([2.0**40 + 1], dtype=torch.float64)
    floored = floor_multiple(big, 1.0)
    ceiled = ceil_multiple(big, 1.0)
    assert isinstance(floored, Tensor)
    assert isinstance(ceiled, Tensor)
    assert floored.item() == 2.0**40 + 1
    assert ceiled.item() == 2.0**40 + 1
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        scaled = ceil_multiple(torch.ones(3, dtype=dtype), 2.0)
        assert isinstance(scaled, Tensor)
        assert scaled.dtype == dtype
    # An integer multiple still yields an integer result.
    as_int = ceil_multiple(torch.ones(3, dtype=torch.float64), 2)
    assert isinstance(as_int, Tensor)
    assert as_int.dtype == torch.int64


def test_multiple_keeps_the_numpy_dtype_it_was_given() -> None:
    """The numpy branch had the same width bug the Tensor branch was fixed for.

    ``astype(type(multiple))`` is float64 for every float ``multiple``, so a
    float32 array came back float64 -- doubling its memory and changing what a
    downstream kernel dispatches to.
    """
    for dtype in (np.float16, np.float32, np.float64):
        up = ceil_multiple(np.ones(3, dtype=dtype), 2.0)
        down = floor_multiple(np.ones(3, dtype=dtype), 2.0)
        assert isinstance(up, np.ndarray)
        assert isinstance(down, np.ndarray)
        assert up.dtype == dtype
        assert down.dtype == dtype
    # An integer multiple still yields an integer result.
    as_int = ceil_multiple(np.ones(3, dtype=np.float32), 2)
    assert isinstance(as_int, np.ndarray)
    assert as_int.dtype == np.int64


def test_a_float_grid_promotes_an_integer_input() -> None:
    """The result must land ON the grid, which an integer cast destroys.

    Casting back to the input's own dtype returned 2 for
    ``ceil_multiple(ones(int32), 2.5)`` -- not a multiple of 2.5 at all. Only
    an INTEGER multiple guarantees an integral result.
    """
    for dtype in (np.int32, np.int64):
        got = ceil_multiple(np.ones(3, dtype=dtype), 2.5)
        assert isinstance(got, np.ndarray)
        assert got.tolist() == [2.5, 2.5, 2.5]
    for dtype in (torch.int32, torch.int64):
        got_t = ceil_multiple(torch.ones(3, dtype=dtype), 2.5)
        assert isinstance(got_t, Tensor)
        assert got_t.tolist() == [2.5, 2.5, 2.5]
    # A float input keeps its own width rather than being widened.
    narrow = ceil_multiple(torch.ones(3, dtype=torch.float32), 2.5)
    assert isinstance(narrow, Tensor)
    assert narrow.dtype == torch.float32


def test_multiple_rejects_a_type_it_cannot_scale() -> None:
    """A Sequence is ``Tensorable``, so this is caller input and must raise."""
    with pytest.raises(TypeError, match="Unsupported scalar type"):
        _ = ceil_multiple(cast("Tensorable", "nope"), 2)


def test_multiple_is_exact_beyond_float53() -> None:
    """Integer inputs above 2**53 must round exactly, not through float.

    ``float(2**53 + 1)`` is ``2**53``, so routing the scalar path through
    ``float(x)`` returned the input minus one for a value already on the grid
    -- breaking idempotence precisely where an exact answer is cheapest.
    """
    big = 2**53 + 1
    assert ceil_multiple(big, 1) == big
    assert floor_multiple(big, 1) == big
    assert ceil_multiple(big, 2) == big + 1
    assert floor_multiple(big, 2) == big - 1


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
