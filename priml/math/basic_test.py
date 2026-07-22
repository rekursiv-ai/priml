from __future__ import annotations

import numpy as np
import pytest
import torch

from priml.math.basic import (
    ceil_multiple,
    factors,
    floor_multiple,
)


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
