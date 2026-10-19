"""Tests for the typed numpy helpers."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.craftax.lib.arrays import int_rows, ints, typed


def test_typed_hands_back_an_array_of_its_kind_itself() -> None:
    values = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert typed(values, np.uint8) is values


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.arange(3, dtype=np.int16), "expected an array of uint8, got array"),
        ([1, 2, 3], "expected an array of uint8, got [1, 2, 3]"),
    ],
)
def test_typed_refuses_another_element_type_or_a_non_array(
    values: object,
    message: str,
) -> None:
    with pytest.raises(TypeError) as error:
        typed(values, np.uint8)
    assert str(error.value).startswith(message)


def test_typed_cuts_a_long_repr_to_eighty_characters() -> None:
    with pytest.raises(TypeError) as error:
        typed(list(range(100)), np.int64)
    shown = str(error.value).removeprefix("expected an array of int64, got ")
    assert len(shown) == 80


def test_ints_reads_each_element_as_a_python_int() -> None:
    values = np.array([3, -1, 7], dtype=np.int16)
    elements = ints(values)
    assert elements == [3, -1, 7]
    assert all(type(e) is int for e in elements)


def test_int_rows_reads_each_row_as_a_tuple_of_ints() -> None:
    table = np.arange(6, dtype=np.int64).reshape(2, 3)
    rows = int_rows(table)
    assert rows == [(0, 1, 2), (3, 4, 5)]
    assert all(type(v) is int for row in rows for v in row)
    assert int_rows(np.zeros((0, 4), np.int32)) == []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
