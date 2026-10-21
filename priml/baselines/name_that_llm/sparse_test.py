"""Keep stored order through every sparse operation the lexical branch uses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.name_that_llm.sparse import CsrMatrix, hstack


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def test_dense_rows_store_nonzeros_in_row_major_order() -> None:
    actual = CsrMatrix.from_dense(np.array([[0, 1.5, -2], [3, 0, 0]]))
    _assert_stored(
        actual,
        data=[1.5, -2, 3],
        indices=[1, 2, 0],
        indptr=[0, 2, 3],
        shape=(2, 3),
    )


def test_sorting_orders_each_row_by_column_with_its_values() -> None:
    _assert_stored(
        _unsorted().sorted_indices(),
        data=[1, 3, 4, 5, 6],
        indices=[0, 2, 0, 1, 3],
        indptr=[0, 2, 2, 5],
        shape=(3, 4),
    )


def test_row_selection_keeps_each_row_stored_order() -> None:
    _assert_stored(
        _unsorted().take_rows(np.array([2, 1, 0])),
        data=[6, 5, 4, 3, 1],
        indices=[3, 1, 0, 2, 0],
        indptr=[0, 3, 3, 5],
        shape=(3, 4),
    )


def test_column_selection_renumbers_and_keeps_stored_order() -> None:
    _assert_stored(
        _unsorted().take_columns(np.array([0, 2, 3])),
        data=[3, 1, 6, 4],
        indices=[1, 0, 2, 0],
        indptr=[0, 2, 2, 4],
        shape=(3, 3),
    )


def test_column_selection_keeps_rows_it_empties() -> None:
    _assert_stored(
        _unsorted().take_columns(np.array([2])),
        data=[3],
        indices=[0],
        indptr=[0, 1, 1, 1],
        shape=(3, 1),
    )


def test_joined_blocks_offset_and_sort_every_row() -> None:
    dense = CsrMatrix.from_dense(np.array([[0, 7], [8, 0], [0, 0]], dtype=np.float64))
    _assert_stored(
        hstack([_unsorted(), dense]),
        data=[1, 3, 7, 8, 4, 5, 6],
        indices=[0, 2, 5, 4, 0, 1, 3],
        indptr=[0, 3, 4, 7],
        shape=(3, 6),
    )


def test_one_joined_block_sorts_rows_and_keeps_trailing_empty_rows() -> None:
    block = CsrMatrix(
        data=np.array([1, 2], dtype=np.float64),
        indices=np.array([2, 1], dtype=np.int32),
        indptr=np.array([0, 2, 2], dtype=np.int32),
        shape=(2, 3),
    )
    _assert_stored(
        hstack([block]),
        data=[2, 1],
        indices=[1, 2],
        indptr=[0, 2, 2],
        shape=(2, 3),
    )


def test_joined_blocks_require_equal_row_counts() -> None:
    with pytest.raises(ValueError, match=r"^Expected blocks with equal row counts\.$"):
        hstack([_unsorted(), CsrMatrix.from_dense(np.zeros((2, 5)))])


def test_gram_multiplies_overlapping_columns_and_zeroes_empty_rows() -> None:
    assert np.array_equal(
        _unsorted().gram(),
        a2=[[1 + 3 * 3, 0, 1 * 4], [0, 0, 0], [1 * 4, 0, 4 * 4 + 5 * 5 + 6 * 6]],
    )


def test_gram_adds_in_the_first_row_stored_order() -> None:
    matrix = CsrMatrix(
        data=np.array([2**-53, 2**-53, 1, 1, 1, 1]),
        indices=np.array([1, 2, 0, 0, 1, 2], dtype=np.int32),
        indptr=np.array([0, 3, 6], dtype=np.int32),
        shape=(2, 3),
    )
    # Adding the two tiny terms first keeps them; column order rounds both away.
    assert np.array_equal(matrix.gram(), a2=[[1, 1 + 2**-52], [1, 3]])


def test_gram_rounds_each_product_before_adding_it() -> None:
    gram = CsrMatrix.from_dense(
        np.array([[1, 1 + 2**-27, 0], [-1, 1 - 2**-27, 2]]),
    ).gram()
    # A fused second step would leave -2**-54, the rounding error of the product.
    assert gram[0, 1] == gram[1, 0] == 0


def _unsorted() -> CsrMatrix:
    return CsrMatrix(
        data=np.array([3, 1, 6, 5, 4], dtype=np.float64),
        indices=np.array([2, 0, 3, 1, 0], dtype=np.int32),
        indptr=np.array([0, 2, 2, 5], dtype=np.int32),
        shape=(3, 4),
    )


def _assert_stored(
    actual: CsrMatrix,
    data: ArrayLike,
    indices: ArrayLike,
    indptr: ArrayLike,
    shape: tuple[int, int],
) -> None:
    assert actual.data.dtype == np.float64
    assert actual.indices.dtype == actual.indptr.dtype == np.int32
    assert np.array_equal(actual.data, a2=data)
    assert np.array_equal(actual.indices, a2=indices)
    assert np.array_equal(actual.indptr, a2=indptr)
    assert actual.shape == shape


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
