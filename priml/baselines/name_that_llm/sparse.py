"""Store sparse lexical features as compressed rows over NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import itertools

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class CsrMatrix:
    """Compressed sparse rows whose stored order fixes every summation order.

    Attributes:
      data: Stored values, row after row.
      indices: Column of each stored value.
      indptr: Offset of each row's first stored value, then the stored count.
      shape: Number of rows and columns.

    """

    data: np.ndarray[tuple[int], np.dtype[np.float64]]
    indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    indptr: np.ndarray[tuple[int], np.dtype[np.int32]]
    shape: tuple[int, int]

    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> CsrMatrix:
        """Store every nonzero entry of a dense matrix in row-major order.

        Args:
          dense: Rows by columns; zeros are not stored.

        Returns:
          matrix: The nonzero values with their columns.

        """
        rows, columns = np.nonzero(dense)
        return cls(
            data=dense[rows, columns],
            indices=columns.astype(np.int32),
            indptr=_offsets(np.bincount(rows, minlength=dense.shape[0])),
            shape=dense.shape,
        )

    def sorted_indices(self) -> CsrMatrix:
        """Return a copy storing each row's distinct columns in increasing order."""
        order = np.lexsort((self.indices, _value_rows(self)))
        return replace(self, data=self.data[order], indices=self.indices[order])

    def take_rows(self, selected: NDArray[np.intp]) -> CsrMatrix:
        """Gather rows, each keeping its stored order.

        Args:
          selected: Rows to gather, in their new order.

        Returns:
          matrix: One row per selected row, with every column retained.

        """
        lengths = np.diff(self.indptr)[selected]
        positions = _spans(self.indptr[selected], lengths=lengths)
        return CsrMatrix(
            data=self.data[positions],
            indices=self.indices[positions],
            indptr=_offsets(lengths),
            shape=(len(selected), self.shape[1]),
        )

    def take_columns(self, selected: NDArray[np.intp]) -> CsrMatrix:
        """Keep the selected distinct columns, numbered by position in `selected`.

        Args:
          selected: Distinct columns to retain, in their new order.

        Returns:
          matrix: The retained values, each row keeping its stored order.

        """
        kept = np.isin(self.indices, selected)
        position = np.empty(self.shape[1], dtype=np.int32)
        position[selected] = np.arange(len(selected))
        return CsrMatrix(
            data=self.data[kept],
            indices=position[self.indices[kept]],
            indptr=_offsets(
                np.bincount(_value_rows(self)[kept], minlength=self.shape[0]),
            ),
            shape=(self.shape[0], len(selected)),
        )

    def gram(self) -> NDArray[np.float64]:
        """Return every pairwise row inner product as a dense matrix.

        Entry `(i, k)` sums the products in row `i`'s stored order, so it may
        differ from `(k, i)` in the last bit.

        Returns:
          products: Dense float64 inner products, rows by rows.

        """
        # Each column holds a row at most once, so any sort keeps every sum's order.
        by_column = np.argsort(self.indices)
        column_rows = _value_rows(self)[by_column]
        column_data = self.data[by_column]
        starts = _offsets(np.bincount(self.indices))
        products = np.zeros((self.shape[0], self.shape[0]))
        for row, (start, stop) in enumerate(itertools.pairwise(self.indptr.tolist())):
            columns = self.indices[start:stop]
            lengths = starts[columns + 1] - starts[columns]
            positions = _spans(starts[columns], lengths=lengths)
            sums = np.zeros(self.shape[0])
            np.add.at(
                sums,
                column_rows[positions],
                np.repeat(self.data[start:stop], repeats=lengths)
                * column_data[positions],
            )
            products[row] = sums
        return products


def hstack(blocks: Sequence[CsrMatrix]) -> CsrMatrix:
    """Join blocks side by side, storing each row's columns in increasing order.

    The classifier sums every row in stored order, so the order is part of its bits.

    Args:
      blocks: Matrices with equal row counts and distinct columns in each row.

    Returns:
      matrix: All blocks' values with columns offset by the preceding widths.

    Raises:
      ValueError: The blocks have different row counts.

    """
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError("Expected blocks with equal row counts.")
    widths = list(itertools.accumulate([block.shape[1] for block in blocks], initial=0))
    rows = np.concatenate([_value_rows(block) for block in blocks])
    columns = np.concatenate(
        [block.indices + widths[index] for index, block in enumerate(blocks)],
    )
    order = np.lexsort((columns, rows))
    return CsrMatrix(
        data=np.concatenate([block.data for block in blocks])[order],
        indices=columns[order].astype(np.int32),
        indptr=_offsets(np.bincount(rows, minlength=blocks[0].shape[0])),
        shape=(blocks[0].shape[0], widths[-1]),
    )


def _value_rows(matrix: CsrMatrix) -> NDArray[np.intp]:
    """Return the row of every stored value."""
    return np.repeat(np.arange(matrix.shape[0]), repeats=np.diff(matrix.indptr))


def _offsets(lengths: NDArray[np.integer]) -> NDArray[np.int32]:
    """Return the start of each length, then their total."""
    return np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)


def _spans(
    starts: NDArray[np.integer],
    lengths: NDArray[np.integer],
) -> NDArray[np.intp]:
    """Concatenate every range `[start, start + length)` in order."""
    offsets = _offsets(lengths)
    return np.repeat(starts - offsets[:-1], repeats=lengths) + np.arange(
        offsets.item(-1),
    )
