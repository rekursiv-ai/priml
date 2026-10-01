"""The torch passes must reproduce PSLP's serial loops exactly.

The references below transcribe PSLP 0.0.8's insertion sort, radix sort and
transpose (radix_sort.c, Matrix.c) line for line on small inputs.
"""

from itertools import accumulate
from typing import Final, cast

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.baselines.convextok.presolver.bulk import (
    sort_rows,
    transpose_slots,
)
from priml.lib.custom_json import ListCodec


_CPU: Final = torch.device("cpu")


@pytest.mark.parametrize("n_rows", [40, 300], ids=["insertion", "radix"])
def test_sort_rows_matches_pslp(n_rows: int) -> None:
    rng = np.random.default_rng(n_rows)
    # Few distinct keys, some negative as int32, so ties and signedness matter.
    sp_keys = np.array([-5, -1, 0, 3, 2**30], np.int32)
    ch_keys = np.array([-7, 0, 9], np.int32)
    sparsity = sp_keys[rng.integers(0, sp_keys.size, size=n_rows + 10)]
    coeff = ch_keys[rng.integers(0, ch_keys.size, size=n_rows + 10)]
    active = np.sort(rng.permutation(n_rows + 10)[:n_rows]).astype(np.int32)
    reference = _pslp_insertion_sort if n_rows < 256 else _pslp_radix_sort
    expected = reference(_ints(active), _ints(sparsity), _ints(coeff))
    assert _ints(sort_rows(active, sparsity, coeff, _CPU)) == expected


def test_transpose_slots_match_pslp_layout() -> None:
    rng = np.random.default_rng(0)
    n_rows, n_cols = 7, 5
    rows = [
        _ints(np.sort(rng.permutation(n_cols)[: int(rng.integers(0, 4))]))
        for _ in range(n_rows)
    ]
    starts = [0, *accumulate(len(r) for r in rows)]
    ends = [*starts[1:], starts[-1]]
    cols = [col for row in rows for col in row]
    dest, at_start, at_end, n_alloc = transpose_slots(
        np.array(cols, np.int32),
        np.array(starts, np.int32),
        np.array(ends, np.int32),
        n_cols,
        _CPU,
    )
    assert (_ints(dest), _ints(at_start), _ints(at_end), n_alloc) == _pslp_transpose(
        cols,
        starts,
        ends,
        n_cols,
    )


def _ints(values: NDArray[np.int32] | NDArray[np.int64]) -> list[int]:
    return ListCodec.coerce(cast(object, values.tolist()), int, default=None)


def _pslp_insertion_sort(rows: list[int], sp: list[int], ch: list[int]) -> list[int]:
    """insertion_sort_rows: signed keys; an equal key moves before its equals."""
    for i in range(1, len(rows)):
        key = rows[i]
        j = i
        while j > 0:
            prev = rows[j - 1]
            if sp[prev] < sp[key] or (sp[prev] == sp[key] and ch[prev] < ch[key]):
                break
            rows[j] = prev
            j -= 1
        rows[j] = key
    return rows


def _pslp_radix_sort(rows: list[int], sp: list[int], ch: list[int]) -> list[int]:
    """radix_sort_rows: unsigned bytes, coefficient hash first, first pass reversed."""
    src = rows
    for phase, keys in enumerate((ch, sp)):
        for byte_pass in range(4):
            digits = {
                row: (keys[row] & 0xFFFF_FFFF) >> 8 * byte_pass & 0xFF for row in src
            }
            counts = [0] * 256
            for row in src:
                counts[digits[row]] += 1
            if (phase, byte_pass) != (0, 0) and max(counts) == len(src):
                continue
            offsets = [0, *accumulate(counts)][:-1]
            dst = [0] * len(src)
            for row in reversed(src) if (phase, byte_pass) == (0, 0) else src:
                dst[offsets[digits[row]]] = row
                offsets[digits[row]] += 1
            src = dst
    return src


def _pslp_transpose(
    cols: list[int],
    starts: list[int],
    ends: list[int],
    n_cols: int,
) -> tuple[list[int], list[int], list[int], int]:
    """transpose(): each column's entries in row order, 2 * count + 4 slots each."""
    count = [0] * n_cols
    for k in range(starts[-1]):
        count[cols[k]] += 1
    at_start = [0] * (n_cols + 1)
    at_end = [0] * (n_cols + 1)
    for col in range(n_cols):
        at_end[col] = at_start[col] + count[col]
        at_start[col + 1] = at_start[col] + 2 * count[col] + 4
    n_alloc = 2 * sum(count) + 4 * n_cols
    at_start[n_cols] = at_end[n_cols] = n_alloc
    fill = at_start[:n_cols]
    dest: list[int] = []
    for row in range(len(starts) - 1):
        for k in range(starts[row], ends[row]):
            dest.append(fill[cols[k]])
            fill[cols[k]] += 1
    return dest, at_start, at_end, n_alloc


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
