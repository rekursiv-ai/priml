"""The torch passes must reproduce PSLP's serial loops exactly.

The references below transcribe PSLP 0.0.8's insertion sort, radix sort and
transpose (radix_sort.c, Matrix.c) line for line on small inputs.
"""

from itertools import accumulate
from typing import TYPE_CHECKING, Final, cast

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Callable


from priml.baselines.convextok.presolver.bulk import (
    sort_rows,
    transpose_slots,
)


_CPU: Final = torch.device("cpu")


@pytest.mark.parametrize(
    "n_rows",
    [40, 256, 300],
    ids=["insertion", "boundary", "radix"],
)
def test_sort_rows_matches_pslp(n_rows: int) -> None:
    rng = np.random.default_rng(n_rows)
    # Few distinct keys, some negative as int32, so ties and signedness matter.
    sp_keys = np.array([-(2**31), -5, -3, -2, -1, 0, 3, 2**31 - 1], np.int32)
    ch_keys = np.array([-(2**31), -7, 0, 9, 2**31 - 1], np.int32)
    sparsity = sp_keys[rng.integers(0, sp_keys.size, size=n_rows + 10)]
    coeff = ch_keys[rng.integers(0, ch_keys.size, size=n_rows + 10)]
    active = np.sort(rng.permutation(n_rows + 10)[:n_rows]).astype(np.int32)
    reference = _pslp_insertion_sort if n_rows < 256 else _pslp_radix_sort
    expected = reference(_ints(active), _ints(sparsity), _ints(coeff))
    assert _ints(sort_rows(active, sparsity, coeff, _CPU)) == expected


def test_sort_rows_passes_device_and_dtype_to_tensor_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_to: Callable[..., torch.Tensor] = torch.Tensor.to
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def spy_to(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append((args, kwargs))
        assert not kwargs
        assert len(args) == 1
        target = args[0]
        if isinstance(target, torch.device):
            return original_to(tensor, target)
        assert isinstance(target, torch.dtype)
        return original_to(tensor, target)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    sort_rows(
        np.array([0, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        _CPU,
    )

    assert calls == [
        ((_CPU,), {}),
        ((_CPU,), {}),
        ((torch.int64,), {}),
        ((_CPU,), {}),
        ((torch.int64,), {}),
        ((torch.int32,), {}),
    ]


def test_transpose_slots_pins_slots_for_empty_and_repeated_columns() -> None:
    dest, at_start, at_end, n_alloc = transpose_slots(
        np.array([2, 0, 2, 3, 2], dtype=np.int32),
        np.array([0, 2, 2, 5], dtype=np.int32),
        np.array([2, 2, 5, 5], dtype=np.int32),
        5,
        _CPU,
    )
    assert dest.tolist() == [10, 0, 11, 20, 12]
    assert at_start.tolist() == [0, 6, 10, 20, 26, 30]
    assert at_end.tolist() == [1, 6, 13, 21, 26, 30]
    assert n_alloc == 30
    assert dest.dtype == at_start.dtype == at_end.dtype == np.int32


def test_transpose_slots_pins_tensor_devices_and_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_to: Callable[..., torch.Tensor] = torch.Tensor.to
    original_from_numpy: Callable[..., torch.Tensor] = torch.from_numpy
    original_arange: Callable[..., torch.Tensor] = torch.arange
    original_zeros: Callable[..., torch.Tensor] = torch.zeros
    original_empty: Callable[..., torch.Tensor] = torch.empty
    original_sort: Callable[..., torch.return_types.sort] = torch.sort
    original_bincount: Callable[..., torch.Tensor] = torch.bincount
    sort_stability: list[bool | None] = []
    bincount_minlengths: list[int] = []
    to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    source_dtypes: list[str] = []
    arange_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    zeros_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    empty_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def spy_to(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        to_calls.append((args, kwargs))
        assert not kwargs
        assert len(args) == 1
        target = args[0]
        if isinstance(target, torch.device):
            return original_to(tensor, target)
        assert isinstance(target, torch.dtype)
        return original_to(tensor, target)

    def spy_from_numpy(array: NDArray[np.int32] | NDArray[np.int64]) -> torch.Tensor:
        source_dtypes.append(str(array.dtype))
        return original_from_numpy(array)

    def spy_arange(end: int, *, device: torch.device | None = None) -> torch.Tensor:
        arange_calls.append(((end,), {"device": device}))
        return original_arange(end, device=device)

    def spy_zeros(
        size: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        zeros_calls.append(((size,), {"dtype": dtype, "device": device}))
        return original_zeros(size, dtype=dtype, device=device)

    def spy_empty(
        size: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        empty_calls.append(((size,), {"dtype": dtype, "device": device}))
        return original_empty(size, dtype=dtype, device=device)

    def spy_sort(
        input: torch.Tensor,
        *,
        stable: bool | None = None,
    ) -> torch.return_types.sort:
        sort_stability.append(stable)
        return original_sort(input, stable=stable)

    def spy_bincount(
        input: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        minlength: int = 0,
    ) -> torch.Tensor:
        bincount_minlengths.append(minlength)
        return original_bincount(input, weights=weights, minlength=minlength)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    monkeypatch.setattr(torch, "from_numpy", spy_from_numpy)
    monkeypatch.setattr(torch, "arange", spy_arange)
    monkeypatch.setattr(torch, "zeros", spy_zeros)
    monkeypatch.setattr(torch, "empty", spy_empty)
    monkeypatch.setattr(torch, "sort", spy_sort)
    monkeypatch.setattr(torch, "bincount", spy_bincount)
    transpose_slots(
        np.array([2, 0, 2, 3, 2], dtype=np.int32),
        np.array([0, 2, 2, 5], dtype=np.int32),
        np.array([2, 2, 5, 5], dtype=np.int32),
        5,
        _CPU,
    )

    assert source_dtypes == ["int64", "int64", "int32"]
    assert to_calls == [
        ((_CPU,), {}),
        ((_CPU,), {}),
        ((_CPU,), {}),
        ((torch.int64,), {}),
        ((torch.float64,), {}),
        ((torch.int64,), {}),
        ((torch.int32,), {}),
        ((torch.int32,), {}),
        ((torch.int32,), {}),
    ]
    assert arange_calls == [((5,), {"device": _CPU}), ((5,), {"device": _CPU})]
    assert zeros_calls == [((6,), {"dtype": torch.int64, "device": _CPU})]
    assert empty_calls == [((6,), {"dtype": torch.int64, "device": _CPU})]
    assert sort_stability == [True]
    assert bincount_minlengths == [5]


def test_transpose_slots_matches_pslp_layout() -> None:
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
    return from_plain(cast(object, values.tolist()), list[int])


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
