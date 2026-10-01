"""The presolver's whole-matrix passes in torch, bit-identical to PSLP's serial loops.

Both are orderings of integers, so any device gives PSLP's exact result: the
transpose is a stable sort of entries by column, and the parallel-row grouping
sorts hash keys exactly as PSLP's radix and insertion sorts leave them.
"""

from numpy.typing import NDArray

import numpy as np
import torch

from priml.baselines.convextok.presolver.core import (
    EXTRA_MEMORY_RATIO,
    EXTRA_ROW_SPACE,
)
from priml.baselines.convextok.presolver.custom_typings import IBuf


def transpose_slots(
    cols: IBuf,
    starts: IBuf,
    ends: IBuf,
    n_cols: int,
    device: torch.device,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], int]:
    """Where PSLP's transpose() puts each live entry of a row-major matrix.

    PSLP fills each column in row order into slots with spare capacity: twice
    its count plus four. A stable sort of the entries by column is that order.

    Args:
      cols: Entry columns.
      starts: First entry of each row, then a sentinel.
      ends: One past each row's last live entry, then a sentinel.
      n_cols: Number of columns.
      device: Device the sort runs on.

    Returns:
      dest: Slot of each live entry, entries in row order.
      at_start: First slot of each column, then the allocation size.
      at_end: One past each column's last entry, then the allocation size.
      n_alloc: Allocated slots, 2 per entry plus 4 per column.

    """
    n_rows = starts.size - 1
    row_starts = torch.from_numpy(np.asarray(starts[:n_rows], dtype=np.int64)).to(
        device,
    )
    row_ends = torch.from_numpy(np.asarray(ends[:n_rows], dtype=np.int64)).to(device)
    lengths = row_ends - row_starts
    live = _live_positions(row_starts, lengths)
    col = torch.from_numpy(np.asarray(cols)).to(device)[live].to(torch.int64)
    order = torch.sort(col, stable=True).indices
    counts = torch.bincount(col, minlength=n_cols)
    alloc = (counts.to(torch.float64) * EXTRA_MEMORY_RATIO).to(
        torch.int64,
    ) + EXTRA_ROW_SPACE
    at_start = torch.zeros(n_cols + 1, dtype=torch.int64, device=device)
    at_start[1:] = torch.cumsum(alloc, 0)
    first_sorted = torch.cumsum(counts, 0) - counts
    sorted_col = col[order]
    dest = torch.empty_like(order)
    dest[order] = (
        at_start[sorted_col]
        + torch.arange(order.numel(), device=device)
        - first_sorted[sorted_col]
    )
    n_alloc = int(counts.sum().item() * EXTRA_MEMORY_RATIO) + n_cols * EXTRA_ROW_SPACE
    at_end = torch.empty(n_cols + 1, dtype=torch.int64, device=device)
    at_end[:n_cols] = at_start[:n_cols] + counts
    at_start[n_cols] = n_alloc
    at_end[n_cols] = n_alloc
    return (
        dest.to(torch.int32).cpu().numpy(),
        at_start.to(torch.int32).cpu().numpy(),
        at_end.to(torch.int32).cpu().numpy(),
        n_alloc,
    )


def sort_rows(
    active: IBuf,
    sparsity: IBuf,
    coeff: IBuf,
    device: torch.device,
) -> NDArray[np.int32]:
    """Active rows in the order PSLP's parallel-row sort leaves them.

    Rows sort by (support hash, coefficient hash), equal keys by descending
    position. PSLP compares the hashes as signed ints below 256 rows (insertion
    sort) and as unsigned from 256 up (radix sort, then a 4-way merge whose ties
    go to the later chunk).

    Args:
      active: Active rows, ascending.
      sparsity: Support hash per row, as int32.
      coeff: Coefficient hash per row, as int32.
      device: Device the sort runs on.

    Returns:
      rows: ``active`` reordered.

    """
    rows = torch.from_numpy(np.asarray(active)).to(device).flip(0)
    sp = torch.from_numpy(np.asarray(sparsity)).to(device)[rows].to(torch.int64)
    ch = torch.from_numpy(np.asarray(coeff)).to(device)[rows].to(torch.int64)
    if active.size < 256:
        key = (sp << 32) | (ch + 2**31)
    else:
        key = ((sp & 0xFFFF_FFFF) - 2**31 << 32) | (ch & 0xFFFF_FFFF)
    order = torch.sort(key, stable=True).indices
    return rows[order].to(torch.int32).cpu().numpy()


def _live_positions(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Flat storage positions of every live entry, row by row."""
    offsets = torch.cumsum(lengths, 0) - lengths
    within = torch.arange(int(lengths.sum().item()), device=starts.device)
    within -= torch.repeat_interleave(offsets, lengths)
    return torch.repeat_interleave(starts, lengths) + within
