from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import itertools
import math

from torch import Tensor

import torch


_AVG_POOL = {
    2: torch.nn.functional.avg_pool2d,
    3: torch.nn.functional.avg_pool3d,
}


def adaptive_avg_pool2d(
    input: Tensor,
    output_size: tuple[int, int],
    variance_preserving: bool = False,
) -> Tensor:
    """Adaptive average pooling for 2D inputs (3D or 4D tensors).

    Args:
      input: 3D or 4D input tensor.
      output_size: Target (H, W) spatial dimensions.
      variance_preserving: If True, scale output to preserve
        variance instead of taking the mean.

    Returns:
      result: Pooled tensor with spatial dims matching output_size.

    """
    _check(
        input.ndim in (3, 4),
        lambda: f"Expected 3D or 4D tensor, got {input.ndim}D",
    )
    return _adaptive_avg_pool(input, output_size, variance_preserving)


def adaptive_avg_pool3d(
    input: Tensor,
    output_size: tuple[int, int, int],
    variance_preserving: bool = False,
) -> Tensor:
    """Adaptive average pooling for 3D inputs (4D or 5D tensors).

    Args:
      input: 4D or 5D input tensor.
      output_size: Target (F, H, W) spatial dimensions.
      variance_preserving: If True, scale output to preserve
        variance instead of taking the mean.

    Returns:
      result: Pooled tensor with spatial dims matching output_size.

    """
    _check(
        input.ndim in (4, 5),
        lambda: f"Expected 4D or 5D tensor, got {input.ndim}D",
    )
    return _adaptive_avg_pool(input, output_size, variance_preserving)


# --- Private implementation ---


def _check(
    cond: bool,
    message: Callable[[], str] | None = None,
) -> None:
    """Compile-friendly assert (maps to C++ TORCH_CHECK)."""
    torch._check(cond, message)  # noqa: SLF001  # pyright: ignore[reportUnknownMemberType]


class _DimInfo(NamedTuple):
    idx: Tensor  # [out_size, max_kernel_size] — gathered source indices
    length: int | Tensor  # scalar or [out_size] — window lengths
    max_kernel_size_range: Tensor  # [max_kernel_size] — for masking
    needs_irregular_kernel: bool  # True if window lengths vary


def _dim_info(
    in_size: int,
    out_size: int,
    device: torch.device,
) -> _DimInfo:
    out_range = torch.arange(
        out_size,
        device=device,
        dtype=torch.int64,
    )
    start = torch.div(
        out_range * in_size,
        out_size,
        rounding_mode="trunc",
    )
    max_kernel_size = in_size // out_size + 1
    mod = in_size % out_size
    needs_irregular_kernel = mod != 0 and out_size % mod != 0
    if needs_irregular_kernel:
        max_kernel_size += 1
    elif mod == 0:
        max_kernel_size -= 1

    max_kernel_size_range = torch.arange(
        max_kernel_size,
        device=device,
        dtype=torch.int64,
    )
    idx = start.unsqueeze(-1) + max_kernel_size_range

    if needs_irregular_kernel:
        idx = torch.minimum(
            idx,
            torch.scalar_tensor(
                # in_size == 0 is rejected upstream; clamp to 0 defensively so
                # the index ceiling can never go negative.
                max(0, in_size - 1),
                dtype=idx.dtype,
                device=idx.device,
            ),
        )
        end = torch.div(
            (out_range + 1) * in_size + out_size - 1,
            out_size,
            rounding_mode="trunc",
        )
        length: int | Tensor = end - start
    else:
        length = max_kernel_size

    return _DimInfo(idx, length, max_kernel_size_range, needs_irregular_kernel)


def _trailing(x: Tensor, n: int) -> Tensor:
    if n == 0:
        return x
    return x[(..., *(None,) * n)]


def _adaptive_avg_pool(
    x: Tensor,
    output_size: tuple[int, ...],
    variance_preserving: bool,
) -> Tensor:
    n = len(output_size)
    spatial = x.shape[-n:]

    for s in spatial:
        _check(
            s != 0,
            lambda: f"Expected non-zero spatial dims, got shape {tuple(x.shape)}",
        )

    # Fast path: all dims evenly divisible.
    if all(s % o == 0 for s, o in zip(spatial, output_size, strict=True)):
        stride = tuple(s // o for s, o in zip(spatial, output_size, strict=True))
        kernel = tuple(
            s - (o - 1) * st
            for s, o, st in zip(spatial, output_size, stride, strict=True)
        )
        y = _AVG_POOL[n](x, kernel, stride)
        if variance_preserving:
            y = y * math.prod(stride) ** 0.5
        return y

    # Per-dimension index tables.
    dims = [_dim_info(spatial[i], output_size[i], x.device) for i in range(n)]

    # Gather: pad each idx with trailing unit dims so they broadcast
    # to shape [out_0, max_kernel_size_0, ..., out_{n-1}, max_kernel_size_{n-1}].
    gather = tuple(_trailing(d.idx, 2 * (n - 1 - i)) for i, d in enumerate(dims))
    vals = x[(..., *gather)]

    # Non-adaptive shortcut: uniform windows → mean over max_kernel_size dims.
    if not any(d.needs_irregular_kernel for d in dims):
        max_kernel_size_dims = tuple(-(2 * k + 1) for k in reversed(range(n)))
        y = vals.mean(dim=max_kernel_size_dims)
        if variance_preserving:
            y = y * math.prod(vals.shape[d] for d in max_kernel_size_dims) ** 0.5
        return y

    # Mask out-of-window positions; accumulate per-position window sizes.
    window: int | Tensor = 1
    for i, d in enumerate(dims):
        if isinstance(d.length, int):
            window = window * d.length
            continue
        vals_pad = 2 * (n - 1 - i)
        mask = _trailing(
            d.max_kernel_size_range >= d.length.unsqueeze(-1),
            vals_pad,
        )
        vals = vals.masked_fill(mask, 0.0)
        window = window * _trailing(d.length, n - 1 - i)

    # Sum over kernel elements explicitly (kernel sizes are small).
    acc: Tensor | None = None
    for combo in itertools.product(
        *(range(d.idx.shape[-1]) for d in dims),
    ):
        slc: list[int | slice] = []
        for c in combo:
            slc.extend([slice(None), c])
        term = vals[(..., *slc)]
        acc = term if acc is None else acc + term

    assert acc is not None
    return acc / (window**0.5 if variance_preserving else window)
