"""Tensor operations and utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import functools

from torch import Tensor, nn

import torch

from priml.compile import lazy_torch_compile
from priml.math.custom_types import Tensorable, convert_to_tensor


@functools.wraps(nn.functional.pad)
def pad(
    input: Tensorable,
    pad: Sequence[int],
    mode: Literal[
        "reflect",
        "replicate",
        "circular",
        "constant",
        "reflect_replicate",
    ] = "constant",
    value: float | None = None,
) -> Tensor:
    """Pad tensor with support for reflect_replicate mode.

    Args:
        input: Input tensor to pad
        pad: Padding sizes (left, right) for each dimension from last to first
        mode: Padding mode. Supports standard modes plus:
            - "reflect_replicate": Automatically uses replicate
              padding when reflect would fail, then applies reflect padding
        value: Fill value for constant padding

    Returns:
        padded: Padded tensor.

    The reflect_replicate mode solves the problem where reflect
    padding fails if padding size exceeds tensor dimension. It splits padding
    into replicate + reflect such that reflect never fails.

    For each axis, the smallest replicate amount c such that the remaining
    reflect padding (r - c) is valid for the enlarged dimension (s + c) is
    computed and applied first.

    """
    input = convert_to_tensor(input)
    # Ensure contiguous layout for nn.functional.pad backward pass.
    input = input.contiguous()
    if was_1d := input.ndim < 2:
        input = input[torch.newaxis]
    if mode != "reflect_replicate":
        y = nn.functional.pad(input, pad=pad, mode=mode, value=value)
        return y[0] if was_1d else y
    shape = input.shape
    rank = len(pad) // 2
    shape = [i for i in reversed(shape[-rank:]) for _ in range(2)]
    # For each axis, split padding into replicate (c) then reflect (r - c).
    # Reflect requires dim > pad, i.e., s + c > r - c, i.e., 2c > r - s.
    # So c = ceil((r - s + 1) / 2) = (r - s + 2) // 2, clamped to [0, r + 1].
    replicate_pad = [
        min(r + 1, max(0, r + 2 - s)) // 2 for s, r in zip(shape, pad, strict=True)
    ]
    reflect_pad = [max(0, r - c) for c, r in zip(replicate_pad, pad, strict=True)]
    if any(replicate_pad):
        input = nn.functional.pad(input, pad=replicate_pad, mode="replicate")
    y = nn.functional.pad(input, pad=reflect_pad, mode="reflect")
    return y[0] if was_1d else y


@lazy_torch_compile(fullgraph=True, dynamic=True)
def compiled_scale_and_shift(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply scale and shift to tensor with torch.compile optimization.

    Computes: (x.to(dtype=scale.dtype) * scale) + shift

    This function is compiled with torch.compile for optimal performance.
    Call warmup_compiled_scale_and_shift() before first use to avoid
    compilation overhead during profiling/benchmarking.

    Args:
        x: Input tensor (typically uint8)
        shift: Shift/bias tensor (typically bf16/fp32)
        scale: Scale tensor (typically bf16/fp32)

    Returns:
        result: Scaled and shifted tensor in scale.dtype

    """
    return (x.to(dtype=scale.dtype) * scale) + shift


def warmup_compiled_scale_and_shift(
    dtype_in: torch.dtype = torch.uint8,
    dtype_out: torch.dtype = torch.bfloat16,
) -> None:
    """Warmup compiled_scale_and_shift to avoid compilation during profiling.

    Call this once before profiling to pre-compile the function with realistic
    input shapes and dtypes.

    Args:
        dtype_in: Input dtype (default: uint8)
        dtype_out: Output dtype for shift/scale (default: bfloat16)

    """
    x = torch.randint(0, 255, (3, 1, 224, 224), dtype=dtype_in)
    bias = torch.randn(3, 1, 1, 1, dtype=dtype_out)
    scale = torch.randn(3, 1, 1, 1, dtype=dtype_out)
    _ = compiled_scale_and_shift(x, bias, scale)


def as_batch_tensor(
    x: Tensorable,
    min_ndim: int = 4,
    max_ndim: int = 5,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Normalize tensor to have ndim in range [min_ndim, max_ndim].

    Ensures output tensor has number of dimensions within the specified range by:
    - Adding singleton dimensions on the left if x.ndim < min_ndim
    - Flattening leading dimensions if x.ndim > max_ndim
    - Preserving trailing dimensions when flattening

    Args:
        x: Input tensor or tensor-like object.
        min_ndim: Minimum number of dimensions (must be >= 1).
        max_ndim: Maximum number of dimensions (must be >= 0).
        dtype: Optional dtype for output tensor.
        device: Optional device for output tensor.

    Returns:
        output: Tensor with min_ndim <= output.ndim <= max_ndim.

    Examples:
        >>> # Add dimensions: (H, W, C) -> (1, H, W, C)
        >>> as_batch_tensor(torch.zeros(224, 224, 3), min_ndim=4, max_ndim=5).shape
        torch.Size([1, 224, 224, 3])

        >>> # Flatten leading dims: (B, F, H, W, C) -> (B*F, H, W, C)
        >>> as_batch_tensor(torch.zeros(2, 3, 224, 224, 3), min_ndim=4, max_ndim=4).shape
        torch.Size([6, 224, 224, 3])

        >>> # Already in range: (B, F, H, W, C) -> (B, F, H, W, C)
        >>> as_batch_tensor(torch.zeros(2, 3, 224, 224, 3), min_ndim=4, max_ndim=5).shape
        torch.Size([2, 3, 224, 224, 3])

    """
    if min_ndim < 1:
        raise ValueError(f"Nonpositive {min_ndim=}.")
    if max_ndim < 0:
        raise ValueError(f"Negative {max_ndim=}.")
    x = torch.as_tensor(x, dtype=dtype, device=device)
    d = max(0, min_ndim - x.ndim)
    c = max(1 - max_ndim + x.ndim, 1 - d)
    x = x.reshape(-1, *[1] * max(0, -c), *x.shape[max(0, c) :])
    return x
