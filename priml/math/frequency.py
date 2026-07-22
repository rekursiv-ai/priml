from __future__ import annotations

from collections.abc import Callable, Sequence

import functools
import math

from torch import Tensor

import torch
import torch.fft

from priml.math.custom_types import Tensorable, convert_to_tensor


def dct1d(x: Tensorable, *, normalize: bool = False) -> Tensor:
    """DCT type-II over the last dimension.

    Returns:
      coefficients: DCT-II of the signal.

    References:
      https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
        via https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/model_code/torch_dct.py

    """
    x = convert_to_tensor(x)
    shape = x.shape
    n = shape[-1]
    x = x.contiguous().view(-1, n)

    reordered = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    y = (torch.fft.fft(reordered, dim=1) * torch.polar(torch.ones_like(k), k)).real

    if normalize:
        y[:, 0] /= n**0.5 * 2
        y[:, 1:] /= (n / 2) ** 0.5 * 2

    return (2 * y.view(*shape)).to(x.dtype)


def idct1d(x: Tensorable, *, normalize: bool = False) -> Tensor:
    """Inverse DCT type-II (DCT type-III) over the last dimension.

    Satisfies idct1d(dct1d(x)) == x.

    Returns:
      signal: Reconstructed signal.

    References:
      https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
        via https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/model_code/torch_dct.py

    """
    x = convert_to_tensor(x)
    shape = x.shape
    n = shape[-1]

    c = x.contiguous().view(-1, n) / 2

    if normalize:
        c[:, 0] *= n**0.5 * 2
        c[:, 1:] *= (n / 2) ** 0.5 * 2

    k = torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    phase = torch.polar(torch.ones_like(k), k)

    q = (
        torch.complex(
            c,
            torch.cat([torch.zeros_like(c[:, :1]), -c.flip([1])[:, :-1]], dim=1),
        )
        * phase
    )

    # irfft needs a contiguous complex input; q.contiguous() is bit-identical
    # to re-wrapping q.real/q.imag and reads more directly.
    y = torch.fft.irfft(
        q.contiguous(),
        n=q.shape[1],
        dim=1,
    )
    z = y.new_zeros(y.shape)
    z[:, ::2] += y[:, : n - (n // 2)]
    z[:, 1::2] += y.flip([1])[:, : n // 2]

    return z.view(*shape).to(x.dtype)


def dctnd(
    x: Tensorable,
    *,
    axis: int | Sequence[int] | None = None,
    normalize: bool = False,
) -> Tensor:
    """DCT type-II on arbitrary axes.

    Args:
      x: Input signal.
      axis: Axes along which to compute the DCT.
      normalize: Use orthonormal basis (matches scipy.fft.dct(norm="ortho")).

    Returns:
      coefficients: DCT-II of the signal along specified axes.

    """
    return _dctnd(
        x,
        axis=axis,
        dct_fn=functools.partial(dct1d, normalize=normalize),
    )


def idctnd(
    x: Tensorable,
    *,
    axis: int | Sequence[int] | None = None,
    normalize: bool = False,
) -> Tensor:
    """Inverse DCT type-II on arbitrary axes.

    Args:
      x: Input coefficients.
      axis: Axes along which to compute the inverse DCT.
      normalize: Use orthonormal basis (matches scipy.fft.idct(norm="ortho")).

    Returns:
      signal: Reconstructed signal along specified axes.

    """
    return _dctnd(
        x,
        axis=axis,
        dct_fn=functools.partial(idct1d, normalize=normalize),
    )


# --- Private implementation ---


def _dctnd(
    x: Tensorable,
    *,
    axis: int | Sequence[int] | None = None,
    dct_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    x = convert_to_tensor(x)
    axes = _normalize_axes(x.ndim, axis)
    y = x
    for a in axes:
        y = torch.moveaxis(y, a, -1)
        s = y.shape
        y = dct_fn(y.reshape(-1, s[-1]))
        y = y.reshape(s)
        y = torch.moveaxis(y, -1, a)
    return y


def _normalize_axes(
    ndim: int,
    axis: int | Sequence[int] | None,
) -> list[int]:
    if axis is None:
        return list(range(ndim))
    axes = sorted(a % ndim for a in ([axis] if isinstance(axis, int) else axis))
    if len(axes) != len(set(axes)):
        raise ValueError(
            f"Duplicate axes after normalization: axis={axis!r} maps to {axes} "
            f"for a rank-{ndim} tensor.",
        )
    return axes
