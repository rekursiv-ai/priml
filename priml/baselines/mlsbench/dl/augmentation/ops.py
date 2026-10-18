"""Pillow's image operations on uint8 tensors, reproducing its pixels exactly.

The MLS-Bench augmentation baselines run torchvision's RandAugment and
TrivialAugmentWide on PIL images, so every operation they reach is Pillow's C
code. Each function here repeats that code's arithmetic -- its float32 or
16.16 fixed-point precision, its order of operations and its truncations -- so
the output is Pillow's bit for bit. An algebraically equal rewrite is not a
refactor: it changes the rounding.

Images are ``(C, H, W)`` uint8 tensors. The arithmetic is Pillow 12.3's
``libImaging`` (``Geometry.c``, ``Blend.c``, ``Filter.c``, ``Convert.c``) and
``ImageOps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def inverse_affine(
    center: Sequence[float],
    *,
    translate: Sequence[float],
    shear: Sequence[float],
) -> list[float]:
    """Return the output-to-input map torchvision hands Pillow for an affine.

    torchvision's ``_get_inverse_affine_matrix`` at angle 0 and scale 1, the
    only ones the augmentation policies use, in its operation order.

    Args:
      center: Fixed point of the shear, in pixels from the top left.
      translate: Shift of the output, in pixels.
      shear: Shear angles parallel to x and y, in degrees.

    Returns:
      matrix: ``(a, b, c, d, e, f)`` for :func:`affine`.

    """
    rot = math.radians(0.0)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])
    cx, cy = center
    tx, ty = translate
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
    matrix = [x / 1.0 for x in (d, -b, 0.0, -c, a, 0.0)]
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    matrix[2] += cx
    matrix[5] += cy
    return matrix


def affine(image: Tensor, matrix: Sequence[float]) -> Tensor:
    """Resample through an inverse affine map, nearest neighbour, zero fill.

    ``Image.transform(size, AFFINE, matrix, NEAREST, fillcolor=0)``: output
    pixel ``(x, y)`` reads the input at ``matrix`` applied to its centre.

    Args:
      image: ``(C, H, W)`` uint8 pixels.
      matrix: ``(a, b, c, d, e, f)``, mapping output ``(x, y)`` to input
        ``(a x + b y + c, d x + e y + f)``.

    Returns:
      image: ``(C, H, W)`` uint8 pixels; zero where the map leaves the input.

    """
    a = list(matrix)
    height, width = image.shape[-2:]
    if a[1] == 0 and a[3] == 0:
        rows, columns, inside = _scaled(a, height=height, width=width)
    else:
        rows, columns, inside = _fixed_point(a, height=height, width=width)
    return torch.where(inside, image[:, rows, columns], 0)


def rotate(image: Tensor, angle: float) -> Tensor:
    """Rotate counter-clockwise about the centre: ``Image.rotate(angle)``.

    Args:
      image: ``(C, H, W)`` uint8 pixels.
      angle: Degrees, counter-clockwise.

    Returns:
      image: ``(C, H, W)`` uint8 pixels, zero outside the rotated input.

    """
    angle %= 360.0
    height, width = image.shape[-2:]
    if angle == 0:
        return image.clone()
    if angle == 180:
        return image.flip(-2, -1)
    if angle in (90, 270) and width == height:
        return image.rot90(1 if angle == 90 else -1, (-2, -1))
    center_x, center_y = width / 2, height / 2
    radians = -math.radians(angle)
    a, b, d, e = (
        round(math.cos(radians), 15),
        round(math.sin(radians), 15),
        round(-math.sin(radians), 15),
        round(math.cos(radians), 15),
    )
    return affine(
        image,
        [
            a,
            b,
            a * -center_x + b * -center_y + 0.0 + center_x,
            d,
            e,
            d * -center_x + e * -center_y + 0.0 + center_y,
        ],
    )


def blend(degenerate: Tensor, image: Tensor, factor: float) -> Tensor:
    """Interpolate from ``degenerate`` towards ``image``: ``Image.blend``.

    Args:
      degenerate: ``(C, H, W)`` uint8 pixels at factor 0.
      image: ``(C, H, W)`` uint8 pixels at factor 1.
      factor: Interpolates in ``[0, 1]``, extrapolates outside it.

    Returns:
      image: ``(C, H, W)`` uint8 pixels, truncated and clipped to ``[0, 255]``.

    """
    alpha = torch.tensor(factor, dtype=torch.float32)
    if float(alpha) == 0.0:
        return degenerate.clone()
    if float(alpha) == 1.0:
        return image.clone()
    # Pillow computes in float32: the difference times alpha, then the sum.
    value = degenerate.float() + alpha * (image.int() - degenerate.int()).float()
    return value.clamp(0, 255).to(torch.uint8)


def grayscale(image: Tensor) -> Tensor:
    """Return the ITU-R 601 luma Pillow's ``convert("L")`` computes.

    Args:
      image: ``(3, H, W)`` uint8 RGB pixels.

    Returns:
      luma: ``(H, W)`` uint8 pixels, rounded in 16.16 fixed point.

    """
    red, green, blue = image.int()
    luma = red * 19_595 + green * 38_470 + blue * 7_471 + 0x8000
    return (luma >> 16).to(torch.uint8)


def smooth(image: Tensor) -> Tensor:
    """Apply ``ImageFilter.SMOOTH``, a 3x3 kernel of ones around a five, over 13.

    The border rows and columns are copied, as Pillow leaves them.

    Args:
      image: ``(C, H, W)`` uint8 pixels.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    kernel = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]) / 13.0
    pixels = image.float()
    total = (
        (0.5 + _row(pixels[:, 2:], kernel[0:3])) + _row(pixels[:, 1:-1], kernel[3:6])
    ) + _row(pixels[:, :-2], kernel[6:9])
    out = image.clone()
    out[:, 1:-1, 1:-1] = total.clamp(0, 255).to(torch.uint8)
    return out


def posterize(image: Tensor, bits: int) -> Tensor:
    """Keep the top ``bits`` bits of every channel: ``ImageOps.posterize``."""
    return image & (~((1 << (8 - bits)) - 1) & 0xFF)


def solarize(image: Tensor, threshold: float) -> Tensor:
    """Invert every value at or above ``threshold``: ``ImageOps.solarize``."""
    return torch.where(image.double() < threshold, image, 255 - image)


def autocontrast(image: Tensor) -> Tensor:
    """Stretch each channel so its darkest value is 0 and its lightest 255.

    ``ImageOps.autocontrast`` with no cutoff: a channel of one value is kept.

    Args:
      image: ``(C, H, W)`` uint8 pixels.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    channels: list[Tensor] = []
    for channel in image:
        present = torch.bincount(channel.reshape(-1), minlength=256).nonzero()
        low, high = int(present[0]), int(present[-1])
        if high <= low:
            channels.append(channel.clone())
            continue
        scale = 255.0 / (high - low)
        offset = -low * scale
        lut = (torch.arange(256, dtype=torch.float64) * scale + offset).trunc()
        channels.append(lut.clamp(0, 255).to(torch.uint8)[channel.long()])
    return torch.stack(channels)


def equalize(image: Tensor) -> Tensor:
    """Equalize each channel's histogram: ``ImageOps.equalize``.

    Args:
      image: ``(C, H, W)`` uint8 pixels.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    channels: list[Tensor] = []
    for channel in image:
        histogram = torch.bincount(channel.reshape(-1), minlength=256)
        present = histogram[histogram > 0]
        step = (int(present.sum()) - int(present[-1])) // 255
        if len(present) <= 1 or not step:
            channels.append(channel.clone())
            continue
        before = histogram.cumsum(0) - histogram
        lut = ((step // 2 + before) // step).clamp(max=255).to(torch.uint8)
        channels.append(lut[channel.long()])
    return torch.stack(channels)


def _row(pixels: Tensor, weights: Tensor) -> Tensor:
    """Weigh each interior pixel and its two horizontal neighbours, left first."""
    return (pixels[..., :-2] * weights[0] + pixels[..., 1:-1] * weights[1]) + pixels[
        ...,
        2:,
    ] * weights[2]


# The positions accumulate one step at a time in double precision, as the C loop does; a
# product per pixel would round differently.
def _scaled(
    a: list[float],
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Source rows, columns and validity of ``ImagingScaleAffine``."""
    columns = [0] * width
    first, stop = width, 0
    position = a[2] + a[0] * 0.5
    for x in range(width):
        source = _coord(position)
        if 0 <= source < width:
            stop = x + 1
            first = min(first, x)
            columns[x] = source
        position += a[0]
    rows: list[int] = []
    position = a[5] + a[4] * 0.5
    for _ in range(height):
        rows.append(_coord(position))
        position += a[4]
    row = torch.tensor(rows)
    x = torch.arange(width)
    inside = ((row >= 0) & (row < height))[:, None] & ((x >= first) & (x < stop))
    return (
        row.clamp(0, height - 1)[:, None].expand(height, width),
        torch.tensor(columns)[None, :].expand(height, width),
        inside,
    )


def _fixed_point(
    a: list[float],
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Source rows, columns and validity of ``affine_fixed``, in 16.16 fixed point."""
    corners = ((0, 0), (width, height), (0, height), (width, 0))
    if not all(
        (
            abs(x * a[0] + y * a[1] + a[2]) < 32768.0
            and abs(x * a[3] + y * a[4] + a[5]) < 32768.0
            for x, y in corners
        ),
    ):
        raise ValueError("Pillow takes its floating-point path beyond 16.16 range.")
    a0, a1, a3, a4 = (_fix(a[i]) for i in (0, 1, 3, 4))
    a2 = _fix(a[2] + a[0] * 0.5 + a[1] * 0.5)
    a5 = _fix(a[5] + a[3] * 0.5 + a[4] * 0.5)
    y = torch.arange(height)[:, None]
    x = torch.arange(width)[None, :]
    columns = (a2 + y * a1 + x * a0) >> 16
    rows = (a5 + y * a4 + x * a3) >> 16
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    return rows.clamp(0, height - 1), columns.clamp(0, width - 1), inside


def _coord(value: float) -> int:
    """Pillow's ``COORD``: truncate, with every negative position out of range."""
    return -1 if value < 0.0 else int(value)


def _fix(value: float) -> int:
    """Pillow's ``FIX``: round to the nearest 1/65536, in units of 1/65536."""
    return math.floor(value * 65_536.0 + 0.5)
