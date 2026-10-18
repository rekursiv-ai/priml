"""Training augmentations of the MLS-Bench ``cv-data-augmentation`` baselines.

Each ``*_normalize`` function fills
:attr:`~priml.baselines.mlsbench.dl.vision.data.Cifar100Data.Config.augment`.
The reference builds a torchvision pipeline applied to one image at a time in
a loader worker; these apply it to a batch, image by image, drawing from the
batch's worker stream in the reference's order. Every operation the reference
runs on a PIL image is :mod:`~priml.baselines.mlsbench.dl.augmentation.ops`'s
exact copy of Pillow's.

The ``*_space`` tables and the operation adapters repeat torchvision 0.26's
``autoaugment`` module: the same operations in the same order, with the same
magnitude bins computed the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from priml.baselines.mlsbench.dl.augmentation import ops
from priml.baselines.mlsbench.dl.vision.data import normalize, pad_crop_flip


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

type Op = Callable[[Tensor, float], Tensor]
"""An augmentation operation: ``(C, H, W)`` uint8 image and signed magnitude."""

type Space = list[tuple[Op, Tensor, bool]]
"""Each operation a policy may pick, its magnitude bins, and whether it is signed."""


def cutout_normalize(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
    holes: int,
    length: int,
) -> Tensor:
    """Pad, crop and mirror, then zero square holes, then normalize: Cutout.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      pad: Pixels of zeros added on every side before the crop.
      mean: ``(C,)`` float32 channel means.
      std: ``(C,)`` float32 channel standard deviations.
      holes: Squares zeroed in each image.
      length: Side of each square, before clipping to the image.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    References:
      https://arxiv.org/abs/1708.04552
        DeVries and Taylor 2017, "Improved Regularization of Convolutional
        Neural Networks with Cutout."

    """
    views = [
        cutout(
            pad_crop_flip(image[None], stream, pad=pad)[0],
            stream,
            holes=holes,
            length=length,
        )
        for image in images
    ]
    return normalize(torch.stack(views), mean=mean, std=std)


def cutout(
    image: Tensor,
    stream: torch.Generator,
    *,
    holes: int,
    length: int,
) -> Tensor:
    """Zero ``holes`` squares of side ``2 * (length // 2)`` centred at random pixels.

    Each hole draws its centre's row, then its column. Zeroing the uint8
    pixels before normalizing equals the reference's mask on scaled pixels,
    since both normalize an exact zero.

    Args:
      image: ``(C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      holes: Squares to zero.
      length: Nominal side; the square spans ``length // 2`` pixels each side
        of its centre, so an odd length zeroes one pixel fewer. It is clipped
        where it leaves the image.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    height, width = image.shape[-2:]
    out = image.clone()
    for _ in range(holes):
        y = int(torch.randint(0, height, (1,), generator=stream))
        x = int(torch.randint(0, width, (1,), generator=stream))
        top, bottom = max(0, y - length // 2), min(height, y + length // 2)
        left, right = max(0, x - length // 2), min(width, x + length // 2)
        out[:, top:bottom, left:right] = 0
    return out


def randaugment_normalize(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
    num_ops: int,
    magnitude: int,
    num_magnitude_bins: int,
) -> Tensor:
    """Apply RandAugment, then pad, crop and mirror, then normalize.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      pad: Pixels of zeros added on every side before the crop.
      mean: ``(C,)`` float32 channel means.
      std: ``(C,)`` float32 channel standard deviations.
      num_ops: Operations applied in sequence to each image.
      magnitude: Bin of every operation's magnitude.
      num_magnitude_bins: Bins each magnitude range is split into.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    References:
      https://arxiv.org/abs/1909.13719
        Cubuk et al. 2020, "RandAugment: Practical automated data augmentation
        with a reduced search space."

    """
    height, width = images.shape[-2:]
    space = randaugment_space(num_magnitude_bins, height=height, width=width)
    views = [
        pad_crop_flip(
            randaugment(
                image,
                stream,
                space=space,
                num_ops=num_ops,
                magnitude=magnitude,
            )[None],
            stream,
            pad=pad,
        )[0]
        for image in images
    ]
    return normalize(torch.stack(views), mean=mean, std=std)


def randaugment(
    image: Tensor,
    stream: torch.Generator,
    *,
    space: Space,
    num_ops: int,
    magnitude: int,
) -> Tensor:
    """Apply ``num_ops`` operations drawn uniformly from ``space`` at one magnitude.

    Each operation draws its index, then, if it is signed, a coin whose one
    negates the magnitude.

    Args:
      image: ``(C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      space: The operations, as :func:`randaugment_space` lists them.
      num_ops: Operations applied in sequence.
      magnitude: Bin of every operation's magnitude.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    for _ in range(num_ops):
        op, magnitudes, signed = space[
            int(torch.randint(len(space), (1,), generator=stream))
        ]
        value = float(magnitudes[magnitude]) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,), generator=stream):
            value *= -1.0
        image = op(image, value)
    return image


def trivialaugment_normalize(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
    num_magnitude_bins: int,
) -> Tensor:
    """Apply TrivialAugment Wide, then pad, crop and mirror, then normalize.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      pad: Pixels of zeros added on every side before the crop.
      mean: ``(C,)`` float32 channel means.
      std: ``(C,)`` float32 channel standard deviations.
      num_magnitude_bins: Bins each magnitude range is split into.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    References:
      https://arxiv.org/abs/2103.10158
        Mueller and Hutter 2021, "TrivialAugment: Tuning-free Yet
        State-of-the-Art Data Augmentation."

    """
    space = trivialaugment_space(num_magnitude_bins)
    views = [
        pad_crop_flip(
            trivialaugment(image, stream, space=space)[None],
            stream,
            pad=pad,
        )[0]
        for image in images
    ]
    return normalize(torch.stack(views), mean=mean, std=std)


def trivialaugment(image: Tensor, stream: torch.Generator, *, space: Space) -> Tensor:
    """Apply one operation drawn from ``space`` at a magnitude drawn from its bins.

    Draws the operation's index; then, if it has bins, the bin; then, if it
    is signed, a coin whose one negates the magnitude.

    Args:
      image: ``(C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      space: The operations, as :func:`trivialaugment_space` lists them.

    Returns:
      image: ``(C, H, W)`` uint8 pixels.

    """
    op, magnitudes, signed = space[
        int(torch.randint(len(space), (1,), generator=stream))
    ]
    value = (
        float(magnitudes[torch.randint(len(magnitudes), (1,), generator=stream)])
        if magnitudes.ndim > 0
        else 0.0
    )
    if signed and torch.randint(2, (1,), generator=stream):
        value *= -1.0
    return op(image, value)


def randaugment_space(num_bins: int, *, height: int, width: int) -> Space:
    """Return RandAugment's operations for an image of ``height`` by ``width``.

    Args:
      num_bins: How many magnitudes each ranged operation spans.
      height: Image rows, which scale ``translate_y``'s range.
      width: Image columns, which scale ``translate_x``'s range.

    Returns:
      space: Per operation its function, magnitudes and whether a coin negates it.

    """
    return [
        (identity, torch.tensor(0.0), False),
        (shear_x, torch.linspace(0.0, 0.3, num_bins), True),
        (shear_y, torch.linspace(0.0, 0.3, num_bins), True),
        (translate_x, torch.linspace(0.0, 150.0 / 331.0 * width, num_bins), True),
        (translate_y, torch.linspace(0.0, 150.0 / 331.0 * height, num_bins), True),
        (rotate, torch.linspace(0.0, 30.0, num_bins), True),
        (brightness, torch.linspace(0.0, 0.9, num_bins), True),
        (color, torch.linspace(0.0, 0.9, num_bins), True),
        (contrast, torch.linspace(0.0, 0.9, num_bins), True),
        (sharpness, torch.linspace(0.0, 0.9, num_bins), True),
        (posterize, _posterize_bins(num_bins, span=4), False),
        (solarize, torch.linspace(255.0, 0.0, num_bins), False),
        (autocontrast, torch.tensor(0.0), False),
        (equalize, torch.tensor(0.0), False),
    ]


def trivialaugment_space(num_bins: int) -> Space:
    """Return TrivialAugment Wide's operations, whose ranges ignore the image size.

    Args:
      num_bins: How many magnitudes each ranged operation spans.

    Returns:
      space: Per operation its function, magnitudes and whether a coin negates it.

    """
    return [
        (identity, torch.tensor(0.0), False),
        (shear_x, torch.linspace(0.0, 0.99, num_bins), True),
        (shear_y, torch.linspace(0.0, 0.99, num_bins), True),
        (translate_x, torch.linspace(0.0, 32.0, num_bins), True),
        (translate_y, torch.linspace(0.0, 32.0, num_bins), True),
        (rotate, torch.linspace(0.0, 135.0, num_bins), True),
        (brightness, torch.linspace(0.0, 0.99, num_bins), True),
        (color, torch.linspace(0.0, 0.99, num_bins), True),
        (contrast, torch.linspace(0.0, 0.99, num_bins), True),
        (sharpness, torch.linspace(0.0, 0.99, num_bins), True),
        (posterize, _posterize_bins(num_bins, span=6), False),
        (solarize, torch.linspace(255.0, 0.0, num_bins), False),
        (autocontrast, torch.tensor(0.0), False),
        (equalize, torch.tensor(0.0), False),
    ]


def identity(image: Tensor, magnitude: float) -> Tensor:
    """Return ``image`` unchanged."""
    del magnitude
    return image


def shear_x(image: Tensor, magnitude: float) -> Tensor:
    """Shear along x by ``magnitude``, the tangent of the angle, about the corner."""
    angle = math.degrees(math.atan(magnitude))
    return ops.affine(
        image,
        ops.inverse_affine([0, 0], translate=[0, 0], shear=[angle, 0.0]),
    )


def shear_y(image: Tensor, magnitude: float) -> Tensor:
    """Shear along y by ``magnitude``, the tangent of the angle, about the corner."""
    angle = math.degrees(math.atan(magnitude))
    return ops.affine(
        image,
        ops.inverse_affine([0, 0], translate=[0, 0], shear=[0.0, angle]),
    )


def translate_x(image: Tensor, magnitude: float) -> Tensor:
    """Shift right by ``magnitude`` pixels, truncated towards zero.

    Args:
      image: ``(3, H, W)`` uint8 pixels.
      magnitude: Pixels to shift; negative shifts left.

    Returns:
      image: ``(3, H, W)`` uint8 pixels, shifted.

    """
    # torchvision's inverse map at zero shear, whose centre cancels exactly.
    return ops.affine(image, [1.0, 0.0, -float(int(magnitude)), 0.0, 1.0, 0.0])


def translate_y(image: Tensor, magnitude: float) -> Tensor:
    """Shift down by ``magnitude`` pixels, truncated towards zero.

    Args:
      image: ``(3, H, W)`` uint8 pixels.
      magnitude: Pixels to shift; negative shifts up.

    Returns:
      image: ``(3, H, W)`` uint8 pixels, shifted.

    """
    # torchvision's inverse map at zero shear, whose centre cancels exactly.
    return ops.affine(image, [1.0, 0.0, 0.0, 0.0, 1.0, -float(int(magnitude))])


def rotate(image: Tensor, magnitude: float) -> Tensor:
    """Rotate counter-clockwise by ``magnitude`` degrees about the centre."""
    return ops.rotate(image, magnitude)


def brightness(image: Tensor, magnitude: float) -> Tensor:
    """Blend from black by ``1 + magnitude``: ``ImageEnhance.Brightness``."""
    return ops.blend(torch.zeros_like(image), image, 1.0 + magnitude)


def color(image: Tensor, magnitude: float) -> Tensor:
    """Blend from the image's luma by ``1 + magnitude``: ``ImageEnhance.Color``."""
    return ops.blend(ops.grayscale(image).expand_as(image), image, 1.0 + magnitude)


def contrast(image: Tensor, magnitude: float) -> Tensor:
    """Blend from the mean luma by ``1 + magnitude``: ``ImageEnhance.Contrast``.

    The mean is ``ImageStat``'s, a float quotient of integer totals, rounded
    half up to an integer.

    Args:
      image: ``(3, H, W)`` uint8 pixels.
      magnitude: Blend factor less one; zero returns the image.

    Returns:
      image: ``(3, H, W)`` uint8 pixels, contrast adjusted.

    """
    luma = ops.grayscale(image)
    total = int(
        (torch.bincount(luma.reshape(-1), minlength=256) * torch.arange(256)).sum(),
    )
    level = int(total / luma.numel() + 0.5)
    return ops.blend(torch.full_like(image, level), image, 1.0 + magnitude)


def sharpness(image: Tensor, magnitude: float) -> Tensor:
    """Blend from the smoothed image by ``1 + magnitude``: ``ImageEnhance.Sharpness``."""
    return ops.blend(ops.smooth(image), image, 1.0 + magnitude)


def posterize(image: Tensor, magnitude: float) -> Tensor:
    """Keep the top ``int(magnitude)`` bits of every channel."""
    return ops.posterize(image, int(magnitude))


def solarize(image: Tensor, magnitude: float) -> Tensor:
    """Invert every value at or above ``magnitude``."""
    return ops.solarize(image, magnitude)


def autocontrast(image: Tensor, magnitude: float) -> Tensor:
    """Stretch each channel to the full range; ``magnitude`` is unused."""
    del magnitude
    return ops.autocontrast(image)


def equalize(image: Tensor, magnitude: float) -> Tensor:
    """Equalize each channel's histogram; ``magnitude`` is unused."""
    del magnitude
    return ops.equalize(image)


def _posterize_bins(num_bins: int, *, span: int) -> Tensor:
    """Bits kept per bin, from 8 down by ``span``, as torchvision rounds them."""
    return 8 - (torch.arange(num_bins) / ((num_bins - 1) / span)).round().int()
