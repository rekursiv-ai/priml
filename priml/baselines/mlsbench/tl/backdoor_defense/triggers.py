"""Backdoor triggers of the MLS-Bench ``security-backdoor-defense`` driver.

A trigger takes normalized images, undoes the normalization, stamps its pattern
into ``[0, 1]`` pixel space, clips, and normalizes again, in the driver's
operation order: images that must match the driver's bit for bit follow its
arithmetic, so an algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

import torch


class TriggerFn(Protocol):
    """Stamps a backdoor pattern into normalized images."""

    def __call__(self, media: Tensor, /, *, mean: Tensor, std: Tensor) -> Tensor:
        """Return ``(..., C, H, W)`` triggered images for normalized ``media``."""
        raise NotImplementedError


def badnets(media: Tensor, /, *, mean: Tensor, std: Tensor, size: int = 3) -> Tensor:
    """Whiten a square in the bottom-right corner of every image.

    Args:
      media: ``(..., C, H, W)`` normalized float32 images.
      mean: ``(C,)`` channel means the images were normalized with.
      std: ``(C,)`` channel standard deviations.
      size: Side of the square, in pixels.

    Returns:
      media: The triggered images, normalized.

    References:
      https://arxiv.org/abs/1708.06733
        Gu, Dolan-Gavitt and Garg 2017, "BadNets: Identifying Vulnerabilities
        in the Machine Learning Model Supply Chain."

    """
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    pixels = media * std + mean
    pixels[..., -size:, -size:] = 1.0
    return (pixels.clamp(0.0, 1.0) - mean) / std


def blend(media: Tensor, /, *, mean: Tensor, std: Tensor, alpha: float = 0.2) -> Tensor:
    """Blend a one-pixel checkerboard into every image.

    Args:
      media: ``(..., C, H, W)`` normalized float32 images.
      mean: ``(C,)`` channel means the images were normalized with.
      std: ``(C,)`` channel standard deviations.
      alpha: Weight of the checkerboard; the image keeps ``1 - alpha``.

    Returns:
      media: The triggered images, normalized.

    References:
      https://arxiv.org/abs/1712.05526
        Chen et al. 2017, "Targeted Backdoor Attacks on Deep Learning Systems
        Using Data Poisoning."

    """
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    pixels = media * std + mean
    pattern = torch.zeros_like(pixels)
    pattern[..., ::2, ::2] = 1.0
    pattern[..., 1::2, 1::2] = 1.0
    pixels = (1 - alpha) * pixels + alpha * pattern
    return (pixels.clamp(0.0, 1.0) - mean) / std
