"""GPU-side batch augmentation primitives.

Vectorized augmentations that operate directly on GPU tensors.
All functions take and return (B, C, H, W) tensors.
"""

from __future__ import annotations

from torch import Tensor, nn

import torch


def flip_lr(images: Tensor) -> Tensor:
    """Random horizontal flip per image (50% probability)."""
    mask = torch.rand(len(images), device=images.device) < 0.5
    return torch.where(mask.view(-1, 1, 1, 1), images.flip(-1), images)


def random_crop(images: Tensor, crop_size: int) -> Tensor:
    """Fast random crop via gather, vectorized across the batch.

    Expects pre-padded square input with ``H == W >= crop_size``; the crop
    offset is drawn uniformly from the ``H - crop_size`` slack on each axis.

    Args:
      images: (B, C, H, W) pre-padded input tensor.
      crop_size: Output spatial size; must not exceed H or W.

    Returns:
      cropped: (B, C, crop_size, crop_size) tensor.

    """
    B, C, H, W = images.shape
    assert crop_size <= H, (
        f"crop_size={crop_size} exceeds input height H={H}; pad before cropping."
    )
    assert crop_size <= W, (
        f"crop_size={crop_size} exceeds input width W={W}; pad before cropping."
    )
    pad = (H - crop_size) // 2
    dy = torch.randint(0, 2 * pad + 1, (B, 1, 1, 1), device=images.device)
    dx = torch.randint(0, 2 * pad + 1, (B, 1, 1, 1), device=images.device)
    row_idx = dy + torch.arange(crop_size, device=images.device).view(1, 1, -1, 1)
    col_idx = dx + torch.arange(crop_size, device=images.device).view(1, 1, 1, -1)
    flat_idx = (row_idx * W + col_idx).expand(B, C, crop_size, crop_size)
    return (
        images.reshape(B, C, -1)
        .gather(2, flat_idx.reshape(B, C, -1))
        .reshape(B, C, crop_size, crop_size)
    )


def cutout(images: Tensor, size: int) -> Tensor:
    """Zero out a random square per image.

    The square's top-left corner is drawn uniformly over the full image, so a
    square near an edge is clipped to the image bounds and the zeroed region
    may be smaller than ``size x size``.

    Args:
      images: (B, C, H, W) input tensor.
      size: Side length of the cutout square in pixels.

    Returns:
      out: Tensor with the cutout region set to zero.

    """
    B, _C, H, W = images.shape
    cy = torch.randint(0, H, (B, 1, 1, 1), device=images.device)
    cx = torch.randint(0, W, (B, 1, 1, 1), device=images.device)
    ys = torch.arange(H, device=images.device).view(1, 1, -1, 1)
    xs = torch.arange(W, device=images.device).view(1, 1, 1, -1)
    mask = (ys >= cy) & (ys < cy + size) & (xs >= cx) & (xs < cx + size)
    return images.masked_fill(mask, 0.0)


def pad_crop_flip(
    images: Tensor,
    crop_size: int,
    *,
    pad: int = 2,
    pad_mode: str = "reflect",
    flip: bool | None = None,
    cutout_size: int = 0,
) -> Tensor:
    """Common training augmentation: pad, random-crop, flip, then cutout.

    The input is padded by ``pad`` on all sides, so ``crop_size`` may equal
    the original spatial size while still leaving ``2 * pad`` slack to crop.

    Args:
      images: (B, C, H, W) input tensor.
      crop_size: Output spatial size after cropping.
      pad: Padding pixels added on every side before cropping.
      pad_mode: Padding mode ("reflect", "replicate", "constant").
      flip: None=random 50% per image, True=flip all, False=flip none.
      cutout_size: Side length of cutout square; 0 disables.

    Returns:
      out: Contiguous (B, C, crop_size, crop_size) augmented tensor.

    """
    padded = nn.functional.pad(images, (pad,) * 4, pad_mode)
    out = random_crop(padded, crop_size)
    if flip is None:
        out = flip_lr(out)
    elif flip:
        out = out.flip(-1)
    if cutout_size > 0:
        out = cutout(out, cutout_size)
    return out.contiguous()
