"""Utility functions for data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import math

from PIL import Image as PILImage
from torch import Tensor

import torch
import torchvision.transforms.functional as tvf

from priml.math.pixel import float2rgb


if TYPE_CHECKING:
    from priml.math.custom_types import Tensorable


InterpolationMode = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "area",
    "nearest-exact",
    "lanczos",
    "hybrid",
]

__all__ = [
    "InterpolationMode",
    "as_image_batch_tensor",
    "compute_keyframes_as_progressive_bisection",
    "image_batch_to_pil_list",
    "preprocess_images",
    "safe_aspect_ratio",
    "sample_frame_indices",
]


def safe_aspect_ratio(height: float, width: float) -> float:
    """Compute aspect ratio with division by zero protection.

    Args:
      height: Height value
      width: Width value

    Returns:
      aspect_ratio: width / height, or torch.inf if height is zero

    """
    return width / height if height != 0 else math.inf


def compute_keyframes_as_progressive_bisection(
    total_frames: int,
    num_keyframes: int,
) -> list[int]:
    """Compute keyframe indices using progressive bisection.

    Progressively adds frames in priority order:
    1. First frame (index 0)
    2. Last frame (index total_frames - 1)
    3. Middle frame (bisect between first and last)
    4. Continue bisecting largest gaps until num_keyframes reached

    Args:
        total_frames: Total number of frames available.
        num_keyframes: Number of frames to sample (will be clamped to total_frames).

    Returns:
        indices: List of frame indices to sample, sorted.

    Examples:
      >>> compute_keyframes_as_progressive_bisection(10, 1)
      [0]
      >>> compute_keyframes_as_progressive_bisection(10, 2)
      [0, 9]
      >>> compute_keyframes_as_progressive_bisection(10, 3)
      [0, 4, 9]
      >>> compute_keyframes_as_progressive_bisection(10, 5)
      [0, 2, 4, 7, 9]

    """
    num_keyframes = min(num_keyframes, total_frames)

    if num_keyframes == 0:
        return []

    # Start with first frame.
    indices: list[int] = [0]

    if num_keyframes == 1:
        return indices

    # Add last frame.
    last_idx = total_frames - 1
    indices.append(last_idx)

    if num_keyframes == 2:
        return sorted(indices)

    # Continue bisecting until we have enough frames.
    while len(indices) < num_keyframes:
        # Find the largest gap between consecutive indices.
        indices_sorted = sorted(indices)
        max_gap = 0
        max_gap_idx = 0

        for i in range(len(indices_sorted) - 1):
            gap = indices_sorted[i + 1] - indices_sorted[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i

        # Bisect the largest gap.
        left = indices_sorted[max_gap_idx]
        right = indices_sorted[max_gap_idx + 1]
        mid = (left + right) // 2

        # Only add if mid is different from both endpoints (avoid duplicates)
        if mid not in {left, right}:
            indices.append(mid)
        else:
            # No more unique frames to add.
            break

    return sorted(indices)


def sample_frame_indices(total_frames: int, num_frames: int) -> Tensor:
    """Sample evenly spaced frame indices from a video.

    Args:
      total_frames: Total number of frames available.
      num_frames: Number of frames to sample (will be clamped to total_frames).

    Returns:
      indices: Tensor of frame indices to sample, shape (num_frames,).

    """
    num_frames = min(num_frames, total_frames)

    if num_frames == 1:
        return torch.tensor([total_frames // 2])

    return torch.linspace(0, total_frames - 1, num_frames).long()


def as_image_batch_tensor(
    x: Tensorable,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Normalize image tensor to have format BFCHW or FCHW.

    Converts decoder format (C, F, H, W) to processor format (F, C, H, W).

    Args:
        x: Input tensor in decoder format:
           - (C, F, H, W) for unbatched
           - (B, C, F, H, W) for batched
           - (*, C, F, H, W) for arbitrary leading dims
        dtype: Optional dtype for output tensor.
        device: Optional device for output tensor.

    Returns:
        x: Tensor in processor format: (F, C, H, W) for unbatched,
            (B, F, C, H, W) for batched or flattened leading dims.

    Examples:
        >>> # Image (F=1): (C, F, H, W) → (F, C, H, W)
        >>> x = torch.zeros(3, 1, 224, 224)
        >>> as_image_batch_tensor(x).shape
        torch.Size([1, 3, 224, 224])

        >>> # Video: (C, F, H, W) → (F, C, H, W)
        >>> x = torch.zeros(3, 8, 224, 224)
        >>> as_image_batch_tensor(x).shape
        torch.Size([8, 3, 224, 224])

        >>> # Batched: (B, C, F, H, W) → (B, F, C, H, W)
        >>> x = torch.zeros(4, 3, 8, 224, 224)
        >>> as_image_batch_tensor(x).shape
        torch.Size([4, 8, 3, 224, 224])

    """
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.ndim < 4:
        raise ValueError(f"Too few dimensions; {x.shape=}.")
    x = x.moveaxis(-4, -3)  # *BFCHW.
    if x.ndim in (4, 5):
        return x  # BFCHW or FCHW.
    return x.reshape(-1, *x.shape[-4:])  # Always BFCHW.


def preprocess_images(
    x: Tensor,
    size: tuple[int, int],
    *,
    mean: Tensorable | None = None,
    std: Tensorable | None = None,
    mode: InterpolationMode = "hybrid",
    antialias: bool = False,
    align_corners: bool | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Preprocess images on GPU without CPU roundtrip.

    Replaces HuggingFace image processors that require CPU→GPU transfers.
    Performs resize and normalize entirely on GPU.

    Args:
        x: Input tensor (N, C, H, W) in float [-1, 1] range, on GPU.
        size: Target (height, width) for resizing.
        mean: Normalization mean per channel (list or pre-created tensor).
              If None, normalization is skipped.
        std: Normalization std per channel (list or pre-created tensor).
             If None, normalization is skipped.
        mode: Interpolation mode for resizing (default: "hybrid").
              "hybrid" uses "area" for downsampling, "bicubic" for upsampling.
              Can also specify "bilinear", "bicubic", "area", "lanczos", etc. directly.
              Note: "lanczos" uses torchvision's resize (supports Lanczos3).
        antialias: Whether to apply antialiasing. Silently ignored for every
                   mode torch does not accept it for -- only bilinear and
                   bicubic use it, area already averages, and Lanczos always
                   antialiases.
        align_corners: How to align corner pixels. Only applies to the
                       interpolating modes (linear, bilinear, bicubic,
                       trilinear); ignored for area/nearest/nearest-exact and
                       for hybrid downsampling (which resolves to area). If
                       None, uses the PyTorch default.
        dtype: Target dtype for output tensor. If None, preserves input dtype.

    Returns:
        x: Preprocessed tensor (N, C, H, W) in the specified dtype.

    """
    if not torch.is_floating_point(x):
        raise TypeError(f"Input tensor must be a float type, but got {x.dtype}")
    if x.ndim != 4:
        raise TypeError(f"Input format must be NCHW but {x.shape=}.")
    if dtype is None:
        dtype = x.dtype

    # Use input dtype for resize (no conversion overhead)
    resize_dtype = dtype

    # Resize (skip if already correct size)
    height, width = size
    if x.shape[-2:] != (height, width):
        if mode == "lanczos":
            if align_corners is not None:
                raise ValueError(
                    "align_corners is not supported with Lanczos interpolation "
                    "(PIL resize does not have this parameter)",
                )
            # Use PIL backend for Lanczos (returns fp32)
            pil_images = image_batch_to_pil_list(x)
            resized_samples: list[Tensor] = []
            for img in pil_images:
                img_resized = img.resize((width, height), PILImage.Resampling.LANCZOS)
                tensor_resized = tvf.to_tensor(img_resized)
                resized_samples.append(tensor_resized)
            x = torch.stack(resized_samples)
            x = (x * 2.0 - 1.0).to(resize_dtype)
        else:
            # Convert to resize dtype before resize.
            if x.dtype != resize_dtype:
                x = x.to(resize_dtype)

            if mode == "hybrid":
                source_pixels = x.shape[-2] * x.shape[-1]
                target_pixels = height * width
                if target_pixels < source_pixels:
                    interp_mode = "area"  # Downsampling.
                else:
                    interp_mode = "bicubic"  # Upsampling.
            else:
                interp_mode = mode

            # align_corners is only valid for the interpolating modes; PyTorch
            # rejects it for area/nearest/nearest-exact.
            effective_align_corners = (
                align_corners
                if interp_mode in ("linear", "bilinear", "bicubic", "trilinear")
                else None
            )

            # Same restriction, and torch enforces it by raising. Under
            # "hybrid" the mode is chosen per sample, so passing the flag
            # through unfiltered made an upsample succeed and a downsample of
            # the identical config fail -- area does its own averaging, which
            # is what antialiasing asks for anyway.
            effective_antialias = antialias and interp_mode in ("bilinear", "bicubic")

            x = torch.nn.functional.interpolate(
                x,
                size=(height, width),
                mode=interp_mode,
                antialias=effective_antialias,
                align_corners=effective_align_corners,
            )
    elif x.dtype != resize_dtype:
        # Convert to resize dtype if resize was skipped.
        x = x.to(resize_dtype)

    # Normalize after resize for efficiency when downsampling (common case).
    # Since normalization and interpolation are both linear operations, they commute.
    # Downsampling example: 1920x1080 → 224x224 reduces pixels by 40x before normalization.
    if mean is not None:
        mean_tensor = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(
            -1,
            1,
            1,
        )
        x = x - mean_tensor

    if std is not None:
        std_tensor = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        x = x / std_tensor

    # Convert to final target dtype if different from current dtype.
    if x.dtype != dtype:
        x = x.to(dtype)

    return x


def image_batch_to_pil_list(x: Tensor) -> list[PILImage.Image]:
    """Convert batch of images to list of PIL Images.

    Args:
        x: Input tensor (N, C, H, W) in float [-1, 1] range.

    Returns:
        images: List of PIL Image objects.

    """
    if not torch.is_floating_point(x):
        raise TypeError(f"Input tensor must be a float type, but got {x.dtype}")
    if x.ndim != 4:
        raise TypeError(f"Input format must be NCHW but {x.shape=}.")
    y = float2rgb(x)
    # Convert each image using tvf.to_pil_image (expects CHW)
    return [tvf.to_pil_image(y_) for y_ in y.unbind()]
