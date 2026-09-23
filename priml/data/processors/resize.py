"""Image and video resizing processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from configgle import Fig
from torch import Tensor

import torch

from priml.data.processors.utils import (
    InterpolationMode,
    as_image_batch_tensor,
    preprocess_images,
)
from priml.math.pixel import float2rgb, rgb2float


if TYPE_CHECKING:
    from collections.abc import Iterator

    import cv2
else:
    # Defer the ~120 ms OpenCV import to the first uint8 area resize.
    from wrapt import lazy_import

    cv2 = lazy_import("cv2")


__all__ = [
    "Interpolate",
]


class Interpolate:
    """Interpolate media_tensor to target dimensions.

    Assumes input already has correct aspect ratio (from CropDuringDecodeImage crop-during-decode).
    Just interpolates to exact target dimensions.

    Example:
        config = Interpolate.Config(mode="area")
        processor = config.make()

    """

    class Config(Fig["Interpolate"]):
        mode: InterpolationMode = "hybrid"
        """Interpolation mode ("hybrid", "area", "bilinear", etc.)."""

        align_corners: bool | None = None
        """Align corner pixels (None = mode default)."""

        antialias: bool = False
        """Apply antialiasing filter when downsampling."""

    class Input(TypedDict, total=False):
        """Input required by Interpolate."""

        media_tensor: Tensor
        """``(C, F, H, W)``, or ``(B, C, F, H, W)`` when batched."""

        target_height: int
        target_width: int

    class Output(Input, total=False):
        """Output produced by Interpolate.

        Inherits ``Input`` because the sample is passed through: this stage
        rewrites one field and forwards the rest, so a fresh TypedDict would
        declare every other key gone.
        """

        media_tensor: Tensor
        """The input resized to ``(..., target_height, target_width)``, in the
        dtype it arrived as."""

    def __init__(self, config: Config):
        self.mode: InterpolationMode = config.mode
        self.align_corners = config.align_corners
        self.antialias = config.antialias

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Resize media_tensor to target dimensions.

        Assumes input tensor already has correct aspect ratio (from CropDuringDecodeImage
        crop-during-decode). Just interpolates to exact target dimensions.

        Works on rightmost 2 dimensions (H, W) of (C, F, H, W) format.
        Supports batched tensors (B, C, F, H, W).

        Requires:
          - media_tensor: Tensor - (C, F, H, W) or (B, C, F, H, W)
          - target_height: int - target height
          - target_width: int - target width

        Modifies:
          - media_tensor: Resized to (C, F, target_height, target_width),
            in the dtype it arrived as

        Skips sample if required fields are missing.
        """
        for sample in samples:
            x = sample.get("media_tensor")
            target_height = sample.get("target_height")
            target_width = sample.get("target_width")

            if x is None or target_height is None or target_width is None:
                yield sample
                continue

            *batch_shape, channels, frames, height, width = x.shape
            if self._uses_opencv_area(x, target_height, target_width):
                sample["media_tensor"] = _opencv_area(x, target_height, target_width)
                yield sample
                continue

            # Convert (*B, C, F, H, W) → (prod(B)*F, C, H, W)
            x = as_image_batch_tensor(x)

            # Convert uint8 → float [-1, 1] for preprocess_images.
            input_was_uint8 = x.dtype == torch.uint8
            if input_was_uint8:
                x = rgb2float(x.to(torch.float16), inplace=True)

            x = x.reshape(-1, channels, height, width)
            x = preprocess_images(
                x,
                size=(target_height, target_width),
                mode=self.mode,
                align_corners=self.align_corners,
                antialias=self.antialias,
            )

            # Reshape back to original format.
            x = x.reshape(*batch_shape, frames, channels, target_height, target_width)
            x = x.moveaxis(-3, -4)  # Swap F and C back.

            if input_was_uint8:
                x = float2rgb(x, inplace=True)

            sample["media_tensor"] = x

            yield sample

    # A uint8 CPU downsample under area averaging is OpenCV's INTER_AREA on each
    # frame. The torch path widens to float16 and rounds back, measured 1.21 ms
    # against 0.3 per 160 px ImageNet crop -- more than the decode feeding it.
    def _uses_opencv_area(self, x: Tensor, height: int, width: int) -> bool:
        """Whether this resize is a uint8 CPU area downsample."""
        downsample = height * width < x.shape[-2] * x.shape[-1]
        area = self.mode == "area" or (self.mode == "hybrid" and downsample)
        return area and downsample and x.dtype == torch.uint8 and x.device.type == "cpu"


def _opencv_area(x: Tensor, height: int, width: int) -> Tensor:
    """Resize ``(..., C, F, H, W)`` uint8 with ``cv2.INTER_AREA``, frame by frame."""
    *batch_shape, channels, frames, _, _ = x.shape
    # (..., C, F, H, W) -> (N, H, W, C): OpenCV's interleaved layout.
    frames_hwc = x.reshape(-1, channels, frames, *x.shape[-2:]).permute(0, 2, 3, 4, 1)
    frames_hwc = frames_hwc.reshape(-1, *frames_hwc.shape[-3:])
    out = torch.empty(
        (len(frames_hwc), height, width, channels),
        dtype=torch.uint8,
    )
    for source, destination in zip(frames_hwc, out, strict=True):
        _ = cv2.resize(
            source.numpy(),
            (width, height),
            dst=destination.numpy(),
            interpolation=cv2.INTER_AREA,
        )
    return (
        out.reshape(-1, frames, height, width, channels)
        .permute(0, 4, 1, 2, 3)
        .reshape(*batch_shape, channels, frames, height, width)
        .contiguous()
    )
