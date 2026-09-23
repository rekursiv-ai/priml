"""Media shape analysis and resize dimension calculation.

Classes for tracking image statistics and calculating resize dimensions based on
aspect-ratio bucketing.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, TypedDict, cast

import logging
import math

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray
    from torch import Tensor

    import numpy as np
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")  # ~90 ms; shape reports and validation use it.

from priml.data.pipeline.dataset import add_filter_reason_typed
from priml.data.processors.utils import (
    compute_keyframes_as_progressive_bisection,
)
from priml.math.basic import ceil_div


logger = logging.getLogger(__name__)


__all__ = [
    "CalcResizeDimensions",
    "ImageShapeStatistics",
    "SetCropFromTargetDimensions",
    "SubsampleFramesViaBisection",
]


class ImageShapeStatistics:
    """Track raw pixel density and aspect ratio statistics, report deciles on deletion.

    Passthrough processor that collects image dimension statistics and outputs
    decile distributions when the processor is garbage collected or explicitly deleted.

    Tracks:
    - Pixel density (width * height)
    - Aspect ratio (width / height)

    Example:
        stats = ImageShapeStatistics.Config().make()

        for sample in stats(source):
            # sample passes through unchanged
            # stats are collected in background
            pass

        stats.report()  # Log the collected distributions explicitly.

    """

    class Config(Fig["ImageShapeStatistics"]):
        """Configuration for ImageShapeStatistics."""

        quantiles: int = 10
        """Buckets the reported distribution is split into (10 = deciles)."""

    class Input(TypedDict, total=False):
        """Input read by ImageShapeStatistics."""

        height: int
        width: int

    Output = Input

    def __init__(self, config: Config):
        if config.quantiles <= 0:
            raise ValueError(f"quantiles must be positive; got {config.quantiles}.")
        self.quantiles = config.quantiles
        self.pixel_densities: list[int] = []
        self.aspect_ratios: list[float] = []
        self.effective_resolutions: list[int] = []  # sqrt(w*h) rounded.

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Collect statistics and pass through samples unchanged.

        Requires:
          - width: int - Image width
          - height: int - Image height

        Adds:
          - (none) - passes through unchanged

        """
        for sample in samples:
            # ``width``/``height``, not ``original_*``: those are the pre-rename
            # names, so reading them made this collect nothing -- and report
            # nothing -- anywhere RenameFields had already run.
            width = sample.get("width")
            height = sample.get("height")

            # BOTH widths are guarded, not height alone: a negative width makes
            # the density negative, and ``(-n) ** 0.5`` is COMPLEX, which
            # ``round`` refuses -- so one malformed record killed the whole
            # pipeline from inside a passthrough that only collects statistics.
            if width is not None and height is not None and width > 0 and height > 0:
                # Track pixel density (total pixels)
                pixel_density = int(width) * int(height)
                self.pixel_densities.append(pixel_density)

                # Track effective resolution (sqrt of pixel density, rounded)
                effective_res = round((pixel_density) ** 0.5)
                self.effective_resolutions.append(effective_res)

                # Track aspect ratio (width / height)
                aspect_ratio = float(width) / float(height)
                self.aspect_ratios.append(aspect_ratio)

            yield sample

    def report(self) -> None:
        """Log quantiles and bucket recommendations for collected samples."""
        if not self.pixel_densities or not self.aspect_ratios:
            return

        logger.info(
            "ImageShapeStatistics: collected %d samples",
            len(self.pixel_densities),
        )

        # Compute and display pixel density analysis.
        if self.pixel_densities:
            self._log_pixel_density_analysis()

        # Compute and display aspect ratio analysis.
        if self.aspect_ratios:
            self._log_aspect_ratio_analysis()

    # Both analyses label their output with it, and each carried its own copy of this
    # table -- so adding a name meant remembering to edit two places that no test
    # compares.
    @property
    def _quantile_name(self) -> str:
        """What one bucket of the configured split is called."""
        return {
            4: "quartile",
            5: "quintile",
            10: "decile",
            20: "vigintile",
            100: "percentile",
        }.get(self.quantiles, f"{self.quantiles}-quantile")

    def _log_pixel_density_analysis(self) -> None:
        """Log pixel density statistics with bucket recommendations."""
        effective_res: NDArray[np.int64] = np.array(self.effective_resolutions)

        # Basic quantiles on effective resolution (sqrt of pixel area)
        percentiles = [100.0 * i / self.quantiles for i in range(self.quantiles + 1)]
        res_quantiles: NDArray[np.float64] = np.percentile(
            effective_res,
            percentiles,
        )

        logger.info(
            "Effective resolution %ss (round(sqrt(width * height))):",
            self._quantile_name,
        )
        res_values = cast(list[float], res_quantiles.tolist())
        for i, value in enumerate(res_values):
            pct = percentiles[i]
            logger.info("  %5.1f%%: %12.0f", pct, value)

        # Suggest resolution buckets for uniform coverage.
        logger.info("\nSuggested resolution buckets (for ~uniform coverage):")

        # All requested resolutions (sorted)
        reference_resolutions = sorted(
            [
                32,
                64,
                96,
                128,
                192,
                240,
                256,
                384,
                480,
                512,
                720,
                768,
                960,
                1024,
                1440,
                1536,
                2048,
                2160,
                2880,
                3072,
            ],
        )

        # Filter to relevant range.
        min_res = int(effective_res.min())
        max_res = int(effective_res.max())
        relevant_refs = [
            r for r in reference_resolutions if min_res <= r <= max_res * 1.2
        ]

        if relevant_refs:
            logger.info("  Bucket boundaries and coverage:")
            prev_res = 0
            for res in relevant_refs:
                count = np.sum((effective_res > prev_res) & (effective_res <= res))
                pct = 100.0 * count / len(effective_res)
                logger.info("    %d: %d samples (%.1f%%)", res, count, pct)
                prev_res = res
            # Final bucket for anything above.
            count = np.sum(effective_res > prev_res)
            if count > 0:
                pct = 100.0 * count / len(effective_res)
                logger.info(
                    "    >%d: %d samples (%.1f%%)",
                    relevant_refs[-1],
                    count,
                    pct,
                )

    def _log_aspect_ratio_analysis(self) -> None:
        """Log aspect ratio statistics with bucket recommendations."""
        ratios: NDArray[np.float64] = np.array(self.aspect_ratios)

        # Basic quantiles.
        percentiles = [100.0 * i / self.quantiles for i in range(self.quantiles + 1)]
        aspect_quantiles: NDArray[np.float64] = np.percentile(
            ratios,
            percentiles,
        )

        logger.info("\nAspect ratio %ss (width / height):", self._quantile_name)
        aspect_values = cast(list[float], aspect_quantiles.tolist())
        for i, value in enumerate(aspect_values):
            pct = percentiles[i]
            logger.info("  %5.1f%%: %8.4f", pct, value)

        # Suggest common aspect ratio buckets.
        logger.info("\nSuggested aspect ratio buckets (for ~uniform coverage):")

        # All requested aspect ratios as reference points (sorted)
        reference_ratios = [
            (1 / 3, "1:3"),
            (1 / 2, "1:2"),
            (9 / 16, "9:16"),
            (5 / 8, "5:8"),
            (2 / 3, "2:3"),
            (3 / 4, "3:4"),
            (4 / 5, "4:5"),
            (1 / 1, "1:1 (square)"),
            (5 / 4, "5:4"),
            (4 / 3, "4:3"),
            (3 / 2, "3:2"),
            (8 / 5, "8:5"),
            (5 / 3, "5:3"),
            (16 / 9, "16:9"),
            (2 / 1, "2:1"),
            (2.6, "2.6:1"),
            (3 / 1, "3:1"),
        ]

        # Filter to relevant range and find bucket boundaries.
        min_ratio = float(ratios.min())
        max_ratio = float(ratios.max())
        relevant_refs = [
            (r, name)
            for r, name in reference_ratios
            if min_ratio * 0.8 <= r <= max_ratio * 1.2
        ]

        if relevant_refs:
            logger.info("  Bucket boundaries and coverage:")
            prev_ratio = 0.0
            for ratio, name in relevant_refs:
                count = np.sum((ratios > prev_ratio) & (ratios <= ratio))
                pct = 100.0 * count / len(ratios)
                logger.info(
                    "    %s (%.4f): %d samples (%.1f%%)",
                    name,
                    ratio,
                    count,
                    pct,
                )
                prev_ratio = ratio
            # Final bucket for anything above.
            count = np.sum(ratios > prev_ratio)
            if count > 0:
                pct = 100.0 * count / len(ratios)
                logger.info(
                    "    >%s: %d samples (%.1f%%)",
                    relevant_refs[-1][1],
                    count,
                    pct,
                )


class CalcResizeDimensions:
    """Calculate target resize dimensions using aspect-ratio bucketing.

    This processor calculates target dimensions without resizing. Each
    resolution names the square side with the same pixel area; aspect buckets
    preserve that area before rounding to compression strides.

    This processor only adds metadata fields - actual resizing should be
    performed by a downstream processor (e.g. CropDuringDecodeImage,
    DecodeVideo).

    Example:
        # Before: sample with original dimensions
        sample = {"key": "a", "frames": 1, "width": 1920, "height": 1080}

        # After CalcResizeDimensions: adds target dimensions
        sample = {
            "key": "a",
            "frames": 1,
            "width": 1920,
            "height": 1080,
            "target_frames": 1,
            "target_height": 720,
            "target_width": 1280,
        }

    """

    class Config(Fig["CalcResizeDimensions"]):
        square_resolutions: list[int] = field(
            default_factory=lambda: [120, 160, 240],
        )
        """Geometric-mean side lengths defining pixel-area buckets."""

        aspects: list[float] = field(
            default_factory=lambda: [
                # 16 / 9,  # 9.
                2 / 1,  # 8.
                8 / 5,  # 5.
                5 / 4,  # 7.
                4 / 3,  # 3.
                3 / 2,  # 2.
                1 / 1,  # 1.
                2 / 3,  # 6.
                3 / 4,  # 4.
                4 / 5,  # 10.
                # 5 / 8,   # ?
                # 2 / 3,
                # 9 / 16.
            ],
        )
        """Width/height ratios a sample is bucketed into, nearest first."""

        compression: tuple[int, int, int] = (1, 16, 16)
        """Latent downsampling per (frame, height, width); target dims are
        rounded up to a multiple of it."""

    class Input(TypedDict, total=False):
        """Input required by CalcResizeDimensions."""

        frames: int
        height: int
        width: int

    class Output(Input):
        """Output produced by CalcResizeDimensions."""

        target_frames: int
        target_height: int
        target_width: int

    def __init__(self, config: Config):
        if not config.square_resolutions:
            raise ValueError("square_resolutions must name at least one resolution.")
        if any(resolution < 1 for resolution in config.square_resolutions):
            raise ValueError("square_resolutions must be positive.")
        if not config.aspects:
            raise ValueError("aspects must name at least one aspect ratio.")
        if any(aspect <= 0 or not math.isfinite(aspect) for aspect in config.aspects):
            raise ValueError("aspects must be finite and positive.")
        if any(c < 1 for c in config.compression):
            raise ValueError(
                f"compression strides must be positive; got {config.compression}.",
            )
        self.square_resolutions = tuple(config.square_resolutions)
        self.aspects = tuple(config.aspects)
        self.compression = config.compression

        _, compression_height, compression_width = self.compression
        self.reference_areas = {
            resolution: resolution * resolution
            for resolution in self.square_resolutions
        }
        self.aspect_buckets: dict[int, list[tuple[int, int]]] = {}
        for resolution in self.square_resolutions:
            buckets: list[tuple[int, int]] = []
            for aspect in self.aspects:
                aspect_sqrt = aspect**0.5
                height = round(resolution / aspect_sqrt)
                width = round(resolution * aspect_sqrt)
                buckets.append(
                    (
                        ceil_div(height, compression_height) * compression_height,
                        ceil_div(width, compression_width) * compression_width,
                    ),
                )
            self.aspect_buckets[resolution] = buckets

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Calculate target resize dimensions for each sample.

        Uses two-stage bucketing:
        1. Find the nearest square-equivalent pixel area.
        2. Find the nearest configured aspect within that area bucket.

        Requires:
          - frames: int (optional) - number of frames (defaults to 1)
          - width: int - original sample width
          - height: int - original sample height

        Adds:
          - target_frames: int - frame count rounded up to the temporal
            compression multiple; 1 for a still
          - target_height: int - target height after resizing
          - target_width: int - target width after resizing

        """
        for sample in samples:
            sample = cast(CalcResizeDimensions.Output, sample)
            result = self._extract_fields(sample)
            if result is None:
                yield sample
                continue
            frames, height, width = result

            # Pick the nearest square-equivalent area, then the nearest aspect.
            input_area = height * width
            target_resolution = min(
                self.square_resolutions,
                key=lambda resolution: abs(
                    self.reference_areas[resolution] - input_area,
                ),
            )

            # Step 2: Within that resolution, find nearest aspect ratio.
            height_and_width_pairs = self.aspect_buckets[target_resolution]
            input_aspect = width / height
            target_height, target_width = min(
                height_and_width_pairs,
                key=lambda h_w: abs(h_w[1] / h_w[0] - input_aspect),
            )

            compression_frames = self.compression[0]
            sample["target_frames"] = (
                1
                if frames <= 1
                else ceil_div(frames, compression_frames) * compression_frames
            )
            sample["target_height"] = target_height
            sample["target_width"] = target_width
            yield sample

    def _extract_fields(
        self,
        sample: CalcResizeDimensions.Output,
    ) -> tuple[int, int, int] | None:
        """Extract and validate dimension fields from sample."""
        frames = sample.get("frames", 1)
        height = sample.get("height")
        width = sample.get("width")

        # Check for None values.
        if height is None or width is None:
            add_filter_reason_typed(
                sample,
                type(self).__name__,
                "missing_dimensions",
            )
            return None

        # NaN and inf are checked before the int conversion below, which would
        # otherwise raise rather than filter. Both reach here despite the int
        # annotation: a sample is parsed data, not a constructed TypedDict.
        if math.isnan(frames) or math.isnan(height) or math.isnan(width):
            add_filter_reason_typed(
                sample,
                type(self).__name__,
                "nan_dimensions",
            )
            return None

        if math.isinf(frames) or math.isinf(height) or math.isinf(width):
            add_filter_reason_typed(
                sample,
                type(self).__name__,
                "inf_dimensions",
            )
            return None

        # Check for invalid values.
        if frames <= 0 or height <= 0 or width <= 0:
            add_filter_reason_typed(
                sample,
                type(self).__name__,
                f"invalid_dimensions:f={frames}_h={height}_w={width}",
            )
            return None

        return (int(frames), int(height), int(width))


class SubsampleFramesViaBisection:
    """Subsample video frames using progressive bisection strategy.

    Progressively adds frames in priority order:
    1. First frame (index 0)
    2. Last frame (index total_frames - 1)
    3. Middle frame (bisect between first and last)
    4. Continue bisecting largest gaps until num_frames reached

    This ensures temporal coverage with priority on endpoints and middle,
    making it ideal for capturing key moments across the full video timeline.

    Example:
        # 10 frames, request 5: [0, 2, 4, 7, 9]
        config = SubsampleFramesViaBisection.Config(max_num_keyframes=5)
        processor = config.make()

        sample = {"media_tensor": torch.randn(3, 10, 224, 224)}
        result = next(processor([sample]))
        # result["media_tensor"].shape == (3, 5, 224, 224)
        # result["keyframes"] == [0, 2, 4, 7, 9]

    """

    class Config(Fig["SubsampleFramesViaBisection"]):
        max_num_keyframes: int = 5
        """Frames kept per clip, chosen by progressive bisection."""

    class Input(TypedDict, total=False):
        """Input required by SubsampleFramesViaBisection."""

        media_tensor: Tensor

    class Output(Input, total=False):
        """Output produced by SubsampleFramesViaBisection."""

        keyframes: list[int]

    def __init__(self, config: Config):
        # Rejected here rather than at the first sample: a non-positive count
        # yields a tensor with an EMPTY frame axis, which fails inside whatever
        # consumes it -- naming a shape instead of the field that produced it.
        if config.max_num_keyframes < 1:
            raise ValueError(
                f"max_num_keyframes must be positive; got {config.max_num_keyframes}.",
            )
        self.max_num_keyframes = config.max_num_keyframes

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Subsample frames from media_tensor using bisection strategy.

        Requires:
          - media_tensor: Tensor - decoder layout [*B, C, F, H, W], F = frame count

        Adds/Modifies:
          - media_tensor: Tensor - [*B, C, max_num_keyframes, H, W], subsampled
          - keyframes: list[int] - Sorted list of selected frame indices

        """
        for sample in samples:
            sample = cast(SubsampleFramesViaBisection.Output, sample)
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue

            # Decoder layout (*B, C, F, H, W): F is the frame axis at -3.
            total_frames = x.shape[-3]

            # Skip subsampling for single-frame inputs (images)
            # Avoids creating unnecessary tensor copy in ParMap output queue.
            if total_frames == 1:
                sample["keyframes"] = [0]
                yield sample
                continue

            idx = compute_keyframes_as_progressive_bisection(
                total_frames=total_frames,
                num_keyframes=self.max_num_keyframes,
            )

            # Skip if all frames selected in order (no actual subsampling needed)
            if idx == list(range(total_frames)):
                sample["keyframes"] = idx
                yield sample
                continue

            # Perform actual subsampling on the frame axis via advanced indexing.
            y = x[..., idx, :, :]
            sample["media_tensor"] = y
            sample["keyframes"] = idx
            yield sample


class SetCropFromTargetDimensions:
    """Convert target_height/target_width fields to crop field.

    Reads target_height and target_width, sets crop = (target_height, target_width).
    Used with CropDuringDecodeImage which expects crop field.
    """

    class Config(Fig["SetCropFromTargetDimensions"]):
        pass

    def __init__(self, config: Config):
        del config

    class Input(TypedDict, total=False):
        target_height: int
        target_width: int

    class Output(Input):
        """Adds ``crop`` while forwarding every input field.

        Inherits ``Input`` because the sample really is passed through: the
        immediate consumer, ``CropDuringDecodeImage``, reads ``height`` and
        ``width``, and a fresh TypedDict would declare those gone.
        """

        crop: tuple[int, int]

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample_in in samples:
            sample = cast(SetCropFromTargetDimensions.Output, sample_in)
            target_height = sample_in.get("target_height")
            target_width = sample_in.get("target_width")

            if target_height is not None and target_width is not None:
                sample["crop"] = (target_height, target_width)

            yield sample
