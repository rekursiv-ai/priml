"""Tests for shapes module."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import torch

from priml.data.processors.shapes import (
    CalcResizeDimensions,
    ImageShapeStatistics,
    SetCropFromTargetDimensions,
    SubsampleFramesViaBisection,
)


def test_image_shape_statistics_rejects_a_nonpositive_quantile_count():
    with pytest.raises(ValueError, match="quantiles must be positive"):
        _ = ImageShapeStatistics.Config(quantiles=0).make()


def test_image_shape_statistics_report_is_silent_without_samples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stats = ImageShapeStatistics.Config().make()
    caplog.set_level("INFO")

    stats.report()

    assert caplog.text == ""


def test_image_shape_statistics_report_buckets_resolutions_and_aspects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stats = ImageShapeStatistics.Config(quantiles=4).make()
    # Effective resolutions 64, 128, 181, 181 and 3464 (above every reference
    # bucket), with aspects spanning 1:2 through 2:1 and one wider than 3:1.
    samples: list[ImageShapeStatistics.Input] = [
        {"width": 64, "height": 64},
        {"width": 128, "height": 128},
        {"width": 256, "height": 128},
        {"width": 128, "height": 256},
        {"width": 12_000, "height": 1_000},
    ]
    _ = list(stats(iter(samples)))
    caplog.set_level("INFO")

    stats.report()

    assert "Effective resolution quartiles" in caplog.text
    assert "    64: 1 samples (20.0%)" in caplog.text
    assert "    >3072: 1 samples (20.0%)" in caplog.text
    assert "Aspect ratio quartiles" in caplog.text
    assert "    1:1 (square) (1.0000): 2 samples (40.0%)" in caplog.text
    assert "    2:1 (2.0000): 1 samples (20.0%)" in caplog.text
    assert "    >3:1: 1 samples (20.0%)" in caplog.text


def test_image_shape_statistics_names_an_uncommon_quantile_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stats = ImageShapeStatistics.Config(quantiles=3).make()
    samples: list[ImageShapeStatistics.Input] = [{"width": 8, "height": 8}]
    _ = list(stats(iter(samples)))
    caplog.set_level("INFO")

    stats.report()

    assert "Effective resolution 3-quantiles" in caplog.text
    assert "Aspect ratio 3-quantiles" in caplog.text


def test_subsample_frames_keeps_a_clip_that_already_fits():
    processor = SubsampleFramesViaBisection.Config(max_num_keyframes=5).make()

    x = torch.arange(3 * 3 * 2 * 2).reshape(3, 3, 2, 2).float()
    still = torch.zeros(3, 1, 2, 2)
    samples: list[SubsampleFramesViaBisection.Input] = [
        {"media_tensor": x},
        {"media_tensor": still},
        {"media_tensor": x.clone()},
    ]
    results = list(processor(iter(samples)))

    assert [r.get("keyframes") for r in results] == [[0, 1, 2], [0], [0, 1, 2]]
    assert results[0].get("media_tensor") is x
    assert results[1].get("media_tensor") is still


def test_subsample_frames_passes_samples_without_media_through():
    processor = SubsampleFramesViaBisection.Config(max_num_keyframes=5).make()

    samples: list[SubsampleFramesViaBisection.Input] = [{}, {}]
    results = [dict(r) for r in processor(iter(samples))]

    assert results == [{}, {}]


def test_set_crop_from_target_dimensions_sets_crop_only_when_both_are_present():
    processor = SetCropFromTargetDimensions.Config().make()

    samples: list[SetCropFromTargetDimensions.Input] = [
        {"target_height": 4, "target_width": 8},
        {"target_height": 4},
        {},
    ]
    results = list(processor(iter(samples)))

    assert results[0]["crop"] == (4, 8)
    assert results[0].get("target_height") == 4
    assert "crop" not in results[1]
    assert "crop" not in results[2]


def test_resize_dimension_calculator_area_based_bucketing():
    """Test that CalcResizeDimensions uses area-based bucketing."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240, 480, 720],
        aspects=[2.0, 1.0, 0.5],  # Wide, square, tall.
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # A 400x400 square is nearest the 480x480 area bucket.
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 400, "width": 400},
    ]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    r = result[0]
    # 240² is 57,600; 480² is 230,400 and nearest; 720² is 518,400.
    assert r["target_height"] == 480
    assert r["target_width"] == 480

    # A 1000x500 image is nearest the 720² area bucket.
    samples = [{"frames": 1, "height": 500, "width": 1000}]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    r = result[0]
    assert r["target_width"] > r["target_height"]
    assert 400 <= r["target_height"] <= 600
    assert 800 <= r["target_width"] <= 1200

    # The same area bucket selects the reciprocal portrait aspect.
    samples = [{"frames": 1, "height": 1000, "width": 500}]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    r = result[0]
    assert r["target_height"] > r["target_width"]
    assert 400 <= r["target_width"] <= 600
    assert 800 <= r["target_height"] <= 1200


def test_resize_dimension_calculator_square_image():
    """Test CalcResizeDimensions with square image."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Square image.
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 512, "width": 512},
    ]
    result = list(calc(iter(samples)))

    assert len(result) == 1
    r = result[0]
    assert r["target_frames"] == 1
    assert (r["target_height"], r["target_width"]) == (256, 256)
    # Should be divisible by compression factor.
    assert r["target_height"] % 16 == 0
    assert r["target_width"] % 16 == 0


def test_resize_dimension_calculator_landscape_image():
    """Test CalcResizeDimensions with landscape image (wider than tall)."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[16 / 9],  # Wide aspect ratio.
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Landscape 16:9 image (1920x1080)
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 1080, "width": 1920},
    ]
    result = list(calc(iter(samples)))

    assert len(result) == 1
    r = result[0]
    assert r["target_frames"] == 1

    # Landscape input should match landscape bucket.
    resize_h = r["target_height"]
    resize_w = r["target_width"]

    # 1920x1080 (w/h=1.778) should match landscape bucket with aspect 16/9~1.778.
    assert resize_w > resize_h, "Expected landscape output for landscape input"

    # Should be divisible by compression factor.
    assert resize_h % 16 == 0
    assert resize_w % 16 == 0


def test_resize_dimension_calculator_portrait_image():
    """Test CalcResizeDimensions with portrait image (taller than wide)."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[9 / 16],  # Tall aspect ratio.
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Portrait 9:16 image (1080x1920)
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 1920, "width": 1080},
    ]
    result = list(calc(iter(samples)))

    assert len(result) == 1
    r = result[0]
    assert r["target_frames"] == 1

    resize_h = r["target_height"]
    resize_w = r["target_width"]

    # Portrait input (w/h=0.5625) should match portrait bucket with aspect 9/16=0.5625.
    assert resize_h > resize_w, "Expected portrait output for portrait input"

    # Should be divisible by compression factor.
    assert resize_h % 16 == 0
    assert resize_w % 16 == 0


def test_resize_dimension_calculator_multiple_aspect_ratios():
    """Test CalcResizeDimensions with multiple aspect ratios."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[16 / 9, 4 / 3, 1.0, 3 / 4, 9 / 16],
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Test various inputs.
    test_cases: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 1920, "width": 1080},  # Portrait.
        {"frames": 1, "height": 1080, "width": 1920},  # Landscape.
        {"frames": 1, "height": 512, "width": 512},  # Square.
        {"frames": 1, "height": 1024, "width": 768},  # Portrait 4:3.
        {"frames": 1, "height": 768, "width": 1024},  # Landscape 4:3.
    ]

    for sample in test_cases:
        result = list(calc(iter([sample])))
        assert len(result) == 1
        r = result[0]
        assert r["target_frames"] == 1
        assert r["target_height"] % 16 == 0
        assert r["target_width"] % 16 == 0
        # Output should have reasonable dimensions.
        assert 100 <= r["target_height"] <= 1000
        assert 100 <= r["target_width"] <= 1000


def test_resize_dimension_calculator_multiple_resolutions():
    """Choose among multiple square-equivalent area buckets."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[180, 360, 540, 720],
        aspects=[16 / 9, 1.0, 9 / 16],
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Should create buckets for each resolution.
    assert len(calc.aspect_buckets) == 4
    # Each resolution should have buckets for each aspect.
    assert all(len(buckets) == 3 for buckets in calc.aspect_buckets.values())

    # Test that different inputs get reasonable outputs.
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 2160, "width": 3840},
    ]  # 4K.
    result = list(calc(iter(samples)))

    assert len(result) == 1
    r = result[0]
    assert r["target_frames"] == 1
    assert r["target_height"] % 16 == 0
    assert r["target_width"] % 16 == 0


def test_resize_dimension_calculator_frames_passthrough():
    """A single-frame sample is an image and keeps frames=1.

    Images are exempt from temporal rounding: the processor pins
    ``f=1`` for zero-duration input, so padding one frame up to a temporal
    compression multiple would invent frames that do not exist.
    """
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(8, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 512, "width": 512},
    ]
    result = list(calc(iter(samples)))
    r = result[0]
    assert r["target_frames"] == 1


def test_target_frames_round_up_to_the_temporal_compression():
    """Video frame counts round up to a multiple of the temporal factor.

    The height and width axes were already rounded, but ``target_frames`` was
    passed through untouched -- so a latent encoder with temporal stride 8
    received a 30-frame clip it cannot evenly encode. This is the same
    ``ceil_div(f, comp_f) * comp_f`` used for encoder alignment.
    """
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(8, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 30, "height": 512, "width": 512},
    ]
    result = list(calc(iter(samples)))
    r = result[0]
    assert r["target_frames"] == 32
    assert r["target_frames"] % 8 == 0


def test_target_frames_are_unchanged_when_already_aligned():
    """Rounding is a ceiling, so an aligned count must not grow."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(8, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 16, "height": 512, "width": 512},
    ]
    result = list(calc(iter(samples)))
    r = result[0]
    assert r["target_frames"] == 16


def test_unit_temporal_compression_keeps_every_frame_count():
    """With a factor of 1 every count is already a multiple, so nothing moves."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 30, "height": 512, "width": 512},
    ]
    result = list(calc(iter(samples)))
    r = result[0]
    assert r["target_frames"] == 30


def test_resize_dimension_calculator_compression_factor():
    """Test that output dimensions respect compression factor."""
    # Test with different compression factors.
    for mult_h, mult_w in [(8, 8), (16, 16), (32, 32)]:
        config = CalcResizeDimensions.Config(
            square_resolutions=[256],
            aspects=[1.0],
            compression=(1, mult_h, mult_w),
        )
        calc = CalcResizeDimensions(config)

        samples: list[CalcResizeDimensions.Input] = [
            {"frames": 1, "height": 512, "width": 512},
        ]
        result = list(calc(iter(samples)))

        r = result[0]
        # Output should be divisible by compression factor.
        assert r["target_height"] % mult_h == 0
        assert r["target_width"] % mult_w == 0


def test_resize_dimension_calculator_selects_closest_bucket():
    """Test that CalcResizeDimensions selects closest aspect ratio bucket."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[2.0, 1.0, 0.5],  # Wide, square, tall.
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # Square input should match square bucket.
    samples: list[CalcResizeDimensions.Input] = [
        {"frames": 1, "height": 1000, "width": 1000},
    ]
    result = list(calc(iter(samples)))
    r = result[0]
    assert r["target_height"] == r["target_width"]

    # Wide input (h=1000, w=2000, w/h=2.0) should match wide bucket (aspect=2.0)
    samples = [{"frames": 1, "height": 1000, "width": 2000}]
    result = list(calc(iter(samples)))
    r = result[0]
    # Result should be wide (w > h) because input is wide.
    assert r["target_width"] > r["target_height"]

    # Tall input (h=2000, w=1000, w/h=0.5) should match tall bucket (aspect=0.5)
    samples = [{"frames": 1, "height": 2000, "width": 1000}]
    result = list(calc(iter(samples)))
    r = result[0]
    # Result should be tall (h > w) because input is tall.
    assert r["target_height"] > r["target_width"]


def test_resize_dimension_calculator_filters_invalid_inputs():
    """Test that CalcResizeDimensions filters invalid inputs."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[256],
        aspects=[1.0],
        compression=(1, 16, 16),
    )
    calc = CalcResizeDimensions(config)

    # ABSENT dimensions.
    samples: list[CalcResizeDimensions.Input] = [{"frames": 1}]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    assert "filter_reasons" in result[0]

    # Zero dimensions.
    samples = [{"frames": 1, "height": 0, "width": 512}]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    assert "filter_reasons" in result[0]

    # Negative dimensions.
    samples = [{"frames": 1, "height": -100, "width": 512}]
    result = list(calc(iter(samples)))
    assert len(result) == 1
    assert "filter_reasons" in result[0]


# Tests from media_shape_test.py merged below:


def test_image_shape_statistics_basic():
    """Test ImageShapeStatistics collects and processes statistics."""
    config = ImageShapeStatistics.Config(quantiles=4)
    stats = ImageShapeStatistics(config)

    # Create test samples with varying dimensions.
    samples: list[ImageShapeStatistics.Input] = [
        {"width": 1920, "height": 1080},  # 16:9.
        {"width": 1280, "height": 720},  # 16:9.
        {"width": 800, "height": 600},  # 4:3.
        {"width": 640, "height": 480},  # 4:3.
        {"width": 1024, "height": 1024},  # 1:1.
    ]

    # Process samples.
    results = list(stats(iter(samples)))

    # Verify samples pass through unchanged.
    assert len(results) == 5
    assert results[0] == samples[0]
    assert results[4] == samples[4]

    # Verify statistics were collected.
    assert len(stats.pixel_densities) == 5
    assert len(stats.aspect_ratios) == 5
    assert len(stats.effective_resolutions) == 5

    # Verify pixel densities are correct.
    assert stats.pixel_densities[0] == 1920 * 1080
    assert stats.pixel_densities[1] == 1280 * 720
    assert stats.pixel_densities[4] == 1024 * 1024

    # Verify aspect ratios are correct.
    assert abs(stats.aspect_ratios[0] - 1920 / 1080) < 0.001
    assert abs(stats.aspect_ratios[2] - 800 / 600) < 0.001
    assert abs(stats.aspect_ratios[4] - 1.0) < 0.001

    # Verify effective resolutions (sqrt of pixel density)
    assert stats.effective_resolutions[0] == round((1920 * 1080) ** 0.5)
    assert stats.effective_resolutions[4] == 1024


def test_image_shape_statistics_reports_explicitly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stats = ImageShapeStatistics.Config(quantiles=4).make()
    samples: list[ImageShapeStatistics.Input] = [{"width": 4, "height": 4}]
    _ = list(stats(iter(samples)))
    caplog.set_level("INFO")

    stats.report()

    assert "ImageShapeStatistics: collected 1 samples" in caplog.text


def test_image_shape_statistics_missing_dimensions():
    """Test ImageShapeStatistics handles missing dimensions gracefully."""
    stats = ImageShapeStatistics.Config().make()

    samples: list[ImageShapeStatistics.Input] = [
        {"width": 1920, "height": 1080},
        {"width": 1280},  # ABSENT height.
        {"height": 720},  # ABSENT width.
        {},  # ABSENT both.
        {"width": 640, "height": 480},
    ]

    _results = list(stats(iter(samples)))

    # Only valid samples are tracked.
    assert len(stats.pixel_densities) == 2
    assert len(stats.aspect_ratios) == 2


def test_image_shape_statistics_zero_height():
    """Test ImageShapeStatistics handles zero height."""
    stats = ImageShapeStatistics.Config().make()

    samples: list[ImageShapeStatistics.Input] = [
        {"width": 1920, "height": 0},  # Zero height.
        {"width": 1280, "height": 720},  # Valid.
    ]

    _results = list(stats(iter(samples)))

    # Only valid sample is tracked.
    assert len(stats.pixel_densities) == 1
    assert stats.pixel_densities[0] == 1280 * 720


def test_image_shape_statistics_skips_a_negative_width():
    """A negative width is skipped, not crashed on.

    Only ``height`` was guarded, so a negative width made ``width * height``
    negative and ``(-n) ** 0.5`` a COMPLEX number -- which ``round`` refuses
    with ``TypeError: type complex doesn't define __round__``. A statistics
    passthrough killing the pipeline over one malformed record is the opposite
    of what it is for; the sibling zero-height case above already skips.
    """
    stats = ImageShapeStatistics.Config().make()

    samples: list[ImageShapeStatistics.Input] = [
        {"width": -1920, "height": 1080},
        {"width": 1280, "height": 720},
    ]

    _results = list(stats(iter(samples)))

    assert len(stats.pixel_densities) == 1
    assert stats.pixel_densities[0] == 1280 * 720


def test_subsample_frames_rejects_a_nonpositive_keyframe_count():
    """Zero keyframes must be refused at construction, not yield zero frames.

    ``max_num_keyframes=0`` built fine and emitted a tensor with an EMPTY
    frame axis, which fails later inside whatever consumes it -- naming a
    shape rather than the config field that produced it.
    """
    config = SubsampleFramesViaBisection.Config()
    config.max_num_keyframes = 0

    with pytest.raises(ValueError, match="max_num_keyframes"):
        _ = config.make()


def test_calc_resize_dimensions_basic():
    """Test CalcResizeDimensions calculates correct dimensions."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240, 480],
        aspects=[16 / 9, 4 / 3, 1 / 1, 3 / 4, 9 / 16],
    )
    calc = CalcResizeDimensions(config)

    # Test 16:9 widescreen (1920x1080) - should match nearest aspect.
    sample: CalcResizeDimensions.Input = {
        "frames": 1,
        "width": 1920,
        "height": 1080,
    }
    result = next(iter(calc(iter([sample]))))

    assert "target_frames" in result
    assert "target_height" in result
    assert "target_width" in result
    assert result["target_frames"] == 1
    # Should pick 480 resolution (closer to 1920*1080 area)
    # and 16:9 aspect ratio.
    assert result["target_width"] > result["target_height"]


def test_calc_resize_dimensions_square():
    """Test CalcResizeDimensions handles square images."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240],
        aspects=[1 / 1],
    )
    calc = CalcResizeDimensions(config)

    sample: CalcResizeDimensions.Input = {"width": 1024, "height": 1024}
    result = next(iter(calc(iter([sample]))))

    assert "target_width" in result
    assert "target_height" in result
    # For square aspect, width should equal height.
    assert result["target_width"] == result["target_height"]


def test_calc_resize_dimensions_portrait():
    """Test CalcResizeDimensions handles portrait (tall) images."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240],
        aspects=[16 / 9, 9 / 16],
    )
    calc = CalcResizeDimensions(config)

    # Portrait image (taller than wide)
    sample: CalcResizeDimensions.Input = {"width": 1080, "height": 1920}
    result = next(iter(calc(iter([sample]))))

    # Should pick 9:16 aspect (portrait)
    assert result["target_height"] > result["target_width"]


def test_calc_resize_dimensions_missing_dimensions():
    """Test CalcResizeDimensions handles missing dimensions."""
    calc = CalcResizeDimensions.Config().make()

    samples: list[CalcResizeDimensions.Input] = [
        {"width": 1920},  # ABSENT height.
        {"height": 1080},  # ABSENT width.
        {},  # ABSENT both.
    ]

    results = list(calc(iter(samples)))

    # All samples should have filter_reasons added.
    assert len(results) == 3
    assert "filter_reasons" in results[0]
    assert "filter_reasons" in results[1]
    assert "filter_reasons" in results[2]


def test_calc_resize_dimensions_invalid_dimensions():
    """Test CalcResizeDimensions handles invalid dimensions."""
    calc = CalcResizeDimensions.Config().make()

    samples: list[CalcResizeDimensions.Input] = [
        {"width": 0, "height": 1080},  # Zero width.
        {"width": 1920, "height": 0},  # Zero height.
        {"width": -1920, "height": 1080},  # Negative width.
        {"width": 1920, "height": -1080},  # Negative height.
    ]

    results = list(calc(iter(samples)))

    # All should have filter reasons.
    for result in results:
        assert "filter_reasons" in result


def test_calc_resize_dimensions_nan_dimensions():
    """Test CalcResizeDimensions handles NaN dimensions."""
    calc = CalcResizeDimensions.Config().make()

    # Deliberately out-of-contract: `width`/`height` are `int` per `Input`, but
    # the processor's job is to defend against dirty NaN/inf dimensions, so the
    # test feeds values the type system (correctly) rejects.
    samples: list[dict[str, object]] = [
        {"width": float("nan"), "height": 1080},
        {"width": 1920, "height": float("nan")},
    ]

    results = list(calc(cast(Iterator[CalcResizeDimensions.Input], iter(samples))))

    for result in results:
        assert "filter_reasons" in result


def test_calc_resize_dimensions_inf_dimensions():
    """Test CalcResizeDimensions handles infinity dimensions."""
    calc = CalcResizeDimensions.Config().make()

    samples: list[dict[str, object]] = [
        {"width": float("inf"), "height": 1080},
        {"width": 1920, "height": float("inf")},
    ]

    results = list(calc(cast(Iterator[CalcResizeDimensions.Input], iter(samples))))

    for result in results:
        assert "filter_reasons" in result


def test_calc_resize_dimensions_float_dimensions():
    """Test CalcResizeDimensions handles float dimensions."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240],
        aspects=[16 / 9],
    )
    calc = CalcResizeDimensions(config)

    # Float dimensions should be converted to int. The floats are deliberate:
    # a sample is parsed data, so it reaches here despite the ``int``
    # annotation -- which is why the input is cast rather than suppressed, the
    # same way the sibling test above hands its samples to ``calc``.
    sample: dict[str, object] = {"width": 1920.5, "height": 1080.7}
    samples = cast(Iterator[CalcResizeDimensions.Input], iter([sample]))
    result = next(iter(calc(samples)))

    assert "target_width" in result
    assert "target_height" in result
    assert isinstance(result["target_width"], int)
    assert isinstance(result["target_height"], int)


def test_calc_resize_dimensions_defaults_frames():
    """Test CalcResizeDimensions defaults frames to 1."""
    calc = CalcResizeDimensions.Config().make()

    # No frames field provided.
    sample: CalcResizeDimensions.Input = {"width": 1920, "height": 1080}
    result = next(iter(calc(iter([sample]))))

    assert result["target_frames"] == 1


def test_calc_resize_dimensions_preserves_frames():
    """Test CalcResizeDimensions preserves custom frame count."""
    calc = CalcResizeDimensions.Config().make()

    sample: CalcResizeDimensions.Input = {
        "frames": 30,
        "width": 1920,
        "height": 1080,
    }
    result = next(iter(calc(iter([sample]))))

    assert result["target_frames"] == 30


def test_calc_resize_dimensions_multiple_resolutions():
    """Test CalcResizeDimensions picks closest resolution."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[120, 240, 480],
        aspects=[1 / 1],
    )
    calc = CalcResizeDimensions(config)

    # Small image - should pick 120.
    sample1: CalcResizeDimensions.Input = {"width": 100, "height": 100}
    result1 = next(iter(calc(iter([sample1]))))

    # Medium image - should pick 240.
    sample2: CalcResizeDimensions.Input = {"width": 300, "height": 300}
    result2 = next(iter(calc(iter([sample2]))))

    # Large image - should pick 480.
    sample3: CalcResizeDimensions.Input = {"width": 600, "height": 600}
    result3 = next(iter(calc(iter([sample3]))))

    # Verify different resolutions were chosen.
    area1 = result1["target_width"] * result1["target_height"]
    area2 = result2["target_width"] * result2["target_height"]
    area3 = result3["target_width"] * result3["target_height"]

    assert area1 < area2 < area3


def test_calc_resize_dimensions_aspect_bucketing():
    """Test CalcResizeDimensions correctly buckets by aspect ratio."""
    config = CalcResizeDimensions.Config(
        square_resolutions=[240],
        aspects=[2 / 1, 1 / 1, 1 / 2],  # Wide, square, tall.
    )
    calc = CalcResizeDimensions(config)

    # Wide image.
    wide: CalcResizeDimensions.Input = {"width": 2000, "height": 1000}
    result_wide = next(iter(calc(iter([wide]))))
    aspect_wide = result_wide["target_width"] / result_wide["target_height"]

    # Square image.
    square: CalcResizeDimensions.Input = {"width": 1000, "height": 1000}
    result_square = next(iter(calc(iter([square]))))
    aspect_square = result_square["target_width"] / result_square["target_height"]

    # Tall image.
    tall: CalcResizeDimensions.Input = {"width": 1000, "height": 2000}
    result_tall = next(iter(calc(iter([tall]))))
    aspect_tall = result_tall["target_width"] / result_tall["target_height"]

    # Verify aspect ratios are correctly bucketed.
    assert aspect_wide > 1.5  # Close to 2:1.
    assert abs(aspect_square - 1.0) < 0.1  # Close to 1:1.
    assert aspect_tall < 0.7  # Close to 1:2.


def test_subsample_frames_via_bisection_cfhw_axis():
    """Subsampling must operate on the F axis of decoder (C, F, H, W) tensors."""
    config = SubsampleFramesViaBisection.Config(max_num_keyframes=3)
    processor = config.make()

    # Decoder layout (C=3, F=8, H=2, W=2): F is at axis -3, not -4 (channels).
    x = torch.arange(3 * 8 * 2 * 2).reshape(3, 8, 2, 2).float()
    sample: SubsampleFramesViaBisection.Input = {"media_tensor": x}
    result = next(iter(processor(iter([sample]))))

    # Must detect 8 frames and subsample them to 3, preserving 3 channels.
    assert "keyframes" in result
    assert "media_tensor" in result
    assert len(result["keyframes"]) == 3
    assert tuple(result["media_tensor"].shape) == (3, 3, 2, 2)


def test_subsample_frames_via_bisection_single_frame_image():
    """A single-frame (C, 1, H, W) image must pass through with keyframes [0]."""
    config = SubsampleFramesViaBisection.Config(max_num_keyframes=5)
    processor = config.make()

    x = torch.zeros(3, 1, 4, 4)
    sample: SubsampleFramesViaBisection.Input = {"media_tensor": x}
    result = next(iter(processor(iter([sample]))))

    assert "keyframes" in result
    assert "media_tensor" in result
    assert result["keyframes"] == [0]
    assert tuple(result["media_tensor"].shape) == (3, 1, 4, 4)


def test_subsample_frames_via_bisection_batched_cfhw():
    """Batched (B, C, F, H, W) must subsample the F axis, preserving B and C."""
    config = SubsampleFramesViaBisection.Config(max_num_keyframes=2)
    processor = config.make()

    # (B=2, C=3, F=6, H=2, W=2).
    x = torch.arange(2 * 3 * 6 * 2 * 2).reshape(2, 3, 6, 2, 2).float()
    sample: SubsampleFramesViaBisection.Input = {"media_tensor": x}
    result = next(iter(processor(iter([sample]))))

    assert "keyframes" in result
    assert "media_tensor" in result
    assert len(result["keyframes"]) == 2
    assert tuple(result["media_tensor"].shape) == (2, 3, 2, 2, 2)


def test_calc_resize_rejects_empty_buckets_and_zero_compression():
    """Degenerate bucket sets are rejected at construction, not per sample.

    An empty ``square_resolutions`` built fine and then raised
    ``min() iterable argument is empty`` on the first sample -- far from the
    config that caused it -- and a zero compression stride divided by zero
    during bucket construction.
    """
    with pytest.raises(ValueError, match="square_resolutions"):
        _ = CalcResizeDimensions.Config(square_resolutions=[]).make()
    with pytest.raises(ValueError, match="aspects"):
        _ = CalcResizeDimensions.Config(aspects=[]).make()
    with pytest.raises(ValueError, match="square_resolutions"):
        _ = CalcResizeDimensions.Config(square_resolutions=[0]).make()
    with pytest.raises(ValueError, match="aspects"):
        _ = CalcResizeDimensions.Config(aspects=[0]).make()
    with pytest.raises(ValueError, match="compression"):
        _ = CalcResizeDimensions.Config(compression=(1, 0, 16)).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
