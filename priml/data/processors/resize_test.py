"""Tests for the Interpolate processor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import cv2
import pytest
import torch

from priml.data.processors.resize import Interpolate
from priml.math.pixel import float2rgb, rgb2float


if TYPE_CHECKING:
    from priml.data.processors.utils import InterpolationMode


def _run(
    config: Interpolate.Config,
    x: torch.Tensor,
    size: int | tuple[int, int],
) -> torch.Tensor:
    target_height, target_width = (size, size) if isinstance(size, int) else size
    sample: Interpolate.Input = {
        "media_tensor": x,
        "target_height": target_height,
        "target_width": target_width,
    }
    out = list(config.make()(iter([sample])))
    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    return tensor


@pytest.mark.parametrize("mode", ["area", "hybrid", "nearest", "bilinear"])
@pytest.mark.parametrize("size", [8, 32])
def test_antialias_is_ignored_by_the_modes_torch_rejects_it_for(
    mode: InterpolationMode,
    size: int,
) -> None:
    """``antialias=True`` must never raise, whatever the mode resolves to.

    torch accepts the flag only for bilinear and bicubic. Forwarding it
    unfiltered made ``mode="area"`` fail outright, and ``"hybrid"`` -- which
    picks the mode per sample -- succeed when upsampling and raise when
    downsampling the identical config.
    """
    config = Interpolate.Config(mode=mode, antialias=True)
    assert _run(config, torch.randn(3, 2, 4, 5), size).shape[-2:] == (size, size)


@pytest.mark.parametrize(
    ("mode", "dtype", "size", "expected"),
    [
        ("area", torch.uint8, (2, 3), True),
        ("area", torch.uint8, (5, 6), True),
        ("area", torch.uint8, (5, 7), False),
        ("hybrid", torch.uint8, (2, 3), True),
        ("hybrid", torch.uint8, (5, 7), False),
        ("bilinear", torch.uint8, (2, 3), False),
        ("area", torch.float32, (2, 3), False),
    ],
)
def test_opencv_area_is_limited_to_uint8_cpu_downsamples(
    mode: InterpolationMode,
    dtype: torch.dtype,
    size: tuple[int, int],
    expected: bool,
) -> None:
    processor = Interpolate.Config(mode=mode).make()
    sample = torch.zeros((2, 3, 4, 5, 7), dtype=dtype)
    assert processor._uses_opencv_area(sample, *size) is expected


def test_interpolate_resizes_to_the_requested_size() -> None:
    out = _run(Interpolate.Config(), torch.randn(3, 2, 4, 5), 8)
    assert out.shape == (3, 2, 8, 8)


def test_a_uint8_sample_survives_the_round_trip() -> None:
    """uint8 in, uint8 out: the processor rescales to float and back.

    The conversion runs through ``rgb2float``/``float2rgb``, so a resize that
    changes nothing must return the bytes it was given.
    """
    x = torch.randint(0, 256, (3, 2, 4, 5), dtype=torch.uint8)
    with (
        patch(
            "priml.data.processors.resize.rgb2float",
            wraps=rgb2float,
        ) as to_float,
        patch(
            "priml.data.processors.resize.float2rgb",
            wraps=float2rgb,
        ) as to_uint8,
    ):
        out = _run(Interpolate.Config(), x, (4, 5))

    assert out.dtype == torch.uint8
    assert torch.equal(out, x)
    assert to_float.call_args is not None
    assert to_float.call_args.kwargs == {"inplace": True}
    assert to_uint8.call_args is not None
    assert to_uint8.call_args.kwargs == {"inplace": True}


@pytest.mark.parametrize("mode", ["area", "hybrid"])
def test_a_uint8_cpu_downsample_is_opencv_area_on_every_frame(
    mode: InterpolationMode,
) -> None:
    """uint8 area downsampling stays in uint8, pixel-for-pixel OpenCV's.

    The torch path widens to float16 and rounds back per sample, which cost
    more than the JPEG decode feeding it.
    """
    x = torch.randint(0, 256, (3, 2, 5, 7), dtype=torch.uint8)
    out = _run(Interpolate.Config(mode=mode), x, 4)
    assert out.dtype == torch.uint8
    for frame in range(2):
        hwc = x[:, frame].permute(1, 2, 0).numpy()
        want = cv2.resize(hwc, (4, 4), interpolation=cv2.INTER_AREA)
        assert torch.equal(out[:, frame], torch.from_numpy(want).permute(2, 0, 1))


def test_a_batched_uint8_area_downsample_restores_every_axis() -> None:
    x = torch.randint(0, 256, (2, 3, 4, 5, 7), dtype=torch.uint8)
    out = _run(Interpolate.Config(mode="area"), x, (2, 3))

    assert out.shape == (2, 3, 4, 2, 3)
    for batch in range(2):
        for frame in range(4):
            hwc = x[batch, :, frame].permute(1, 2, 0).numpy()
            want = cv2.resize(hwc, (3, 2), interpolation=cv2.INTER_AREA)
            assert torch.equal(
                out[batch, :, frame],
                torch.from_numpy(want).permute(2, 0, 1),
            )


def test_a_uint8_upsample_keeps_the_torch_path() -> None:
    """``hybrid`` upsamples bicubically; OpenCV's area kernel is not that."""
    x = torch.randint(0, 256, (3, 2, 4, 5), dtype=torch.uint8)
    out = _run(Interpolate.Config(), x, 4)
    assert out.shape == (3, 2, 4, 4)
    assert out.dtype == torch.uint8


def test_missing_each_required_field_passes_through() -> None:
    processor = Interpolate.Config().make()
    image = torch.randn(3, 2, 4, 5)
    samples: list[Interpolate.Input] = [
        {"target_height": 8, "target_width": 7},
        {"media_tensor": image, "target_width": 7},
        {"media_tensor": image, "target_height": 8},
    ]

    results = list(processor(iter(samples)))

    assert len(results) == 3
    assert all(
        result is sample for result, sample in zip(results, samples, strict=True)
    )
    assert all(result.get("media_tensor") is image for result in results[1:])


def test_interpolate_continues_after_missing_and_opencv_samples() -> None:
    processor = Interpolate.Config(mode="area").make()
    skipped: Interpolate.Input = {"media_tensor": torch.zeros(3, 2, 4, 5)}
    area_x = torch.randint(0, 256, (3, 2, 5, 7), dtype=torch.uint8)
    area_sample: Interpolate.Input = {
        "media_tensor": area_x,
        "target_height": 2,
        "target_width": 3,
    }
    torch_x = torch.randn(3, 2, 5, 7)
    torch_sample: Interpolate.Input = {
        "media_tensor": torch_x,
        "target_height": 4,
        "target_width": 3,
    }

    results = list(processor(iter([skipped, area_sample, torch_sample])))

    assert len(results) == 3
    assert results[0] is skipped
    assert results[1] is area_sample
    assert results[2] is torch_sample
    assert results[0].get("media_tensor") is skipped.get("media_tensor")
    area_tensor = results[1].get("media_tensor")
    torch_tensor = results[2].get("media_tensor")
    assert isinstance(area_tensor, torch.Tensor)
    assert isinstance(torch_tensor, torch.Tensor)
    assert area_tensor.shape == (3, 2, 2, 3)
    assert torch_tensor.shape == (3, 2, 4, 3)


def test_bilinear_resize_honors_alignment_and_antialias() -> None:
    x = torch.linspace(-1, 1, 2 * 5 * 6 * 7 * 8).reshape(2, 5, 6, 7, 8)
    sample: Interpolate.Input = {
        "media_tensor": x,
        "target_height": 4,
        "target_width": 3,
    }
    config = Interpolate.Config(
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )

    result = next(iter(config.make()(iter([sample]))))

    expected = (
        torch.nn.functional.interpolate(
            x.permute(0, 2, 1, 3, 4).reshape(-1, 5, 7, 8),
            size=(4, 3),
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        .reshape(2, 6, 5, 4, 3)
        .permute(0, 2, 1, 3, 4)
    )
    actual = result.get("media_tensor")
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)


def test_a_sample_missing_its_target_passes_through() -> None:
    original = torch.randn(3, 2, 4, 5)
    sample: Interpolate.Input = {"media_tensor": original}
    out = list(Interpolate.Config().make()(iter([sample])))
    result = out[0].get("media_tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, original)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
