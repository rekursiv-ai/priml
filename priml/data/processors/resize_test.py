"""Tests for the Interpolate processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import pytest
import torch

from priml.data.processors.resize import Interpolate


if TYPE_CHECKING:
    from priml.data.processors.utils import InterpolationMode


def _run(config: Interpolate.Config, x: torch.Tensor, size: int) -> torch.Tensor:
    sample: Interpolate.Input = {
        "media_tensor": x,
        "target_height": size,
        "target_width": size,
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
    assert _run(config, torch.randn(3, 1, 16, 16), size).shape[-2:] == (size, size)


def test_interpolate_resizes_to_the_requested_size() -> None:
    out = _run(Interpolate.Config(), torch.randn(3, 2, 16, 16), 8)
    assert out.shape == (3, 2, 8, 8)


def test_a_uint8_sample_survives_the_round_trip() -> None:
    """uint8 in, uint8 out: the processor rescales to float and back.

    The conversion runs through ``rgb2float``/``float2rgb``, so a resize that
    changes nothing must return the bytes it was given.
    """
    x = torch.randint(0, 256, (3, 1, 8, 8), dtype=torch.uint8)
    out = _run(Interpolate.Config(), x, 8)
    assert out.dtype == torch.uint8
    assert torch.equal(out, x)


@pytest.mark.parametrize("mode", ["area", "hybrid"])
def test_a_uint8_cpu_downsample_is_opencv_area_on_every_frame(
    mode: InterpolationMode,
) -> None:
    """uint8 area downsampling stays in uint8, pixel-for-pixel OpenCV's.

    The torch path widens to float16 and rounds back per sample, which cost
    more than the JPEG decode feeding it.
    """
    x = torch.randint(0, 256, (3, 2, 37, 53), dtype=torch.uint8)
    out = _run(Interpolate.Config(mode=mode), x, 16)
    assert out.dtype == torch.uint8
    for frame in range(2):
        hwc = x[:, frame].permute(1, 2, 0).numpy()
        want = cv2.resize(hwc, (16, 16), interpolation=cv2.INTER_AREA)
        assert torch.equal(out[:, frame], torch.from_numpy(want).permute(2, 0, 1))


def test_a_uint8_upsample_keeps_the_torch_path() -> None:
    """``hybrid`` upsamples bicubically; OpenCV's area kernel is not that."""
    x = torch.randint(0, 256, (3, 1, 8, 8), dtype=torch.uint8)
    out = _run(Interpolate.Config(), x, 16)
    assert out.shape == (3, 1, 16, 16)
    assert out.dtype == torch.uint8


def test_a_sample_missing_its_target_passes_through() -> None:
    original = torch.randn(3, 1, 8, 8)
    sample: Interpolate.Input = {"media_tensor": original}
    out = list(Interpolate.Config().make()(iter([sample])))
    result = out[0].get("media_tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, original)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
