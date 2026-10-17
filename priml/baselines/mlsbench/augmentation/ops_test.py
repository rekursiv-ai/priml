"""Tests for the Pillow operations port, against Pillow itself on small images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import math

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import functional

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.augmentation import ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

HEIGHT: Final = 5
WIDTH: Final = 7


def image(seed: int = 0, *, low: int = 0, high: int = 256) -> Tensor:
    """Return a ``(3, HEIGHT, WIDTH)`` uint8 image of random pixels."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, (3, HEIGHT, WIDTH), generator=generator).to(
        torch.uint8,
    )


def pil(pixels: Tensor) -> Image.Image:
    return Image.fromarray(pixels.permute(1, 2, 0).contiguous().numpy())


def pixels(picture: Image.Image) -> Tensor:
    array = np.asarray(picture)
    if array.ndim == 2:
        return torch.from_numpy(array.copy())
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def pillow(function: Callable[..., object]) -> Callable[..., Image.Image]:
    """Type a torchvision function, annotated for tensors, as called on an image."""
    return cast("Callable[..., Image.Image]", function)


def assert_pillow(
    ours: Callable[[Tensor], Tensor],
    theirs: Callable[[Image.Image], Image.Image],
    source: Tensor,
) -> None:
    assert torch.equal(ours(source), pixels(theirs(pil(source))))


@pytest.mark.parametrize(
    "shear",
    [[30.0, 0.0], [-12.0, 0.0], [0.0, 25.0], [0.0, -40.0]],
    ids=["x", "-x", "y", "-y"],
)
def test_a_shear_matches_pillow_in_fixed_point(shear: list[float]) -> None:
    matrix = ops.inverse_affine([0, 0], translate=[0, 0], shear=shear)
    assert_pillow(
        lambda x: ops.affine(x, matrix),
        lambda p: pillow(functional.affine)(
            p,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=shear,
            center=[0, 0],
        ),
        image(),
    )


@pytest.mark.parametrize("shift", [[3, 0], [-2, 0], [0, 4], [0, -9]])
def test_a_translation_matches_pillow_through_its_scaling_path(
    shift: list[int],
) -> None:
    center = [WIDTH * 0.5, HEIGHT * 0.5]
    matrix = ops.inverse_affine(center, translate=shift, shear=[0.0, 0.0])
    assert_pillow(
        lambda x: ops.affine(x, matrix),
        lambda p: pillow(functional.affine)(
            p,
            angle=0.0,
            translate=shift,
            scale=1.0,
            shear=[0.0, 0.0],
        ),
        image(),
    )


@pytest.mark.parametrize("angle", [0.0, 17.0, -100.0, 180.0, 90.0, 270.0, 360.0])
def test_a_rotation_matches_pillow(angle: float) -> None:
    assert_pillow(
        lambda x: ops.rotate(x, angle),
        lambda p: p.rotate(angle),
        image(),
    )


@pytest.mark.parametrize("angle", [90.0, 270.0])
def test_a_quarter_turn_of_a_square_image_takes_pillows_transpose(angle: float) -> None:
    square = image()[:, :HEIGHT, :HEIGHT]
    assert_pillow(lambda x: ops.rotate(x, angle), lambda p: p.rotate(angle), square)


def test_the_fixed_point_path_refuses_a_map_beyond_its_range() -> None:
    with pytest.raises(AssertionError, match=r"16\.16"):
        ops.affine(image(), [1.0, 0.5, 40_000.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("factor", [0.0, 1.0, 0.3, 1.7, -0.4])
def test_blend_matches_pillow_inside_and_outside_unit_range(factor: float) -> None:
    degenerate = image(1)
    assert torch.equal(
        ops.blend(degenerate, image(), factor),
        pixels(Image.blend(pil(degenerate), pil(image()), factor)),
    )


def test_grayscale_matches_pillows_conversion() -> None:
    assert_pillow(
        lambda x: ops.grayscale(x)[None].expand(3, -1, -1),
        lambda p: p.convert("L").convert("RGB"),
        image(),
    )


def test_smooth_matches_pillow_and_keeps_the_border() -> None:
    assert_pillow(ops.smooth, lambda p: p.filter(ImageFilter.SMOOTH), image())


@pytest.mark.parametrize("bits", [0, 3, 8])
def test_posterize_matches_pillow(bits: int) -> None:
    assert_pillow(
        lambda x: ops.posterize(x, bits),
        lambda p: ImageOps.posterize(p, bits),
        image(),
    )


@pytest.mark.parametrize("threshold", [0.0, 127.5, 255.0])
def test_solarize_matches_pillow(threshold: float) -> None:
    assert_pillow(
        lambda x: ops.solarize(x, threshold),
        lambda p: pillow(functional.solarize)(p, threshold),
        image(),
    )


def test_autocontrast_matches_pillow_and_keeps_a_flat_channel() -> None:
    source = image(low=40, high=90)
    source[1] = 7
    assert_pillow(ops.autocontrast, ImageOps.autocontrast, source)


def test_equalize_matches_pillow_and_keeps_a_flat_or_sparse_channel() -> None:
    source = image()
    source[1] = 7
    source[2] = 0
    source[2, 0, 0] = 9
    assert_pillow(ops.equalize, ImageOps.equalize, source)


def test_equalize_remaps_a_channel_of_more_than_255_pixels_as_pillow_does() -> None:
    # Pillow's step is the pixel count over 255, so a smaller image is kept as is.
    generator = torch.Generator().manual_seed(0)
    source = torch.randint(0, 256, (3, 16, 17), generator=generator).to(torch.uint8)
    assert not torch.equal(ops.equalize(source), source)
    assert_pillow(ops.equalize, ImageOps.equalize, source)


def test_enhancers_are_blends_from_pillows_degenerate_images() -> None:
    source = image()
    assert_pillow(
        lambda x: ops.blend(ops.smooth(x), x, 1.6),
        lambda p: ImageEnhance.Sharpness(p).enhance(1.6),
        source,
    )


def test_inverse_affine_fixes_the_center_of_a_shear() -> None:
    matrix = ops.inverse_affine([2.0, 3.0], translate=[0, 0], shear=[45.0, 0.0])
    x, y = 2.0, 3.0
    assert matrix[0] * x + matrix[1] * y + matrix[2] == pytest.approx(x)
    assert matrix[1] == pytest.approx(math.tan(math.radians(45.0)))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
