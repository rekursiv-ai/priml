"""Tests for the Pillow operations port, against Pillow itself on small images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import math

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import functional

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

HEIGHT: Final = 5
WIDTH: Final = 7


GRAY_CASES: Final[
    list[tuple[Callable[[Tensor], Tensor], Callable[[Image.Image], Image.Image]]]
] = [
    (
        lambda x: ops.affine(
            x,
            ops.inverse_affine([0, 0], translate=[0, 0], shear=[30.0, 0.0]),
        ),
        lambda p: pillow(functional.affine)(
            p,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[30.0, 0.0],
            center=[0, 0],
        ),
    ),
    (
        lambda x: ops.affine(
            x,
            ops.inverse_affine([0, 0], translate=[0, 0], shear=[0.0, -40.0]),
        ),
        lambda p: pillow(functional.affine)(
            p,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, -40.0],
            center=[0, 0],
        ),
    ),
    (
        lambda x: ops.affine(
            x,
            ops.inverse_affine(
                [WIDTH * 0.5, HEIGHT * 0.5],
                translate=[3, -2],
                shear=[0.0, 0.0],
            ),
        ),
        lambda p: pillow(functional.affine)(
            p,
            angle=0.0,
            translate=[3, -2],
            scale=1.0,
            shear=[0.0, 0.0],
        ),
    ),
    (lambda x: ops.rotate(x, 17.0), lambda p: p.rotate(17.0)),
    (lambda x: ops.rotate(x, 180.0), lambda p: p.rotate(180.0)),
    (
        lambda x: ops.blend(torch.zeros_like(x), x, 1.6),
        lambda p: ImageEnhance.Brightness(p).enhance(1.6),
    ),
    (
        lambda x: ops.blend(ops.grayscale(x).expand_as(x), x, 0.4),
        lambda p: ImageEnhance.Color(p).enhance(0.4),
    ),
    (
        lambda x: ops.blend(ops.smooth(x), x, 1.6),
        lambda p: ImageEnhance.Sharpness(p).enhance(1.6),
    ),
    (ops.smooth, lambda p: p.filter(ImageFilter.SMOOTH)),
    (lambda x: ops.grayscale(x)[None], lambda p: p.convert("L")),
    (lambda x: ops.posterize(x, 3), lambda p: ImageOps.posterize(p, 3)),
    (
        lambda x: ops.solarize(x, 127.5),
        lambda p: pillow(functional.solarize)(p, 127.5),
    ),
    (ops.autocontrast, ImageOps.autocontrast),
    (ops.equalize, ImageOps.equalize),
]
"""Each Pillow operation the policies reach, and the port's, on an ``L`` image."""


def image(seed: int = 0, *, low: int = 0, high: int = 256) -> Tensor:
    """Return a ``(3, HEIGHT, WIDTH)`` uint8 image of random pixels."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, (3, HEIGHT, WIDTH), generator=generator).to(
        torch.uint8,
    )


def gray(seed: int = 0) -> Tensor:
    """Return a ``(1, HEIGHT, WIDTH)`` uint8 image of random pixels, Pillow's ``L``."""
    return image(seed)[:1].clone()


def pil(pixels: Tensor) -> Image.Image:
    """Return a ``(C, H, W)`` uint8 image as PIL holds it: RGB, or ``L`` for one channel."""
    return Image.fromarray(pixels.permute(1, 2, 0).squeeze(-1).contiguous().numpy())


def pixels(picture: Image.Image) -> Tensor:
    array = np.asarray(picture)
    if array.ndim == 2:
        return torch.from_numpy(array.copy())[None]
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
    with pytest.raises(ValueError, match=r"16\.16"):
        ops.affine(image(), [1.0, 0.5, 40_000.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("factor", [0.0, 1.0, 0.3, 0.7])
def test_blend_matches_pillow_inside_the_unit_range(factor: float) -> None:
    degenerate = image(1)
    assert torch.equal(
        ops.blend(degenerate, image(), factor),
        pixels(Image.blend(pil(degenerate), pil(image()), factor)),
    )


# Pillow extrapolates in float32 as ``in1 + alpha * (in2 - in1)``, truncated and
# clipped. Its x86-64 build, where MLS-Bench runs, rounds the product and the sum
# apart; its aarch64 build fuses them into one FMA, and the two truncate to
# different integers on the first three pixels at 1.7 and the 9th to 14th at -0.4.
# So the expected values are x86-64 Pillow 11.3's, not the host's.
@pytest.mark.parametrize(
    ("factor", "x86"),
    [
        (
            1.7,
            [
                [1, 2, 3, 13, 24, 38, 60, 73, 22, 32, 33, 43],
                [44, 53, 102, 168, 255, 0, 100, 255, 0, 19, 255, 0],
            ],
        ),
        (
            -0.4,
            [
                [22, 23, 24, 34, 45, 59, 81, 94, 1, 1, 2, 1],
                [2, 1, 50, 82, 0, 255, 100, 0, 255, 153, 0, 255],
            ],
        ),
    ],
)
def test_blend_extrapolates_as_x86_64_pillow(
    factor: float,
    x86: list[list[int]],
) -> None:
    def rgb(rows: list[list[int]]) -> Tensor:
        return torch.tensor(rows, dtype=torch.uint8).view(2, 4, 3).permute(2, 0, 1)

    # Two rows of four RGB pixels each, channels last, as ``Image.fromarray`` reads.
    degenerate = rgb(
        [
            [18, 19, 20, 30, 41, 55, 77, 90, 5, 7, 8, 9],
            [10, 11, 60, 99, 0, 255, 100, 37, 200, 128, 3, 250],
        ],
    )
    image = rgb(
        [
            [8, 9, 10, 20, 31, 45, 67, 80, 15, 22, 23, 29],
            [30, 36, 85, 140, 255, 0, 100, 200, 37, 64, 252, 4],
        ],
    )
    blended = ops.blend(degenerate, image, factor)
    assert blended.permute(1, 2, 0).reshape(2, 12).tolist() == x86


def test_grayscale_matches_pillows_conversion() -> None:
    assert_pillow(
        lambda x: ops.grayscale(x)[None].expand(3, -1, -1),
        lambda p: p.convert("L").convert("RGB"),
        image(),
    )


def test_smooth_matches_pillow_and_keeps_the_border() -> None:
    assert_pillow(ops.smooth, lambda p: p.filter(ImageFilter.SMOOTH), image())


@pytest.mark.parametrize("bits", [1, 3, 8])
def test_posterize_matches_pillow(bits: int) -> None:
    assert_pillow(
        lambda x: ops.posterize(x, bits),
        lambda p: ImageOps.posterize(p, bits),
        image(),
    )


def test_posterize_to_no_bits_is_black() -> None:
    # Pillow 13 refuses ``bits=0``; earlier releases masked every bit away.
    assert not ops.posterize(image(), 0).any()


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


@pytest.mark.parametrize(
    ("ours", "theirs"),
    GRAY_CASES,
    ids=[
        "shear-x",
        "shear-y",
        "translate",
        "rotate",
        "rotate-180",
        "brightness",
        "color",
        "sharpness",
        "smooth",
        "grayscale",
        "posterize",
        "solarize",
        "autocontrast",
        "equalize",
    ],
)
def test_every_operation_matches_pillow_on_a_grayscale_image(
    ours: Callable[[Tensor], Tensor],
    theirs: Callable[[Image.Image], Image.Image],
) -> None:
    source = gray()
    assert pil(source).mode == "L"
    assert_pillow(ours, theirs, source)


def test_a_quarter_turn_of_a_square_grayscale_image_matches_pillow() -> None:
    square = gray()[:, :HEIGHT, :HEIGHT]
    assert_pillow(lambda x: ops.rotate(x, 90.0), lambda p: p.rotate(90.0), square)


def test_equalize_remaps_a_large_grayscale_image_as_pillow_does() -> None:
    generator = torch.Generator().manual_seed(0)
    # One channel is Pillow's ``L`` mode, which ``ops.grayscale`` and ``pil`` key on.
    source = torch.randint(0, 256, (1, 16, 17), generator=generator).to(torch.uint8)
    assert not torch.equal(ops.equalize(source), source)
    assert_pillow(ops.equalize, ImageOps.equalize, source)


def test_inverse_affine_fixes_the_center_of_a_shear() -> None:
    matrix = ops.inverse_affine([2.0, 3.0], translate=[0, 0], shear=[45.0, 0.0])
    x, y = 2.0, 3.0
    assert matrix[0] * x + matrix[1] * y + matrix[2] == pytest.approx(x)
    assert matrix[1] == pytest.approx(math.tan(math.radians(45.0)))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
