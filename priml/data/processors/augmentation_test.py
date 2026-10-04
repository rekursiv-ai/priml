"""Tests for the image augmentation processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import math
import random

from torch import Tensor

import pytest
import torch
import torchvision.transforms.v2 as tv_transforms
import torchvision.transforms.v2.functional as tv_functional

from priml.data.processors.augmentation import (
    _RAND_AUGMENT_OPS,
    ColorJitter,
    GetCenterCropBoxFromDimensions,
    GetRandomResizedCropBoxFromDimensions,
    MixupCutmix,
    Normalize,
    RandAugment,
    RandomErasing,
    RandomHorizontalFlip,
    _one_hot_with_smoothing,
    _posterize,
    _rotate,
    _shear_x,
    _shear_y,
    _solarize,
    _translate_x,
    _translate_y,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def test_random_resized_crop_box_stays_inside_the_image() -> None:
    processor = GetRandomResizedCropBoxFromDimensions.Config(size=(32, 48)).make()
    random.seed(0)

    samples: list[GetRandomResizedCropBoxFromDimensions.Input] = [
        {"height": 100, "width": 200} for _ in range(20)
    ]
    samples.append({"height": 100})
    results = list(processor(iter(samples)))

    assert "crop" not in results[-1]
    for result in results[:-1]:
        crop = result.get("crop")
        assert crop is not None
        y, x, h, w = crop
        assert 0 < h <= 100
        assert 0 < w <= 200
        assert 0 <= y <= 100 - h
        assert 0 <= x <= 200 - w
        assert (result.get("target_height"), result.get("target_width")) == (32, 48)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (100, 10, (43, 0, 13, 10)),  # Too tall: full width, ratio-limited height.
        (10, 100, (0, 43, 10, 13)),  # Too wide: full height, ratio-limited width.
        (100, 100, (0, 0, 100, 100)),  # In range: whole image.
        (80, 60, (0, 0, 80, 60)),  # Exactly the narrowest allowed ratio.
        (60, 80, (0, 0, 60, 80)),  # Exactly the widest allowed ratio.
    ],
    ids=["tall", "wide", "square", "min-ratio", "max-ratio"],
)
def test_random_resized_crop_falls_back_to_a_center_crop(
    monkeypatch: pytest.MonkeyPatch,
    height: int,
    width: int,
    expected: tuple[int, int, int, int],
) -> None:
    processor = GetRandomResizedCropBoxFromDimensions.Config().make()
    draws: list[tuple[float, float]] = []

    def oversized(low: float, high: float) -> float:
        # Every attempt asks for more area than the image has, so all ten fail.
        draws.append((low, high))
        return high * 4.0

    monkeypatch.setattr(random, "uniform", oversized)

    sample: GetRandomResizedCropBoxFromDimensions.Input = {
        "height": height,
        "width": width,
    }
    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == expected
    assert (result.get("target_height"), result.get("target_width")) == (224, 224)
    assert len(draws) == 20


def test_random_resized_crop_uses_drawn_scale_ratio_and_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uniform_ranges: list[tuple[float, float]] = []
    integer_ranges: list[tuple[int, int]] = []
    uniform_draws = iter([0.5, 1.0])
    integer_draws = iter([5, 3])

    def uniform(low: float, high: float) -> float:
        uniform_ranges.append((low, high))
        return next(uniform_draws)

    def randint(low: int, high: int) -> int:
        integer_ranges.append((low, high))
        return next(integer_draws)

    monkeypatch.setattr(random, "uniform", uniform)
    monkeypatch.setattr(random, "randint", randint)
    sample: GetRandomResizedCropBoxFromDimensions.Input = {
        "height": 80,
        "width": 120,
    }
    processor = GetRandomResizedCropBoxFromDimensions.Config(
        size=(12, 18),
        scale=(0.25, 0.75),
        ratio=(0.5, 2.0),
    ).make()

    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == (3, 5, 42, 114)
    assert (result.get("target_height"), result.get("target_width")) == (12, 18)
    assert uniform_ranges == [(0.25, 0.75), (math.log(0.5), math.log(2.0))]
    assert integer_ranges == [(0, 6), (0, 38)]


@pytest.mark.parametrize(
    "scale",
    [(1.0, 1.0), (0.0, 0.0)],
    ids=["exact-image-boundary", "zero-area-falls-back"],
)
def test_random_resized_crop_handles_exact_size_and_zero_area(
    monkeypatch: pytest.MonkeyPatch,
    scale: tuple[float, float],
) -> None:
    draws = iter([draw for _ in range(10) for draw in (scale[0], 0.0)])
    integer_ranges: list[tuple[int, int]] = []

    def draw_uniform(low: float, high: float) -> float:
        del low, high
        return next(draws)

    def draw_integer(low: int, high: int) -> int:
        integer_ranges.append((low, high))
        return low

    monkeypatch.setattr(random, "uniform", draw_uniform)
    monkeypatch.setattr(random, "randint", draw_integer)
    processor = GetRandomResizedCropBoxFromDimensions.Config(
        size=(12, 18),
        scale=scale,
        ratio=(1.0, 1.0),
    ).make()
    sample: GetRandomResizedCropBoxFromDimensions.Input = {
        "height": 80,
        "width": 80,
    }

    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == (0, 0, 80, 80)
    assert (result.get("target_height"), result.get("target_width")) == (12, 18)
    expected_ranges = [(0, 0), (0, 0)] if scale == (1.0, 1.0) else []
    assert integer_ranges == expected_ranges


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (4.0, (2, 0, 2, 8)),
        (0.25, (0, 3, 6, 2)),
    ],
    ids=["rounded-height-zero", "rounded-width-zero"],
)
def test_random_resized_crop_rejects_a_zero_sized_candidate(
    monkeypatch: pytest.MonkeyPatch,
    ratio: float,
    expected: tuple[int, int, int, int],
) -> None:
    draws = iter([draw for _ in range(10) for draw in (0.01, math.log(ratio))])
    integer_ranges: list[tuple[int, int]] = []

    def draw_uniform(low: float, high: float) -> float:
        del low, high
        return next(draws)

    def draw_integer(low: int, high: int) -> int:
        integer_ranges.append((low, high))
        return low

    monkeypatch.setattr(random, "uniform", draw_uniform)
    monkeypatch.setattr(random, "randint", draw_integer)
    processor = GetRandomResizedCropBoxFromDimensions.Config(
        scale=(0.01, 0.01),
        ratio=(ratio, ratio),
    ).make()

    result = next(
        iter(
            processor(
                iter(
                    [
                        cast(
                            GetRandomResizedCropBoxFromDimensions.Input,
                            {"height": 6, "width": 8},
                        ),
                    ],
                ),
            ),
        ),
    )

    assert result.get("crop") == expected
    assert integer_ranges == []


def test_random_resized_crop_accepts_a_one_pixel_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integer_ranges: list[tuple[int, int]] = []

    def draw_integer(low: int, high: int) -> int:
        integer_ranges.append((low, high))
        return low

    def draw_uniform(low: float, high: float) -> float:
        del high
        return low

    monkeypatch.setattr(random, "uniform", draw_uniform)
    monkeypatch.setattr(random, "randint", draw_integer)
    processor = GetRandomResizedCropBoxFromDimensions.Config(
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
    ).make()

    result = next(
        iter(
            processor(
                iter(
                    [
                        cast(
                            GetRandomResizedCropBoxFromDimensions.Input,
                            {"height": 1, "width": 1},
                        ),
                    ],
                ),
            ),
        ),
    )

    assert result.get("crop") == (0, 0, 1, 1)
    assert integer_ranges == [(0, 0), (0, 0)]


def test_center_crop_box_takes_the_centered_square() -> None:
    processor = GetCenterCropBoxFromDimensions.Config(size=(16, 16)).make()

    samples: list[GetCenterCropBoxFromDimensions.Input] = [
        {"height": 100, "width": 60},
        {"height": 30, "width": 90},
        {"width": 90},
    ]
    results = list(processor(iter(samples)))

    assert results[0].get("crop") == (20, 0, 60, 60)
    assert results[1].get("crop") == (0, 30, 30, 30)
    assert (results[0].get("target_height"), results[0].get("target_width")) == (16, 16)
    assert "crop" not in results[2]


def test_center_crop_uses_configured_fraction_of_shorter_side() -> None:
    processor = GetCenterCropBoxFromDimensions.Config(
        size=(12, 18),
        ratio=0.5,
    ).make()
    sample: GetCenterCropBoxFromDimensions.Input = {"height": 81, "width": 121}

    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == (20, 40, 40, 40)
    assert (result.get("target_height"), result.get("target_width")) == (12, 18)


def test_crop_processors_continue_after_samples_missing_dimensions() -> None:
    random_crop = GetRandomResizedCropBoxFromDimensions.Config().make()
    center_crop = GetCenterCropBoxFromDimensions.Config().make()
    random_results = list(
        random_crop(
            iter(
                [
                    cast(GetRandomResizedCropBoxFromDimensions.Input, {"height": 50}),
                    cast(
                        GetRandomResizedCropBoxFromDimensions.Input,
                        {"height": 80, "width": 120},
                    ),
                ],
            ),
        ),
    )
    center_results = list(
        center_crop(
            iter(
                [
                    cast(GetCenterCropBoxFromDimensions.Input, {"width": 50}),
                    cast(
                        GetCenterCropBoxFromDimensions.Input,
                        {"height": 80, "width": 120},
                    ),
                ],
            ),
        ),
    )

    assert "crop" not in random_results[0]
    assert random_results[1].get("crop") is not None
    assert "crop" not in center_results[0]
    assert center_results[1].get("crop") == (0, 20, 80, 80)


def test_random_horizontal_flip_mirrors_the_last_axis_with_probability_p() -> None:
    x = torch.arange(24.0).reshape(2, 3, 4)
    always: list[RandomHorizontalFlip.Input] = [{"media_tensor": x}, {}]
    never: list[RandomHorizontalFlip.Input] = [{"media_tensor": x}]

    flipped = list(RandomHorizontalFlip.Config(p=1.0).make()(iter(always)))
    kept = list(RandomHorizontalFlip.Config(p=0.0).make()(iter(never)))

    assert torch.equal(_media(flipped[0]), torch.flip(x, dims=[-1]))
    assert "media_tensor" not in flipped[1]
    assert kept[0].get("media_tensor") is x


def test_random_horizontal_flip_keeps_probability_boundary_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([0.5, 0.4])

    def rand(shape: tuple[int, ...]) -> Tensor:
        del shape
        return torch.tensor([next(values)])

    monkeypatch.setattr(torch, "rand", rand)
    image = torch.arange(24.0).reshape(2, 3, 4)
    processor = RandomHorizontalFlip.Config(p=0.5).make()
    samples: list[RandomHorizontalFlip.Input] = [
        {"media_tensor": image},
        {},
        {"media_tensor": image},
    ]

    results = list(processor(iter(samples)))

    assert _media(results[0]) is image
    assert results[1] == {}
    torch.testing.assert_close(_media(results[2]), torch.flip(image, dims=[-1]))


def test_color_jitter_forwards_config_and_continues_across_missing_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, float]] = []
    transformed: list[Tensor] = []

    def make_transform(
        *,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> Callable[[Tensor], Tensor]:
        constructor_calls.append(
            {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
            },
        )

        def transform(image: Tensor) -> Tensor:
            transformed.append(image)
            return image + 1

        return transform

    monkeypatch.setattr(tv_transforms, "ColorJitter", make_transform)
    image = torch.zeros(2, 3, 4, dtype=torch.uint8)
    samples: list[ColorJitter.Input] = [
        {"media_tensor": image},
        {},
        {"media_tensor": image},
    ]
    processor = ColorJitter.Config(
        brightness=0.1,
        contrast=0.2,
        saturation=0.3,
        hue=0.4,
    ).make()

    results = list(processor(iter(samples)))

    assert constructor_calls == [
        {"brightness": 0.1, "contrast": 0.2, "saturation": 0.3, "hue": 0.4},
    ]
    assert transformed == [image, image]
    assert torch.equal(_media(results[0]), image + 1)
    assert results[1] == {}
    assert torch.equal(_media(results[2]), image + 1)


def test_color_jitter_and_random_erasing_transform_only_present_media() -> None:
    x = torch.randint(0, 256, (3, 8, 9), dtype=torch.uint8)
    jitter = ColorJitter.Config(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    erase = RandomErasing.Config(p=1.0, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=0.0)
    to_jitter: list[ColorJitter.Input] = [{"media_tensor": x}, {}]
    to_erase: list[RandomErasing.Input] = [{"media_tensor": torch.ones(3, 8, 9)}, {}]

    jittered = list(jitter.make()(iter(to_jitter)))
    erased = list(erase.make()(iter(to_erase)))

    # Zero-strength jitter is the identity, and an absent field passes through.
    assert torch.equal(_media(jittered[0]), x)
    assert "media_tensor" not in jittered[1]
    # p=1 always erases a square region: torchvision rounds sqrt(32) to a
    # 6x6 box, so 36 pixels per channel are now the fill value.
    assert int((_media(erased[0]) == 0).sum()) == 3 * 36
    assert "media_tensor" not in erased[1]


def test_rand_augment_applies_num_ops_and_every_op_preserves_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randint(0, 256, (3, 8, 9), dtype=torch.uint8)
    applied: list[float] = []

    def spy(img: Tensor, magnitude: float) -> Tensor:
        applied.append(magnitude)
        return torch.full_like(img, len(applied))

    def choose(ops: Sequence[object]) -> object:
        assert ops is _RAND_AUGMENT_OPS
        return spy

    monkeypatch.setattr(random, "choice", choose)
    fixed = RandAugment.Config(num_ops=3, magnitude=9, num_magnitude_bins=30)
    drawn = RandAugment.Config(num_ops=1, magnitude=-1)
    zero = RandAugment.Config(num_ops=2, magnitude=0, num_magnitude_bins=30)
    monkeypatch.setattr(random, "random", lambda: 0.25)
    samples: list[RandAugment.Input] = [{"media_tensor": x}, {}, {"media_tensor": x}]

    results = list(fixed.make()(iter(samples)))
    drawn_result = next(
        iter(drawn.make()(iter([cast(RandAugment.Input, {"media_tensor": x})]))),
    )
    zero_result = next(
        iter(zero.make()(iter([cast(RandAugment.Input, {"media_tensor": x})]))),
    )

    assert applied == [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.0, 0.0]
    assert torch.equal(_media(results[0]), torch.full_like(x, 3))
    assert results[1] == {}
    assert torch.equal(_media(results[2]), torch.full_like(x, 6))
    assert torch.equal(_media(drawn_result), torch.full_like(x, 7))
    assert torch.equal(_media(zero_result), torch.full_like(x, 9))
    for op in _RAND_AUGMENT_OPS:
        out = op(x, 0.5)
        assert out.shape == x.shape, op.__name__
        assert out.dtype == torch.uint8, op.__name__


def test_random_erasing_forwards_config_and_processes_each_present_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, object]] = []
    transformed: list[Tensor] = []

    def make_transform(
        *,
        p: float,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        value: float,
    ) -> Callable[[Tensor], Tensor]:
        constructor_calls.append(
            {"p": p, "scale": scale, "ratio": ratio, "value": value},
        )

        def transform(image: Tensor) -> Tensor:
            transformed.append(image)
            return image + 1

        return transform

    monkeypatch.setattr(tv_transforms, "RandomErasing", make_transform)
    image = torch.zeros(2, 3, 4)
    processor = RandomErasing.Config(
        p=0.6,
        scale=(0.2, 0.4),
        ratio=(0.5, 1.5),
        value=0.7,
    ).make()
    samples: list[RandomErasing.Input] = [
        {"media_tensor": image},
        {},
        {"media_tensor": image},
    ]

    results = list(processor(iter(samples)))

    assert constructor_calls == [
        {"p": 0.6, "scale": (0.2, 0.4), "ratio": (0.5, 1.5), "value": 0.7},
    ]
    assert transformed == [image, image]
    assert torch.equal(_media(results[0]), image + 1)
    assert results[1] == {}
    assert torch.equal(_media(results[2]), image + 1)


def test_rand_augment_operations_pass_exact_transform_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.zeros(2, 3, 4, dtype=torch.uint8)
    rotate_calls: list[dict[str, object]] = []
    posterize_calls: list[dict[str, object]] = []
    solarize_calls: list[dict[str, object]] = []
    affine_calls: list[dict[str, object]] = []

    def rotate(img: Tensor, *, angle: float) -> Tensor:
        rotate_calls.append({"img": img, "angle": angle})
        return img

    def posterize(img: Tensor, *, bits: int) -> Tensor:
        posterize_calls.append({"img": img, "bits": bits})
        return img

    def solarize(img: Tensor, *, threshold: int) -> Tensor:
        solarize_calls.append({"img": img, "threshold": threshold})
        return img

    def affine(
        img: Tensor,
        *,
        angle: float,
        translate: list[int],
        scale: float,
        shear: list[float],
    ) -> Tensor:
        affine_calls.append(
            {
                "img": img,
                "angle": angle,
                "translate": translate,
                "scale": scale,
                "shear": shear,
            },
        )
        return img

    monkeypatch.setattr(tv_functional, "rotate", rotate)
    monkeypatch.setattr(tv_functional, "posterize", posterize)
    monkeypatch.setattr(tv_functional, "solarize", solarize)
    monkeypatch.setattr(tv_functional, "affine", affine)

    assert _rotate(image, 0.5) is image
    assert _posterize(image, 0.6) is image
    assert _solarize(image, 0.247) is image
    for operation in (_shear_x, _shear_y, _translate_x, _translate_y):
        assert operation(image, 1.0) is image

    assert rotate_calls == [{"img": image, "angle": 15.0}]
    assert posterize_calls == [{"img": image, "bits": 5}]
    assert solarize_calls == [{"img": image, "threshold": 192}]
    assert affine_calls == [
        {
            "img": image,
            "angle": 0,
            "translate": [0, 0],
            "scale": 1.0,
            "shear": [0.3, 0],
        },
        {
            "img": image,
            "angle": 0,
            "translate": [0, 0],
            "scale": 1.0,
            "shear": [0, 0.3],
        },
        {
            "img": image,
            "angle": 0,
            "translate": [1, 0],
            "scale": 1.0,
            "shear": [0, 0],
        },
        {
            "img": image,
            "angle": 0,
            "translate": [0, 1],
            "scale": 1.0,
            "shear": [0, 0],
        },
    ]


def test_normalize_matches_the_mean_std_formula_and_rejects_floats() -> None:
    processor = Normalize.Config().make()
    x = torch.randint(0, 256, (3, 2, 4, 5), dtype=torch.uint8)
    samples: list[Normalize.Input] = [{"media_tensor": x}, {}]

    results = list(processor(iter(samples)))

    # Normalize broadcasts ImageNet channel statistics over spatial dimensions.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    torch.testing.assert_close(_media(results[0]), (x.float() / 255 - mean) / std)
    assert "media_tensor" not in results[1]
    floats: list[Normalize.Input] = [{"media_tensor": x.float()}]
    with pytest.raises(TypeError, match="expects uint8"):
        _ = list(processor(iter(floats)))


def test_normalize_continues_after_missing_media() -> None:
    image = torch.ones(3, 2, 4, 5, dtype=torch.uint8)
    samples: list[Normalize.Input] = [
        {"media_tensor": image},
        {},
        {"media_tensor": image},
    ]

    results = list(Normalize.Config().make()(iter(samples)))

    assert len(results) == 3
    assert results[1] == {}
    torch.testing.assert_close(_media(results[0]), _media(results[2]))


def test_normalize_honors_custom_statistics_and_output_dtype() -> None:
    config = Normalize.Config(
        mean=(0.2, 0.4, 0.6),
        std=(0.1, 0.3, 0.5),
        dtype=torch.float64,
    )
    processor = config.make()
    image = torch.arange(120, dtype=torch.uint8).reshape(3, 2, 4, 5)
    normalized = next(
        iter(processor(iter([cast(Normalize.Input, {"media_tensor": image})]))),
    )
    # Channel statistics broadcast across the four spatial axes by contract.
    mean = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64).view(3, 1, 1, 1)
    std = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64).view(3, 1, 1, 1)

    assert _media(normalized).dtype == torch.float64
    assert processor.bias.dtype == torch.float64
    assert processor.scale.dtype == torch.float64
    torch.testing.assert_close(_media(normalized), (image.double() / 255 - mean) / std)


def test_normalize_keeps_cached_statistics_and_output_on_configured_device() -> None:
    config = Normalize.Config(device="meta", dtype=torch.float64)
    processor = config.make()
    image = torch.zeros(3, 2, 4, 5, dtype=torch.uint8)

    result = next(
        iter(processor(iter([cast(Normalize.Input, {"media_tensor": image})]))),
    )

    assert processor.bias.device.type == "meta"
    assert processor.scale.device.type == "meta"
    assert _media(result).device.type == "meta"
    assert _media(result).dtype == torch.float64


def _media(sample: RandomHorizontalFlip.Output) -> Tensor:
    value = sample.get("media_tensor")
    assert value is not None
    return value


def _mixup_batch() -> MixupCutmix.Input:
    images = torch.zeros(2, 3, 4, 5)
    images[1] = 1.0
    return {"image": images, "label": torch.tensor([0, 1])}


def _single_label_batch() -> MixupCutmix.Input:
    # The label is a list on purpose: ``_one_hot_with_smoothing`` accepts the
    # pre-tensorized form even though ``Input`` declares a Tensor.
    batch: dict[str, object] = {
        "image": torch.zeros(2, 3, 4, 5),
        "label": torch.tensor([1, 1]),
    }
    return cast(MixupCutmix.Input, batch)


def _mixed(sample: MixupCutmix.Output) -> tuple[Tensor, Tensor]:
    image = sample.get("image")
    label = sample.get("label")
    assert image is not None
    assert label is not None
    return image, label


def _swap_pair(n: int, device: torch.device | None = None) -> Tensor:
    del n, device
    return torch.tensor([1, 0])


def test_smoothed_one_hot_stays_with_labels_device_and_float32() -> None:
    labels = torch.tensor([0, 2], device="meta")

    targets = _one_hot_with_smoothing(labels, num_classes=4, smoothing=0.2)

    assert targets.device == labels.device
    assert targets.dtype == torch.float32
    assert targets.shape == (2, 4)


def test_smoothed_one_hot_converts_boolean_labels_to_integer_indices() -> None:
    labels: list[int] = [False, True]

    targets = _one_hot_with_smoothing(labels, num_classes=2, smoothing=0.0)

    torch.testing.assert_close(
        targets,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )


def test_zero_smoothing_keeps_float32_under_a_float64_default() -> None:
    labels = torch.tensor([0, 2])
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        targets = _one_hot_with_smoothing(labels, num_classes=4, smoothing=0.0)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert targets.dtype == torch.float32
    torch.testing.assert_close(
        targets,
        torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32),
    )


def test_mixup_cutmix_mixes_images_and_labels_by_the_same_lambda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MixupCutmix.Config(num_classes=4, label_smoothing=0.0, switch_prob=0.0)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    monkeypatch.setattr(random, "betavariate", _constant_draw(0.75))
    monkeypatch.setattr(torch, "randperm", _swap_pair)

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    image, label = _mixed(result)
    torch.testing.assert_close(image[0], torch.full((3, 4, 5), 0.25))
    torch.testing.assert_close(image[1], torch.full((3, 4, 5), 0.75))
    expected = torch.tensor([[0.75, 0.25, 0, 0], [0.25, 0.75, 0, 0]])
    torch.testing.assert_close(label, expected)


def test_mixup_includes_probability_boundary_and_uses_batch_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _mixup_batch().get("image")
    assert image is not None
    draws = iter([0.5, 0.5])
    beta_calls: list[tuple[float, float]] = []
    permutations: list[tuple[int, torch.device | None]] = []

    def beta(alpha: float, other_alpha: float) -> float:
        beta_calls.append((alpha, other_alpha))
        return 0.75

    def randperm(size: int, *, device: torch.device | None = None) -> Tensor:
        permutations.append((size, device))
        return torch.tensor([1, 0], device=device)

    monkeypatch.setattr(random, "random", lambda: next(draws))
    monkeypatch.setattr(random, "betavariate", beta)
    monkeypatch.setattr(torch, "randperm", randperm)
    config = MixupCutmix.Config(
        mixup_alpha=0.7,
        cutmix_alpha=0.0,
        prob=0.5,
        switch_prob=0.5,
        label_smoothing=0.0,
        num_classes=2,
    )

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    assert beta_calls == [(0.7, 0.7)]
    assert permutations == [(2, image.device)]
    torch.testing.assert_close(
        _mixed(result)[1],
        torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
    )


def test_mixup_switch_boundary_selects_mixup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_draws = iter([0.0, 0.5])
    beta_calls: list[tuple[float, float]] = []

    def beta(alpha: float, other_alpha: float) -> float:
        beta_calls.append((alpha, other_alpha))
        return 0.75

    monkeypatch.setattr(random, "random", lambda: next(random_draws))
    monkeypatch.setattr(random, "betavariate", beta)
    monkeypatch.setattr(torch, "randperm", _swap_pair)
    config = MixupCutmix.Config(
        mixup_alpha=0.7,
        cutmix_alpha=1.0,
        switch_prob=0.5,
        label_smoothing=0.0,
        num_classes=2,
    )

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    assert beta_calls == [(0.7, 0.7)]
    torch.testing.assert_close(
        _mixed(result)[1],
        torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
    )


def _constant_draw(value: float) -> Callable[[float, float], float]:
    def draw(alpha: float, beta: float) -> float:
        del alpha, beta
        return value

    return draw


def test_cutmix_pastes_a_box_and_rescales_lambda_to_its_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MixupCutmix.Config(num_classes=2, label_smoothing=0.0, switch_prob=1.0)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    # sqrt(1 - 0.7) * 4 truncates to a 2x2 cut; the label lambda must then be
    # the box's true 12/16, not the 0.7 that was drawn.
    beta_calls: list[tuple[float, float]] = []

    def beta(alpha: float, other_alpha: float) -> float:
        beta_calls.append((alpha, other_alpha))
        return 0.7

    image = _mixup_batch().get("image")
    assert image is not None
    permutations: list[tuple[int, torch.device | None]] = []

    def randperm(size: int, *, device: torch.device | None = None) -> Tensor:
        permutations.append((size, device))
        return torch.tensor([1, 0], device=device)

    monkeypatch.setattr(random, "betavariate", beta)
    monkeypatch.setattr(torch, "randperm", randperm)
    box_ranges: list[tuple[int, int]] = []

    def draw_box(low: int, high: int) -> int:
        box_ranges.append((low, high))
        return 2

    monkeypatch.setattr(random, "randint", draw_box)  # Box [1:3, 1:3].

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    image, label = _mixed(result)
    assert beta_calls == [(1.0, 1.0)]
    assert permutations == [(2, image.device)]
    assert box_ranges == [(0, 5), (0, 4)]
    assert image[0, :, 1:3, 1:3].sum() == 3 * 4
    assert image[0, :, :1, :1].sum() == 0
    assert image[1, :, 1:3, 1:3].sum() == 0
    torch.testing.assert_close(label, torch.tensor([[0.8, 0.2], [0.2, 0.8]]))


def test_cutmix_clips_a_corner_box_to_image_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "betavariate", _constant_draw(0.75))

    def draw_zero(low: int, high: int) -> int:
        del low, high
        return 0

    monkeypatch.setattr(random, "randint", draw_zero)
    monkeypatch.setattr(torch, "randperm", _swap_pair)
    config = MixupCutmix.Config(
        num_classes=2,
        label_smoothing=0.0,
        switch_prob=1.0,
    )

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    image, label = _mixed(result)
    assert image[0, :, 0, 0].sum() == 3
    assert image[0, :, 1:, :].sum() == 0
    torch.testing.assert_close(label, torch.tensor([[0.95, 0.05], [0.05, 0.95]]))


def test_mixup_cutmix_only_smooths_labels_when_not_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoothed = torch.tensor(
        [[0.925, 0.025, 0.025, 0.025], [0.025, 0.925, 0.025, 0.025]],
    )

    # Skipped by prob: the batch is left alone.
    monkeypatch.setattr(random, "random", lambda: 0.9)
    skipped = next(
        iter(
            MixupCutmix.Config(num_classes=4, prob=0.5).make()(iter([_mixup_batch()])),
        ),
    )
    image, label = _mixed(skipped)
    torch.testing.assert_close(label, smoothed)
    assert image.sum() == 3 * 20

    # A single-sample batch has no partner to mix with.
    monkeypatch.setattr(random, "random", lambda: 0.0)
    alone = next(
        iter(MixupCutmix.Config(num_classes=4).make()(iter([_single_label_batch()]))),
    )
    torch.testing.assert_close(
        _mixed(alone)[1],
        torch.tensor([[0.025, 0.925, 0.025, 0.025]] * 2),
    )

    # Both alphas zero: mixing is disabled outright.
    disabled = MixupCutmix.Config(num_classes=4, mixup_alpha=0.0, cutmix_alpha=0.0)
    off = next(iter(disabled.make()(iter([_mixup_batch()]))))
    torch.testing.assert_close(_mixed(off)[1], smoothed)

    # Missing fields pass through, and the stream continues past every branch.
    partial: list[MixupCutmix.Input] = [
        {"image": torch.zeros(2, 3, 4, 5)},
        {},
        _mixup_batch(),
        _single_label_batch(),
        _mixup_batch(),
    ]
    draws = iter([0.9, 0.0, 0.0, 0.0, 0.0])  # Skip the first batch by prob only.
    monkeypatch.setattr(random, "random", lambda: next(draws))
    streamed = list(MixupCutmix.Config(num_classes=4, prob=0.5).make()(iter(partial)))
    assert streamed[:2] == partial[:2]
    shapes = [tuple(_mixed(s)[1].shape) for s in streamed[2:]]
    assert shapes == [(2, 4), (2, 4), (2, 4)]


def test_mixup_cutmix_smooths_single_image_batches_and_continues() -> None:
    # MixupCutmix only label-smooths, never mixes, a batch of one image.
    images = [torch.ones(1, 3, 4, 5), torch.full((1, 3, 4, 5), 2.0)]
    labels = [1, 3]
    batches: list[MixupCutmix.Input] = [
        {"image": image, "label": torch.tensor([label])}
        for image, label in zip(images, labels, strict=True)
    ]

    outputs = list(MixupCutmix.Config(num_classes=4).make()(iter(batches)))

    assert len(outputs) == 2
    for output, batch, image, label in zip(
        outputs,
        batches,
        images,
        labels,
        strict=True,
    ):
        assert output is batch
        result_image, result_label = _mixed(output)
        assert result_image is image
        # One smoothed label row per single-image batch.
        expected = torch.full((1, 4), 0.025)
        expected[0, label] = 0.925
        torch.testing.assert_close(result_label, expected)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
