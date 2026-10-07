"""Tests for the image augmentation processors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import random

from torch import Tensor
from torchvision.transforms import v2

import pytest
import torch

from priml.data.processors.augmentation import (
    ColorJitter,
    GetCenterCropBoxFromDimensions,
    GetRandomResizedCropBoxFromDimensions,
    MixupCutmix,
    Normalize,
    RandAugment,
    RandomErasing,
    RandomHorizontalFlip,
    _one_hot_with_smoothing,
)


if TYPE_CHECKING:
    from collections.abc import Callable


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

    samples: list[GetRandomResizedCropBoxFromDimensions.Input] = [
        {"height": 6, "width": 8},
    ]
    result = next(
        iter(
            processor(
                iter(
                    samples,
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

    samples: list[GetRandomResizedCropBoxFromDimensions.Input] = [
        {"height": 1, "width": 1},
    ]
    result = next(
        iter(
            processor(
                iter(
                    samples,
                ),
            ),
        ),
    )

    assert result.get("crop") == (0, 0, 1, 1)
    assert integer_ranges == [(0, 0), (0, 0)]


def test_center_crop_box_takes_the_centered_square_for_a_square_target() -> None:
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


def test_center_crop_uses_configured_fraction_of_the_fitted_box() -> None:
    processor = GetCenterCropBoxFromDimensions.Config(
        size=(12, 18),
        ratio=0.5,
    ).make()
    sample: GetCenterCropBoxFromDimensions.Input = {"height": 81, "width": 121}

    result = next(iter(processor(iter([sample]))))

    # The 2:3 box fitting 81x121 is 80.67x121; half of it is 40x60.
    assert result.get("crop") == (20, 30, 40, 60)
    assert (result.get("target_height"), result.get("target_width")) == (12, 18)


def test_crop_processors_continue_after_samples_missing_dimensions() -> None:
    random_crop = GetRandomResizedCropBoxFromDimensions.Config().make()
    center_crop = GetCenterCropBoxFromDimensions.Config().make()
    samples: list[GetRandomResizedCropBoxFromDimensions.Input] = [
        {"height": 50},
        {"height": 80, "width": 120},
    ]
    random_results = list(
        random_crop(
            iter(
                samples,
            ),
        ),
    )
    samples_2: list[GetCenterCropBoxFromDimensions.Input] = [
        {"width": 50},
        {"height": 80, "width": 120},
    ]
    center_results = list(
        center_crop(
            iter(
                samples_2,
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
    monkeypatch.setattr(random, "random", lambda: next(values))
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

    monkeypatch.setattr(v2, "ColorJitter", make_transform)
    image = torch.zeros(3, 2, 4, 5, dtype=torch.uint8)
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
    # Each call sees frames-first ``(F, C, H, W)``, the layout torchvision reads.
    assert [frames.shape for frames in transformed] == [(2, 3, 4, 5)] * 2
    assert torch.equal(_media(results[0]), image + 1)
    assert results[1] == {}
    assert torch.equal(_media(results[2]), image + 1)


def test_color_jitter_and_random_erasing_transform_only_present_media() -> None:
    x = _video(2)
    jitter = ColorJitter.Config(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    erase = RandomErasing.Config(p=1.0, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=0.0)
    to_jitter: list[ColorJitter.Input] = [{"media_tensor": x}, {}]
    # RandomErasing takes a [C, T, H, W] clip; one frame keeps the count per channel.
    to_erase: list[RandomErasing.Input] = [
        {"media_tensor": torch.ones(3, 1, 8, 9)},
        {},
    ]

    jittered = list(jitter.make()(iter(to_jitter)))
    erased = list(erase.make()(iter(to_erase)))

    # Zero-strength jitter is the identity, and an absent field passes through.
    assert torch.equal(_media(jittered[0]), x)
    assert "media_tensor" not in jittered[1]
    # p=1 always erases a square region: torchvision rounds sqrt(32) to a
    # 6x6 box, so 36 pixels per channel are now the fill value.
    assert int((_media(erased[0]) == 0).sum()) == 3 * 36
    assert "media_tensor" not in erased[1]


def test_rand_augment_skips_samples_without_media() -> None:
    x = _video(2)
    samples: list[RandAugment.Input] = [{"media_tensor": x}, {}]

    results = list(RandAugment.Config().make()(iter(samples)))

    assert _media(results[0]).shape == x.shape
    assert results[1] == {}


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

    monkeypatch.setattr(v2, "RandomErasing", make_transform)
    image = torch.zeros(3, 2, 4, 5)
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
    assert [frames.shape for frames in transformed] == [(2, 3, 4, 5)] * 2
    assert torch.equal(_media(results[0]), image + 1)
    assert results[1] == {}
    assert torch.equal(_media(results[2]), image + 1)


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
    samples: list[Normalize.Input] = [
        {"media_tensor": image},
    ]
    normalized = next(
        iter(processor(iter(samples))),
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

    samples: list[Normalize.Input] = [
        {"media_tensor": image},
    ]
    result = next(
        iter(processor(iter(samples))),
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
    return {"media_tensor": images, "label": torch.tensor([0, 1])}


def _single_label_batch() -> MixupCutmix.Input:
    return {"media_tensor": torch.zeros(2, 3, 4, 5), "label": torch.tensor([1, 1])}


def _image(sample: MixupCutmix.Output) -> Tensor:
    image = sample.get("media_tensor")
    assert isinstance(image, Tensor)
    return image


def _mixed(sample: MixupCutmix.Output) -> tuple[Tensor, Tensor]:
    label = sample.get("label")
    assert isinstance(label, Tensor)
    return _image(sample), label


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
    image = _image(_mixup_batch())
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

    image = _image(_mixup_batch())
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
        {"media_tensor": torch.zeros(2, 3, 4, 5)},
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
        {"media_tensor": image, "label": torch.tensor([label])}
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


def _video(frames: int, *, seed: int = 0) -> Tensor:
    """Return a ``(C, F, H, W)`` uint8 clip whose frames differ from each other."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (3, frames, 8, 9),
        dtype=torch.uint8,
        generator=generator,
    )


def _on_frames_first(transform: Callable[[Tensor], Tensor], clip: Tensor) -> Tensor:
    """Apply ``transform`` to ``(F, C, H, W)``, as torchvision expects."""
    return transform(clip.moveaxis(0, 1)).moveaxis(1, 0)


def test_color_jitter_treats_the_channel_axis_as_color() -> None:
    clip = _video(5)
    sample: ColorJitter.Input = {"media_tensor": clip}
    torch.manual_seed(0)
    jittered = _media(next(iter(ColorJitter.Config(hue=0.2).make()(iter([sample])))))
    torch.manual_seed(0)
    expected = _on_frames_first(
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        clip,
    )

    assert jittered.shape == clip.shape
    assert torch.equal(jittered, expected)


def test_color_jitter_saturation_changes_a_single_frame_image() -> None:
    image = _video(1)
    sample: ColorJitter.Input = {"media_tensor": image}
    jitter = ColorJitter.Config(brightness=0.0, contrast=0.0, saturation=0.9, hue=0.0)
    torch.manual_seed(1)

    out = _media(next(iter(jitter.make()(iter([sample])))))

    assert not torch.equal(out, image)


def test_rand_augment_is_torchvisions_on_each_frame() -> None:
    clip = _video(2)
    sample: RandAugment.Input = {"media_tensor": clip}
    config = RandAugment.Config(num_ops=3, magnitude=7, num_magnitude_bins=11)
    torch.manual_seed(3)
    out = _media(next(iter(config.make()(iter([sample])))))
    torch.manual_seed(3)
    expected = _on_frames_first(
        v2.RandAugment(num_ops=3, magnitude=7, num_magnitude_bins=11),
        clip,
    )

    assert torch.equal(out, expected)


def test_rand_augment_default_magnitude_is_timms_m9() -> None:
    config = RandAugment.Config()
    # Timm divides by a fixed 10 levels; torchvision indexes num_bins evenly
    # spaced values from 0, so 11 bins put level 9 at 9/10 of the range.
    assert config.magnitude / (config.num_magnitude_bins - 1) == 9 / 10


def test_rand_augment_rejects_a_float_image() -> None:
    sample: RandAugment.Input = {"media_tensor": torch.zeros(3, 2, 4, 5)}
    with pytest.raises(TypeError, match="uint8"):
        list(RandAugment.Config().make()(iter([sample])))


def test_rand_augment_rejects_an_out_of_range_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        RandAugment.Config(magnitude=11, num_magnitude_bins=11).make()


def test_rand_augment_solarize_at_magnitude_zero_does_not_raise() -> None:
    clip = _video(2)
    rand_augment = RandAugment.Config(num_ops=4, magnitude=0).make()
    for _ in range(8):
        sample: RandAugment.Input = {"media_tensor": clip}
        out = _media(next(iter(rand_augment(iter([sample])))))
        assert out.shape == clip.shape


def test_center_crop_matches_the_target_aspect() -> None:
    processor = GetCenterCropBoxFromDimensions.Config(size=(224, 448)).make()
    sample: GetCenterCropBoxFromDimensions.Input = {"height": 300, "width": 400}

    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == (50, 0, 200, 400)


def test_center_crop_rejects_a_ratio_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="ratio"):
        GetCenterCropBoxFromDimensions.Config(ratio=1.5).make()


def test_mixup_cutmix_reads_media_tensor_by_default() -> None:
    batch: dict[str, object] = {
        "media_tensor": torch.zeros(2, 3, 4, 5),
        "label": torch.tensor([0, 1]),
    }
    processor = MixupCutmix.Config(num_classes=2, label_smoothing=0.0, prob=0.0).make()

    result = next(iter(processor(iter([batch]))))

    assert isinstance(result["label"], Tensor)
    assert result["label"].shape == (2, 2)


def test_mixup_cutmix_accepts_list_labels_when_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "randperm", _swap_pair)
    batch: dict[str, object] = {
        "media_tensor": torch.zeros(2, 3, 4, 5),
        "label": [0, 1],
    }
    processor = MixupCutmix.Config(num_classes=2, label_smoothing=0.0, prob=1.0).make()

    result = next(iter(processor(iter([batch]))))

    assert isinstance(result["label"], Tensor)
    assert result["label"].shape == (2, 2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
