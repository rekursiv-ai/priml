"""Tests for the image augmentation processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import random

from torch import Tensor

import pytest
import torch

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
    ],
    ids=["tall", "wide", "square"],
)
def test_random_resized_crop_falls_back_to_a_center_crop(
    monkeypatch: pytest.MonkeyPatch,
    height: int,
    width: int,
    expected: tuple[int, int, int, int],
) -> None:
    processor = GetRandomResizedCropBoxFromDimensions.Config().make()

    def oversized(low: float, high: float) -> float:
        # Every attempt asks for more area than the image has, so all ten fail.
        del low
        return high * 4.0

    monkeypatch.setattr(random, "uniform", oversized)

    sample: GetRandomResizedCropBoxFromDimensions.Input = {
        "height": height,
        "width": width,
    }
    result = next(iter(processor(iter([sample]))))

    assert result.get("crop") == expected
    assert (result.get("target_height"), result.get("target_width")) == (224, 224)


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


def test_random_horizontal_flip_mirrors_the_last_axis_with_probability_p() -> None:
    x = torch.arange(6.0).reshape(1, 2, 3)
    always: list[RandomHorizontalFlip.Input] = [{"media_tensor": x}, {}]
    never: list[RandomHorizontalFlip.Input] = [{"media_tensor": x}]

    flipped = list(RandomHorizontalFlip.Config(p=1.0).make()(iter(always)))
    kept = list(RandomHorizontalFlip.Config(p=0.0).make()(iter(never)))

    assert torch.equal(_media(flipped[0]), torch.flip(x, dims=[-1]))
    assert "media_tensor" not in flipped[1]
    assert kept[0].get("media_tensor") is x


def test_color_jitter_and_random_erasing_transform_only_present_media() -> None:
    x = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    jitter = ColorJitter.Config(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0)
    erase = RandomErasing.Config(p=1.0, scale=(0.5, 0.5), ratio=(1.0, 1.0), value=0.0)
    to_jitter: list[ColorJitter.Input] = [{"media_tensor": x}, {}]
    to_erase: list[RandomErasing.Input] = [{"media_tensor": torch.ones(3, 8, 8)}, {}]

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
    x = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    applied: list[float] = []

    def spy(img: Tensor, magnitude: float) -> Tensor:
        applied.append(magnitude)
        return img

    def choose(ops: Sequence[object]) -> object:
        del ops
        return spy

    monkeypatch.setattr(random, "choice", choose)
    fixed = RandAugment.Config(num_ops=3, magnitude=9, num_magnitude_bins=30)
    drawn = RandAugment.Config(num_ops=1, magnitude=-1)
    monkeypatch.setattr(random, "random", lambda: 0.25)
    samples: list[RandAugment.Input] = [{"media_tensor": x}, {}]

    results = list(fixed.make()(iter(samples)))
    _ = list(drawn.make()(iter(samples[:1])))

    assert applied == [0.3, 0.3, 0.3, 0.25]
    assert results[0].get("media_tensor") is x
    assert "media_tensor" not in results[1]
    for op in _RAND_AUGMENT_OPS:
        out = op(x, 0.5)
        assert out.shape == x.shape, op.__name__
        assert out.dtype == torch.uint8, op.__name__


def test_normalize_matches_the_mean_std_formula_and_rejects_floats() -> None:
    processor = Normalize.Config().make()
    x = torch.randint(0, 256, (3, 1, 4, 4), dtype=torch.uint8)
    samples: list[Normalize.Input] = [{"media_tensor": x}, {}]

    results = list(processor(iter(samples)))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    torch.testing.assert_close(_media(results[0]), (x.float() / 255 - mean) / std)
    assert "media_tensor" not in results[1]
    floats: list[Normalize.Input] = [{"media_tensor": x.float()}]
    with pytest.raises(TypeError, match="expects uint8"):
        _ = list(processor(iter(floats)))


def _media(sample: RandomHorizontalFlip.Output) -> Tensor:
    value = sample.get("media_tensor")
    assert value is not None
    return value


def _mixup_batch() -> MixupCutmix.Input:
    images = torch.zeros(2, 3, 4, 4)
    images[1] = 1.0
    return {"image": images, "label": torch.tensor([0, 1])}


def _single_label_batch() -> MixupCutmix.Input:
    # The label is a list on purpose: ``_one_hot_with_smoothing`` accepts the
    # pre-tensorized form even though ``Input`` declares a Tensor.
    batch: dict[str, object] = {"image": torch.zeros(1, 3, 4, 4), "label": [1]}
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


def test_mixup_cutmix_mixes_images_and_labels_by_the_same_lambda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MixupCutmix.Config(num_classes=4, label_smoothing=0.0, switch_prob=0.0)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    monkeypatch.setattr(random, "betavariate", _constant_draw(0.75))
    monkeypatch.setattr(torch, "randperm", _swap_pair)

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    image, label = _mixed(result)
    torch.testing.assert_close(image[0], torch.full((3, 4, 4), 0.25))
    torch.testing.assert_close(image[1], torch.full((3, 4, 4), 0.75))
    expected = torch.tensor([[0.75, 0.25, 0, 0], [0.25, 0.75, 0, 0]])
    torch.testing.assert_close(label, expected)


def _constant_draw(value: float) -> Callable[[float, float], float]:
    def draw(alpha: float, beta: float) -> float:
        del alpha, beta
        return value

    return draw


def _constant_int(value: int) -> Callable[[int, int], int]:
    def draw(low: int, high: int) -> int:
        del low, high
        return value

    return draw


def test_cutmix_pastes_a_box_and_rescales_lambda_to_its_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MixupCutmix.Config(num_classes=2, label_smoothing=0.0, switch_prob=1.0)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    # sqrt(1 - 0.7) * 4 truncates to a 2x2 cut; the label lambda must then be
    # the box's true 12/16, not the 0.7 that was drawn.
    monkeypatch.setattr(random, "betavariate", _constant_draw(0.7))
    monkeypatch.setattr(random, "randint", _constant_int(1))  # Box [0:2, 0:2].
    monkeypatch.setattr(torch, "randperm", _swap_pair)

    result = next(iter(config.make()(iter([_mixup_batch()]))))

    image, label = _mixed(result)
    assert image[0, :, :2, :2].sum() == 3 * 4
    assert image[0, :, 2:, 2:].sum() == 0
    assert image[1, :, :2, :2].sum() == 0
    # 4 of 16 pixels came from the partner: lambda = 0.75.
    torch.testing.assert_close(label, torch.tensor([[0.75, 0.25], [0.25, 0.75]]))


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
    assert image.sum() == 3 * 16

    # A single-sample batch has no partner to mix with.
    monkeypatch.setattr(random, "random", lambda: 0.0)
    alone = next(
        iter(MixupCutmix.Config(num_classes=4).make()(iter([_single_label_batch()]))),
    )
    torch.testing.assert_close(_mixed(alone)[1], smoothed[1:])

    # Both alphas zero: mixing is disabled outright.
    disabled = MixupCutmix.Config(num_classes=4, mixup_alpha=0.0, cutmix_alpha=0.0)
    off = next(iter(disabled.make()(iter([_mixup_batch()]))))
    torch.testing.assert_close(_mixed(off)[1], smoothed)

    # Missing fields pass through, and the stream continues past every branch.
    partial: list[MixupCutmix.Input] = [
        {"image": torch.zeros(2, 3, 4, 4)},
        {},
        _mixup_batch(),
        _single_label_batch(),
        _mixup_batch(),
    ]
    draws = iter([0.9, 0.0, 0.0, 0.0])  # Skip the first batch by prob only.
    monkeypatch.setattr(random, "random", lambda: next(draws))
    streamed = list(MixupCutmix.Config(num_classes=4, prob=0.5).make()(iter(partial)))
    assert streamed[:2] == partial[:2]
    shapes = [tuple(_mixed(s)[1].shape) for s in streamed[2:]]
    assert shapes == [(2, 4), (1, 4), (2, 4)]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
