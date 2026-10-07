"""Tests for GPU batch augmentation primitives."""

from __future__ import annotations

import pytest
import torch

from priml.data import augmentation_gpu
from priml.data.augmentation_gpu import (
    cutout,
    flip_lr,
    pad_crop_flip,
    random_crop,
)


def test_flip_lr_shape():
    images = torch.randn(8, 3, 4, 5)
    out = flip_lr(images)
    assert out.shape == images.shape


def test_flip_lr_stochastic():
    """With enough images, some should be flipped and some not."""
    torch.manual_seed(0)
    images = torch.arange(6000).float().view(100, 3, 4, 5)
    out = flip_lr(images)
    flipped = (out[:, 0, 0, 0] != images[:, 0, 0, 0]).sum().item()
    assert 20 < flipped < 80  # ~50% should flip.


def test_random_crop_shape():
    images = torch.randn(8, 3, 4, 5)  # Pre-padded.
    out = random_crop(images, 3)
    assert out.shape == (8, 3, 3, 3)


def test_random_crop_content():
    """Cropped output should be a sub-region of the padded input."""
    torch.manual_seed(42)
    images = torch.randn(2, 3, 36, 37)
    out = random_crop(images, 3)
    # Every value in out should exist in images.
    for i in range(min(10, out.numel())):
        val = out.flatten()[i].item()
        assert bool((images == val).any())


def test_random_crop_accepts_crop_equal_to_input_size():
    # random_crop takes one square crop_size; 4 equals both H and W.
    images = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    assert torch.equal(random_crop(images, 4), images)


def test_random_crop_rejects_width_oversized_crop():
    images = torch.randn(2, 3, 6, 5)
    with pytest.raises(
        ValueError,
        match=r"^crop_size=6 exceeds input width W=5; pad before cropping\.$",
    ):
        random_crop(images, 6)


def test_random_crop_rejects_oversized_crop():
    """crop_size larger than the input asserts instead of cropping garbage (M1)."""
    images = torch.randn(2, 3, 4, 5)
    with pytest.raises(
        ValueError,
        match=r"^crop_size=5 exceeds input height H=4; pad before cropping\.$",
    ):
        random_crop(images, 5)


def test_cutout_shape():
    images = torch.randn(8, 3, 4, 5)
    out = cutout(images, 8)
    assert out.shape == images.shape


def test_cutout_zeros():
    """Cutout should zero out an 8x8 region."""
    torch.manual_seed(0)
    images = torch.ones(2, 3, 4, 5)
    out = cutout(images, 8)
    zeros = (out == 0).sum().item()
    # Each image gets 3 * 8 * 8 = 192 zeros (when cutout fully inside).
    assert zeros > 0
    assert zeros <= 2 * 3 * 8 * 8


def test_pad_crop_flip_explicit_flip_modes():
    images = torch.ones(2, 3, 4, 5)
    assert torch.equal(
        pad_crop_flip(images, 4, pad=1, flip=False),
        torch.ones_like(images[..., :4]),
    )
    assert pad_crop_flip(images, 4, pad=1, flip=True).shape == images[..., :4].shape


def test_pad_crop_flip_shape():
    images = torch.randn(8, 3, 4, 5)
    out = pad_crop_flip(images, 3, pad=2)
    assert out.shape == (8, 3, 3, 3)


def test_pad_crop_flip_with_cutout():
    images = torch.ones(2, 3, 4, 5)
    out = pad_crop_flip(images, 4, pad=2, cutout_size=2)
    assert out.shape == (2, 3, 4, 4)
    assert (out == 0).any()  # Cutout should have zeroed some pixels.


def test_pad_crop_flip_zero_cutout_skips_cutout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def record_cutout(images: torch.Tensor, size: int) -> torch.Tensor:
        calls.append(size)
        return images

    def fixed_randint(
        low: int,
        high: int,
        size: tuple[int, int, int, int],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del low, high
        return torch.zeros(size, dtype=torch.long, device=device)

    monkeypatch.setattr(augmentation_gpu, "cutout", record_cutout)
    monkeypatch.setattr(torch, "randint", fixed_randint)
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    out = pad_crop_flip(images, 3, pad=1, flip=False, cutout_size=0)

    assert calls == []
    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="reflect")
    assert torch.equal(out, padded[..., :3, :3])


def test_pad_crop_flip_contiguous():
    images = torch.randn(2, 3, 4, 5)
    out = pad_crop_flip(images, 3, pad=2)
    assert out.is_contiguous()


def test_pad_crop_flip_default_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    calls: list[tuple[int, int, tuple[int, int, int, int]]] = []

    def fixed_randint(
        low: int,
        high: int,
        size: tuple[int, int, int, int],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        assert device == images.device
        calls.append((low, high, size))
        return torch.zeros(size, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    out = pad_crop_flip(images, 2, flip=False)

    padded = torch.nn.functional.pad(images, (2, 2, 2, 2), mode="reflect")
    assert calls == [(0, 7, (2, 1, 1, 1)), (0, 7, (2, 1, 1, 1))]
    assert torch.equal(out, padded[..., :2, :2])


def test_pad_crop_flip_false_never_flips(monkeypatch: pytest.MonkeyPatch) -> None:
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    def fixed_randint(
        low: int,
        high: int,
        size: tuple[int, int, int, int],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del low, high
        return torch.zeros(size, dtype=torch.long, device=device)

    def fixed_rand(size: int, *, device: torch.device) -> torch.Tensor:
        assert size == 2
        return torch.full((size,), 0.25, device=device)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(torch, "rand", fixed_rand)
    out = pad_crop_flip(images, 3, pad=1, flip=False)

    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="reflect")
    assert torch.equal(out, padded[..., :3, :3])


def test_cutout_uses_random_top_left_and_clips_at_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    starts = iter(
        (
            torch.tensor([[[[0]]], [[[2]]]]),
            torch.tensor([[[[0]]], [[[2]]]]),
        ),
    )

    def record_randint(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return next(starts)

    monkeypatch.setattr(torch, "randint", record_randint)
    images = torch.ones(2, 3, 4, 5)

    out = cutout(images, 3)

    assert calls == [
        ((0, 4, (2, 1, 1, 1)), {"device": images.device}),
        ((0, 5, (2, 1, 1, 1)), {"device": images.device}),
    ]
    expected = images.clone()
    expected[0, :, :3, :3] = 0
    expected[1, :, 2:, 2:5] = 0
    assert torch.equal(out, expected)


def test_random_crop_uses_each_axis_and_batch_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    offsets = iter((torch.tensor([[[[1]]], [[[0]]]]), torch.tensor([[[[0]]], [[[1]]]])))

    def record_randint(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return next(offsets)

    monkeypatch.setattr(torch, "randint", record_randint)
    images = torch.arange(2 * 3 * 5 * 6).reshape(2, 3, 5, 6)

    out = random_crop(images, 3)

    assert calls == [
        ((0, 3, (2, 1, 1, 1)), {"device": images.device}),
        ((0, 3, (2, 1, 1, 1)), {"device": images.device}),
    ]
    assert torch.equal(out, torch.stack((images[0, :, 1:4, :3], images[1, :, :3, 1:4])))


def test_flip_lr_flips_only_selected_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_rand(size: int, *, device: torch.device) -> torch.Tensor:
        assert size == 2
        assert device == torch.device("cpu")
        return torch.tensor([0.25, 0.5])

    monkeypatch.setattr(torch, "rand", fixed_rand)
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    out = flip_lr(images)

    assert torch.equal(out, torch.stack((images[0].flip(-1), images[1])))


def test_pad_crop_flip_exact_order_and_zero_cutout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offsets = iter(
        (
            torch.tensor([[[[1]]], [[[0]]]]),
            torch.tensor([[[[0]]], [[[1]]]]),
            torch.tensor([[[[0]]], [[[0]]]]),
            torch.tensor([[[[0]]], [[[0]]]]),
        ),
    )

    def fixed_randint(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        return next(offsets)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    out = pad_crop_flip(
        images,
        3,
        pad=1,
        pad_mode="replicate",
        flip=True,
        cutout_size=1,
    )

    expected = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")
    expected = torch.stack((expected[0, :, 1:4, :3], expected[1, :, :3, 1:4]))
    expected = expected.flip(-1)
    expected[:, :, :1, :1] = 0
    assert torch.equal(out, expected)
    assert out.is_contiguous()


def test_augmentation_random_factories_keep_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[torch.device | None] = []
    randint = torch.randint
    rand = torch.rand

    def record_randint(
        low: int,
        high: int,
        size: tuple[int, int, int, int],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        devices.append(device)
        return randint(low, high, size, device=device)

    def record_rand(
        size: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        devices.append(device)
        return rand(size, device=device)

    monkeypatch.setattr(torch, "randint", record_randint)
    monkeypatch.setattr(torch, "rand", record_rand)
    images = torch.ones(2, 3, 4, 5, device="meta")

    flip_lr(images)
    random_crop(images, 2)
    cutout(images, 2)

    assert devices == [images.device] * 5


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
