"""Tests for GPU batch augmentation primitives."""

from __future__ import annotations

import pytest
import torch

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


def test_random_crop_rejects_width_oversized_crop():
    images = torch.randn(2, 3, 6, 5)
    with pytest.raises(ValueError, match="exceeds input width"):
        random_crop(images, 6)


def test_random_crop_rejects_oversized_crop():
    """crop_size larger than the input asserts instead of cropping garbage (M1)."""
    images = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="exceeds input height"):
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


def test_pad_crop_flip_contiguous():
    images = torch.randn(2, 3, 4, 5)
    out = pad_crop_flip(images, 3, pad=2)
    assert out.is_contiguous()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
