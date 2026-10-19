"""Tests for the augmentation baselines, against torchvision's pipelines on PIL images.

torchvision draws from the global generator and the port from a stream; both
are seeded alike, so equal pixels mean equal draws in equal order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import augment
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.reference import seeded


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

SEED: Final = 3
CONFIG: Final = Cifar100Data.Config()


def images(count: int, *, seed: int = 0) -> Tensor:
    """Return ``count`` uint8 images of 8x10 random pixels."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (count, 3, 8, 10), generator=generator).to(torch.uint8)


def statistics() -> dict[str, Tensor]:
    return {"mean": torch.tensor(CONFIG.mean), "std": torch.tensor(CONFIG.std)}


def gray_images(count: int, *, seed: int = 0) -> Tensor:
    """Return ``count`` single-channel uint8 images of 8x10 random pixels."""
    return images(count, seed=seed)[:, :1].clone()


def repeat3(x: Tensor) -> Tensor:
    """Repeat one channel to three, the reference's Fashion-MNIST ``Lambda``."""
    return x.repeat(3, 1, 1)


def torchvision_pipeline(
    first: list[Callable[..., object]],
    batch: Tensor,
    *,
    last: list[Callable[..., object]] | None = None,
) -> Tensor:
    """Run ``first`` + crop, mirror, ToTensor + ``last`` + Normalize per image.

    A one-channel batch is fed as ``L`` images and repeated to three channels
    after ``ToTensor``, as the reference's Fashion-MNIST pipeline does.
    """
    repeat = [transforms.Lambda(repeat3)] if batch.shape[1] == 1 else []
    pipeline = transforms.Compose(
        [
            *first,
            transforms.RandomCrop((8, 10), padding=CONFIG.translate_pad),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *repeat,
            *(last or []),
            transforms.Normalize(CONFIG.mean, CONFIG.std),
        ],
    )
    with seeded(SEED):
        return torch.stack(
            [
                pipeline(
                    Image.fromarray(image.permute(1, 2, 0).squeeze(-1).numpy()),
                )
                for image in batch
            ],
        )


def table(policy: object, *args: object) -> object:
    """Return a torchvision policy's operation table, which it keeps private."""
    method = cast("Callable[..., object]", vars(type(policy))["_augmentation_space"])
    return method(policy, *args)


class Cutout:
    """The Cutout edit's mask, on a float tensor after ``ToTensor``."""

    def __call__(self, img: Tensor) -> Tensor:
        height, width = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        y = int(torch.randint(0, height, (1,)).item())
        x = int(torch.randint(0, width, (1,)).item())
        mask[
            :,
            max(0, y - 2) : min(height, y + 2),
            max(0, x - 2) : min(width, x + 2),
        ] = 0
        return img * mask


def test_cutout_matches_the_masked_pipeline() -> None:
    batch = images(5)
    got = augment.cutout_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        holes=1,
        length=4,
        **statistics(),
    )
    want = torchvision_pipeline([], batch, last=[Cutout()])
    assert torch.equal(got, want)
    # One black pixel, broadcast over the batch: what Cutout's patch normalizes to.
    assert (
        got == normalize(torch.zeros(1, 3, 1, 1, dtype=torch.uint8), **statistics())
    ).any()


def test_randaugment_matches_torchvision() -> None:
    batch = images(16)
    got = augment.randaugment_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        num_ops=2,
        magnitude=9,
        num_magnitude_bins=31,
        **statistics(),
    )
    want = torchvision_pipeline(
        [transforms.RandAugment(num_ops=2, magnitude=9)],
        batch,
    )
    assert torch.equal(got, want)


def test_trivialaugment_matches_torchvision() -> None:
    batch = images(16)
    got = augment.trivialaugment_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        num_magnitude_bins=31,
        **statistics(),
    )
    want = torchvision_pipeline([transforms.TrivialAugmentWide()], batch)
    assert torch.equal(got, want)


def test_cutout_on_grayscale_matches_the_masked_repeated_pipeline() -> None:
    batch = gray_images(5)
    got = augment.cutout_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        holes=1,
        length=4,
        **statistics(),
    )
    assert torch.equal(got, torchvision_pipeline([], batch, last=[Cutout()]))


# 64 images reach all 14 operations of each policy on the grayscale path.
def test_randaugment_on_grayscale_matches_torchvision_on_l_images() -> None:
    batch = gray_images(64)
    got = augment.randaugment_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        num_ops=2,
        magnitude=9,
        num_magnitude_bins=31,
        **statistics(),
    )
    want = torchvision_pipeline(
        [transforms.RandAugment(num_ops=2, magnitude=9)],
        batch,
    )
    assert got.shape == (64, 3, 8, 10)
    assert torch.equal(got, want)


def test_trivialaugment_on_grayscale_matches_torchvision_on_l_images() -> None:
    batch = gray_images(64)
    got = augment.trivialaugment_normalize(
        batch,
        torch.Generator().manual_seed(SEED),
        pad=CONFIG.translate_pad,
        num_magnitude_bins=31,
        **statistics(),
    )
    want = torchvision_pipeline([transforms.TrivialAugmentWide()], batch)
    assert torch.equal(got, want)


@pytest.mark.parametrize(
    ("space", "theirs"),
    [
        (
            augment.randaugment_space(31, height=8, width=10),
            cast(
                "dict[str, tuple[Tensor, bool]]",
                table(transforms.RandAugment(), 31, (8, 10)),
            ),
        ),
        (
            augment.trivialaugment_space(31),
            cast(
                "dict[str, tuple[Tensor, bool]]",
                table(transforms.TrivialAugmentWide(), 31),
            ),
        ),
    ],
    ids=["randaugment", "trivialaugment"],
)
def test_each_space_lists_torchvisions_operations_and_bins_in_order(
    space: augment.Space,
    theirs: dict[str, tuple[Tensor, bool]],
) -> None:
    assert len(space) == len(theirs)
    for (_, magnitudes, signed), (want, want_signed) in zip(
        space,
        theirs.values(),
        strict=True,
    ):
        assert signed == want_signed
        assert torch.equal(magnitudes, want)


@pytest.mark.parametrize(
    ("op", "magnitude", "shift"),
    [
        (augment.translate_x, 5.7, [5, 0]),
        (augment.translate_x, -3.7, [-3, 0]),
        (augment.translate_y, 4.7, [0, 4]),
        (augment.translate_y, -6.7, [0, -6]),
    ],
    ids=["x", "-x", "y", "-y"],
)
def test_a_translation_matches_torchvisions_policy_operation(
    op: Callable[[Tensor, float], Tensor],
    magnitude: float,
    shift: list[int],
) -> None:
    # 32 pixels a side, so a stray degree of shear moves the far rows and columns.
    generator = torch.Generator().manual_seed(SEED)
    image = torch.randint(0, 256, (3, 32, 32), generator=generator).to(torch.uint8)
    theirs = cast("Callable[..., Image.Image]", functional.affine)(
        Image.fromarray(image.permute(1, 2, 0).numpy()),
        angle=0.0,
        translate=shift,
        scale=1.0,
        shear=[0.0, 0.0],
    )
    want = torch.from_numpy(np.asarray(theirs).copy()).permute(2, 0, 1)
    assert torch.equal(op(image, magnitude), want)


def test_every_operation_of_both_spaces_runs_on_an_image() -> None:
    image = images(1)[0]
    for space in (
        augment.randaugment_space(31, height=8, width=10),
        augment.trivialaugment_space(31),
    ):
        assert len(space) == 14
        for op, magnitudes, _ in space:
            value = float(magnitudes[-1]) if magnitudes.ndim > 0 else 0.0
            assert op(image, -value).shape == image.shape


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
