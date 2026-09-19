"""ARC augmentation configuration and exact source parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    ColorDihedral,
    SpatialAugmentation,
)


def test_token_identity_without_colors() -> None:
    config = ColorDihedral.Config()
    config.colors = ()
    config.transforms = (0,)
    grid = torch.arange(9).reshape(1, 9)
    inputs, labels = config.make().augment_tokens(
        grid,
        grid.flip(1),
        vocab_size=9,
        token_offset=0,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(inputs, grid)
    assert torch.equal(labels, grid.flip(1))


def test_color_dihedral_roundtrip() -> None:
    config = ColorDihedral.Config()
    grid = np.arange(10, dtype=np.uint8).reshape(2, 5)
    augmentation = config.make()
    for seed in range(16):
        name, forward = augmentation.sample("puzzle", rng=np.random.default_rng(seed))
        original, inverse = augmentation.inverse(name)
        assert original == "puzzle"
        assert np.array_equal(inverse(forward(grid)), grid)


def test_spatial_identity_preserves_rng() -> None:
    config = SpatialAugmentation.Config()
    config.translation_prob = 0.0
    config.max_grid = 4
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    grid = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    packed, _ = config.make().pack(grid, grid, training=True, rng=rng)
    assert np.array_equal(
        packed.reshape(4, 4),
        [[2, 3, 1, 0], [4, 5, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    )
    assert rng.bit_generator.state == before


def test_nested_policy_is_injectable() -> None:
    config = ArcAugmentation.Config()
    transform = ColorDihedral.Config()
    transform.transforms = (0,)
    transform.colors = ()
    config.transform = transform
    config.spatial.translation_prob = 0.0
    augmentation = config.make()
    grid = np.array([[1, 2]], dtype=np.uint8)
    _, forward = augmentation.transform.sample("puzzle", rng=np.random.default_rng(0))
    assert np.array_equal(forward(grid), grid)


@pytest.mark.parametrize("probability", [-1.0, 1.1, float("nan")])
def test_invalid_probability_rejected(probability: float) -> None:
    config = SpatialAugmentation.Config()
    config.translation_prob = probability
    with pytest.raises(ValueError, match="translation_prob"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
