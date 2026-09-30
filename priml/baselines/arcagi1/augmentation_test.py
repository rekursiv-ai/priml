"""ARC augmentation configuration and transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
    canonicalize_arc_grid,
)


def test_dataset_spec_controls_packed_geometry() -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.spec.max_grid = 4
    assert config.spec.grid_shape == (16,)
    assert config.spec.vocab_size == 12
    packed = config.make().spatial.pack(
        np.array([[3]], dtype=np.uint8),
        np.array([[4]], dtype=np.uint8),
        training=False,
        rng=np.random.default_rng(0),
    )
    assert all(row.shape == config.spec.grid_shape for row in packed)
    assert packed[0][0] == config.spec.vocab_color_offset + 3


def test_missing_dataset_spec_rejected() -> None:
    with pytest.raises(ValueError, match="spec must be filled"):
        ArcAugmentation.Config().make()
    with pytest.raises(ValueError, match="spec must be filled"):
        SpatialAugmentation.Config().make()


def test_token_identity_without_colors() -> None:
    config = ColorDihedral.Config()
    config.colors = ()
    config.transforms = (0,)
    grid = torch.arange(4).repeat(3, 1)
    inputs, labels = config.make().augment_tokens(
        grid,
        grid.flip(1),
        vocab_size=5,
        token_offset=0,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(inputs, grid)
    assert torch.equal(labels, grid.flip(1))


def test_color_dihedral_roundtrip() -> None:
    config = ColorDihedral.Config()
    config.separator = ArcSpec().puzzle_id_separator
    grid = np.arange(10, dtype=np.uint8).reshape(2, 5)
    augmentation = config.make()
    for seed in range(16):
        name, forward = augmentation.sample("puzzle", rng=np.random.default_rng(seed))
        original, inverse = augmentation.inverse(name)
        assert original == "puzzle"
        assert np.array_equal(inverse(forward(grid)), grid)


def test_unfilled_identifier_separator_rejected() -> None:
    policy = ColorDihedral.Config().make()
    with pytest.raises(ValueError, match="separator must be filled"):
        policy.sample("task", rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="separator must be filled"):
        policy.inverse("task")


def test_spatial_identity_preserves_rng() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.translation_prob = 0.0
    config.spec.max_grid = 4
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    grid = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    packed, _ = config.make().pack(grid, grid, training=True, rng=rng)
    # SpatialAugmentation.pack returns the dataset's square packed grid.
    assert np.array_equal(
        packed.reshape(4, 4),
        [[2, 3, 1, 0], [4, 5, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    )
    assert rng.bit_generator.state == before


def test_prediction_crop_uses_its_own_shape() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.spec.max_grid = 8
    augmentation = config.make()
    inp = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    prediction = np.array([[7, 8]], dtype=np.uint8)
    _, predicted_tokens = augmentation.pack_at(inp, out=prediction, tag=(2, 1, 2))
    name, restored = canonicalize_arc_grid(
        torch.from_numpy(predicted_tokens),
        name="task",
        spatial_tags=torch.tensor([2, 1, 2]),
        spec=config.spec,
    )
    assert name == "task"
    assert restored.shape == (1, 2)
    assert torch.equal(restored, torch.from_numpy(prediction))


def test_canonicalize_uses_configured_transform_separator() -> None:
    config = ColorDihedral.Config(separator="::", transforms=(1,))
    augmentation = config.make()
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    name, forward = augmentation.sample("task", rng=np.random.default_rng(7))
    spatial_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(spatial_config.spec, ArcSpec)
    spatial_config.spec.max_grid = 3
    packed = spatial_config.make().pack(
        forward(grid),
        forward(grid),
        training=False,
        rng=np.random.default_rng(7),
    )[0]

    original, restored = canonicalize_arc_grid(
        torch.from_numpy(packed),
        name=name,
        spatial_tags=torch.tensor([1, 0, 0]),
        spec=spatial_config.spec,
        transform=augmentation,
    )
    assert original == "task"
    assert torch.equal(restored, torch.from_numpy(grid))


def test_nested_policy_is_injectable() -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
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
    config = SpatialAugmentation.Config(spec=ArcSpec())
    config.translation_prob = probability
    with pytest.raises(ValueError, match="translation_prob"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
