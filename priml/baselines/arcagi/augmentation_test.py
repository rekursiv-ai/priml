"""ARC augmentation configuration and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import math
import re

from configgle import InlineConfig

import numpy as np
import pytest

from priml.baselines.arcagi.augmentation import (
    ArcAugmentation,
    ArcSpec,
    SpatialAugmentation,
    arc_grid_to_np,
    bernoulli,
    crop_grid,
    grid_hash,
    inverse_aug,
    normalize_scale_weights,
    sample_scale_factor,
    untranslate_unscale,
)
from priml.data.color_dihedral import ColorDihedral, dihedral_transform


if TYPE_CHECKING:
    from collections.abc import Sequence


class _RecordingRng:
    def __init__(self) -> None:
        self.highs: list[int] = []

    def choice(self, values: Sequence[int], *, p: object) -> int:
        del p
        return values[0]

    def integers(self, high: int) -> int:
        self.highs.append(high)
        return 0


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
    with pytest.raises(
        ValueError,
        match=re.escape("ARC augmentation spec must be filled by its dataset."),
    ):
        ArcAugmentation.Config().make()
    with pytest.raises(
        ValueError,
        match=re.escape("Spatial augmentation spec must be filled by its owner."),
    ):
        SpatialAugmentation.Config().make()


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


def test_spatial_config_validates_grid_and_scale_weights() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=1))
    config.make()
    assert isinstance(config.spec, ArcSpec)
    config.spec.max_grid = 0
    with pytest.raises(ValueError, match=re.escape("max_grid must be positive.")):
        config.make()
    for weights in ({}, {0: 1.0}, {1: float("nan")}, {1: -1.0}, {1: 0.0}):
        config = SpatialAugmentation.Config(spec=ArcSpec())
        config.train_scale_weights = weights
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Scale weights require positive scales and finite nonnegative "
                "weights with a positive total.",
            ),
        ):
            config.make()
    invalid_weights = SpatialAugmentation.Config(spec=ArcSpec())
    invalid_weights.train_scale_weights = {1: -1.0}
    with pytest.raises(ValueError, match=r".+") as error:
        invalid_weights.make()
    assert str(error.value) == (
        "Scale weights require positive scales and finite nonnegative weights "
        "with a positive total."
    )


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
    restored = crop_grid(
        untranslate_unscale(predicted_tokens, scale=2, pad_r=1, pad_c=2),
        spec=config.spec,
    )
    assert np.array_equal(restored, prediction)


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

    original, inverse = augmentation.inverse(name)
    assert original == "task"
    assert np.array_equal(inverse(crop_grid(packed, spec=spatial_config.spec)), grid)


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
    with pytest.raises(
        ValueError,
        match=re.escape("translation_prob must be finite and in [0, 1]."),
    ):
        config.make()
    config = SpatialAugmentation.Config(spec=ArcSpec())
    config.scale_prob = probability
    with pytest.raises(
        ValueError,
        match=re.escape("scale_prob must be finite and in [0, 1]."),
    ):
        config.make()


def test_arc_grid_conversion_and_hash_pin_representation() -> None:
    grid = [[0, 9, 2], [4, 1, 8]]
    converted = arc_grid_to_np(grid, max_grid=3)
    assert converted.dtype == np.uint8
    assert np.array_equal(converted, np.array(grid, dtype=np.uint8))
    assert (
        grid_hash(converted)
        == "a750ba5a2068b0397c923082fade87163159a5825e65dc7e5e13e2874b939119"
    )
    with pytest.raises(
        ValueError,
        match=re.escape("ARC grid colors must be in 0..9."),
    ):
        arc_grid_to_np([[0, 256]], max_grid=3)
    with pytest.raises(
        ValueError,
        match=re.escape("Expected arr.shape[1] <= max_grid=3."),
    ):
        arc_grid_to_np([[0, 1, 2, 3]], max_grid=3)
    with pytest.raises(ValueError, match=re.escape("Expected grid.ndim == 2.")):
        grid_hash(np.array([1, 2], dtype=np.uint8))
    with pytest.raises(
        ValueError,
        match=re.escape("Expected grid.dtype == np.uint8."),
    ):
        grid_hash(np.array([[1, 2]], dtype=np.int64))


def test_scale_weight_normalization() -> None:
    weights = normalize_scale_weights({3: 2.0, 1: 1.0, 2: 0.0})
    assert weights == {1: 1 / 3, 3: 2 / 3}
    for invalid, message in (
        ({}, "train_scale_weights must contain at least one scale."),
        (
            {0: 1.0},
            "scale factors must be positive: {0: 1.0}.",
        ),
        (
            {1: -1.0},
            "scale weights must be finite and nonnegative: {1: -1.0}.",
        ),
        (
            {1: float("inf")},
            "scale weights must be finite and nonnegative: {1: inf}.",
        ),
        (
            {1: 0.0},
            "scale weights must include a positive weight: {1: 0.0}.",
        ),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            normalize_scale_weights(invalid)


def test_scale_sampling_renormalizes_weights_after_fit_filtering() -> None:
    grid = np.zeros((2, 3), dtype=np.uint8)
    assert sample_scale_factor(
        grid,
        grid,
        {1: 1.0, 2: 3.0, 3: 4.0},
        np.random.default_rng(0),
        max_grid=6,
    ) in (1, 2)


def test_scale_sampling_fits_both_rectangular_grids() -> None:
    inp = np.zeros((2, 3), dtype=np.uint8)
    out = np.zeros((3, 2), dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert sample_scale_factor(inp, out, {1: 1.0, 2: 1.0}, rng, max_grid=6) in (1, 2)
    assert sample_scale_factor(inp, out, {1: 1.0, 2: 1.0}, rng, max_grid=5) == 1
    assert sample_scale_factor(inp, out, {2: 1.0}, rng, max_grid=6) == 2
    assert sample_scale_factor(inp, out, {2: 1.0}, rng, max_grid=5) == 1
    tall = np.zeros((3, 2), dtype=np.uint8)
    short = np.zeros((2, 3), dtype=np.uint8)
    assert sample_scale_factor(tall, short, {2: 1.0}, rng, max_grid=5) == 1
    assert sample_scale_factor(tall, tall, {2: 1.0}, rng, max_grid=5) == 1
    wide = np.zeros((2, 3), dtype=np.uint8)
    assert sample_scale_factor(short, wide, {2: 1.0}, rng, max_grid=5) == 1
    narrow_out = np.zeros((2, 3), dtype=np.uint8)
    assert sample_scale_factor(inp, narrow_out, {1: 1.0, 2: 1.0}, rng, max_grid=5) == 1
    weighted_grid = np.zeros((2, 3), dtype=np.uint8)
    assert (
        sample_scale_factor(
            weighted_grid,
            weighted_grid,
            {1: 1e-9, 2: 1.0},
            np.random.default_rng(42),
            max_grid=6,
        )
        == 2
    )


def test_untranslate_unscale_and_crop_exact_rectangle() -> None:
    packed = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 4, 4, 5, 5],
            [0, 4, 4, 5, 5],
            [0, 6, 6, 7, 7],
            [0, 6, 6, 7, 7],
        ],
        dtype=np.uint8,
    ).flatten()
    canonical = untranslate_unscale(packed, scale=2, pad_r=1, pad_c=1)
    assert np.array_equal(
        canonical.reshape(5, 5),  # untranslate_unscale requires square packed grids.
        [
            [4, 5, 0, 0, 0],
            [6, 7, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    tokens = np.array(
        [[2, 3, 1, 0], [4, 5, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    ).flatten()
    assert np.array_equal(
        crop_grid(tokens, spec=ArcSpec()),
        np.array([[0, 1], [2, 3]], dtype=np.uint8),
    )
    # untranslate_unscale calls square_side and rejects rectangular flat grids.
    square = np.arange(9, dtype=np.uint8).reshape(3, 3).flatten()
    assert untranslate_unscale(square, scale=1, pad_r=0, pad_c=0) is square
    assert np.array_equal(
        untranslate_unscale(square, scale=1, pad_r=1, pad_c=0).reshape(
            3,
            3,
        ),  # untranslate_unscale requires square grids.
        [[3, 4, 5], [6, 7, 8], [0, 0, 0]],
    )
    assert np.array_equal(
        untranslate_unscale(square, scale=1, pad_r=0, pad_c=1).reshape(
            3,
            3,
        ),  # untranslate_unscale requires square grids.
        [[1, 2, 0], [4, 5, 0], [7, 8, 0]],
    )
    with pytest.raises(ValueError, match="scale must be >= 1, got 0"):
        untranslate_unscale(tokens, scale=0, pad_r=0, pad_c=0)
    with pytest.raises(ValueError, match="pads must be >= 0"):
        untranslate_unscale(tokens, scale=1, pad_r=-1, pad_c=0)
    with pytest.raises(ValueError, match=r".+") as error:
        untranslate_unscale(tokens[:-1], scale=1, pad_r=0, pad_c=0)
    assert str(error.value) == (
        "untranslate_unscale expects a square flat grid, got length 15."
    )


def test_crop_grid_handles_fully_colored_grids_and_upper_token_boundary() -> None:
    full = np.array([[2, 3], [4, 11]], dtype=np.uint8).flatten()
    assert np.array_equal(crop_grid(full, spec=ArcSpec()), [[0, 1], [2, 9]])
    invalid_color = np.array([[2, 12], [3, 4]], dtype=np.uint8).flatten()
    assert np.array_equal(crop_grid(invalid_color, spec=ArcSpec()), [[0], [1]])
    no_content = np.zeros(4, dtype=np.uint8)
    empty_crop = crop_grid(no_content, spec=ArcSpec())
    assert empty_crop.shape == (0, 0)
    assert not np.shares_memory(empty_crop, no_content)


def test_crop_grid_keeps_first_maximal_rectangle_on_area_ties() -> None:
    flat = np.array([[2, 3, 1], [4, 1, 0], [0, 0, 0]], dtype=np.uint8).flatten()
    cropped = crop_grid(flat, spec=ArcSpec())
    assert cropped.shape == (1, 2)
    assert np.array_equal(cropped, [[0, 1]])
    with pytest.raises(ValueError, match=r"crop_grid expects a square flat grid"):
        crop_grid(np.array([2, 3], dtype=np.uint8), spec=ArcSpec())


def test_inverse_aug_preserves_bare_grid_identity() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    name, undo = inverse_aug("bare-name", spec=ArcSpec())
    assert name == "bare-name"
    assert undo(grid) is grid


def test_inverse_aug_decodes_prefix_permutation_and_transform() -> None:
    name, undo = inverse_aug("p|||t1|||1023456789", spec=ArcSpec())
    assert name == "p"
    transformed = dihedral_transform(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        tid=1,
    )
    restored = undo(transformed)
    assert np.array_equal(restored, [[0, 2, 3], [4, 5, 6]])
    assert restored.dtype == np.uint8
    with pytest.raises(
        ValueError,
        match=re.escape(
            "invalid color-permutation suffix '0123456788' in identifier "
            "'p|||t0|||0123456788'; expected a permutation of '0123456789'.",
        ),
    ):
        inverse_aug("p|||t0|||0123456788", spec=ArcSpec())
    config = ColorDihedral.Config(separator="|||")
    with pytest.raises(ValueError, match=r"Encoded transform must be in 0\.\.7"):
        config.make().inverse("p|||t8|||0123456789")


@pytest.mark.parametrize("tid", [-1, 8])
def test_inverse_aug_rejects_a_transform_outside_the_dihedral_group(tid: int) -> None:
    with pytest.raises(ValueError, match=r"^Encoded transform must be in 0\.\.7\.$"):
        inverse_aug(f"p|||t{tid}|||0123456789", spec=ArcSpec())


def test_boundary_probabilities_preserve_rng_state() -> None:
    rng = np.random.default_rng(9)
    state = rng.bit_generator.state
    assert bernoulli(0.0, rng) is False
    assert bernoulli(1.0, rng) is True
    assert rng.bit_generator.state == state
    assert bernoulli(0.5, np.random.default_rng(3)) is True
    threshold = np.random.default_rng(7).random()
    assert bernoulli(threshold, np.random.default_rng(7)) is False


def test_arc_augmentation_accepts_boundaries_and_retains_config() -> None:
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.num_aug = 0
    config.retries_factor = 1
    augmentation = config.make()
    assert augmentation.spec.max_grid == 30
    assert augmentation.config.num_aug == 0
    assert augmentation.config.retries_factor == 1
    invalid = ArcAugmentation.Config(spec=ArcSpec())
    invalid.num_aug = -1
    with pytest.raises(
        ValueError,
        match=re.escape("num_aug must be nonnegative and retries_factor positive."),
    ):
        invalid.make()
    invalid = ArcAugmentation.Config(spec=ArcSpec())
    invalid.retries_factor = 0
    with pytest.raises(
        ValueError,
        match=re.escape("num_aug must be nonnegative and retries_factor positive."),
    ):
        invalid.make()


def test_spatial_pack_at_exact_tokens_and_boundaries() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.spec.max_grid = 5
    spatial = config.make()
    inp = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = np.array([[5, 6]], dtype=np.uint8)
    packed_in, packed_out = spatial.pack_at(inp, out=out, tag=(2, 1, 0))
    assert np.array_equal(
        packed_in.reshape(5, 5),  # SpatialAugmentation emits square packed grids.
        [
            [0, 0, 0, 0, 0],
            [3, 3, 4, 4, 1],
            [3, 3, 4, 4, 1],
            [5, 5, 6, 6, 1],
            [5, 5, 6, 6, 1],
        ],
    )
    assert np.array_equal(
        packed_out.reshape(5, 5),  # SpatialAugmentation emits square packed grids.
        [
            [0, 0, 0, 0, 0],
            [7, 7, 8, 8, 1],
            [7, 7, 8, 8, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    assert packed_in.dtype == np.uint8
    assert packed_out.dtype == np.uint8
    with pytest.raises(ValueError, match=r"got \(0, -1, 0\)"):
        spatial.pack_at(inp, out=out, tag=(0, -1, 0))
    with pytest.raises(ValueError, match="places the grid outside"):
        spatial.pack_at(inp, out=out, tag=(2, 2, 1))
    with pytest.raises(ValueError, match="places the grid outside"):
        spatial.pack_at(inp, out=out, tag=(2, 0, 2))
    tall = np.zeros((3, 2), dtype=np.uint8)
    small = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="places the grid outside"):
        spatial.pack_at(tall, out=small, tag=(1, 3, 0))
    wide_output = np.zeros((2, 3), dtype=np.uint8)
    spatial.pack_at(small, out=wide_output, tag=(1, 3, 0))


def test_pack_at_handles_padding_tokens_and_grid_edge_boundaries() -> None:
    spec = ArcSpec(max_grid=3, vocab_pad=13)
    spatial = SpatialAugmentation.Config(spec=spec).make()
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    out = np.array([[7, 8], [9, 0], [1, 2]], dtype=np.uint8)
    packed_in, packed_out = spatial.pack_at(inp, out=out, tag=(1, 0, 0))
    assert np.array_equal(
        packed_in.reshape(3, 3),  # SpatialAugmentation emits square packed grids.
        [[3, 4, 5], [6, 7, 8], [1, 1, 1]],
    )
    assert np.array_equal(
        packed_out.reshape(3, 3),  # SpatialAugmentation emits square packed grids.
        [[9, 10, 1], [11, 2, 1], [3, 4, 1]],
    )
    small = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    padded = spatial.pack_at(small, out=small, tag=(1, 0, 0))
    assert np.array_equal(
        padded[0].reshape(3, 3),  # SpatialAugmentation emits square packed grids.
        [[3, 4, 1], [5, 6, 1], [1, 1, 13]],
    )
    for invalid_tag in ((0, 0, 0), (1, -1, 0), (1, 0, -1)):
        with pytest.raises(ValueError, match="tag needs scale"):
            spatial.pack_at(inp, out=out, tag=invalid_tag)
    with pytest.raises(
        ValueError,
        match=re.escape("grid shape exceeds max_grid=3: inp=(4, 2), out=(2, 3)."),
    ):
        spatial.pack_at(
            np.zeros((4, 2), dtype=np.uint8),
            out=np.zeros((2, 3), dtype=np.uint8),
            tag=(1, 0, 0),
        )
    with pytest.raises(
        ValueError,
        match=re.escape("grid shape exceeds max_grid=3: inp=(2, 3), out=(2, 4)."),
    ):
        spatial.pack_at(
            np.zeros((2, 3), dtype=np.uint8),
            out=np.zeros((2, 4), dtype=np.uint8),
            tag=(1, 0, 0),
        )


def test_spatial_pack_with_tags_obeys_training_gates_and_exact_rng_draws() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(config.spec, ArcSpec)
    config.spec.max_grid = 4
    config.train_scale_weights = {1: 0.5, 2: 0.0}
    config.translation_prob = 0.0
    config.scale_prob = 1.0
    inp = np.array([[2, 3]], dtype=np.uint8)
    out = np.array([[4], [5]], dtype=np.uint8)
    rng = np.random.default_rng(11)
    expected_rng = np.random.default_rng(11)
    rows, tag = config.make().pack_with_tags(inp, out=out, training=True, rng=rng)
    assert tag == (1, 0, 0)
    assert np.array_equal(rows[0][:4], [4, 5, 1, 0])
    assert np.array_equal(rows[1][:4], [6, 1, 0, 0])
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    eval_rows, eval_tag = config.make().pack_with_tags(
        inp,
        out=out,
        training=False,
        rng=rng,
    )
    assert eval_tag == (1, 0, 0)
    assert np.array_equal(eval_rows, rows)


def test_pack_forwards_training_mode_and_rng() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=4))
    config.train_scale_weights = {2: 1.0}
    config.scale_prob = 1.0
    config.translation_prob = 0.0
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rng = np.random.default_rng(8)
    rows = config.make().pack(grid, grid, training=True, rng=rng)
    assert np.array_equal(
        rows[0].reshape(4, 4),  # SpatialAugmentation emits square packed grids.
        [[3, 3, 4, 4], [3, 3, 4, 4], [5, 5, 6, 6], [5, 5, 6, 6]],
    )


def test_pack_with_tags_scales_when_training_and_never_scales_in_eval() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=4))
    config.train_scale_weights = {2: 1.0}
    config.scale_prob = 1.0
    config.translation_prob = 0.0
    spatial = config.make()
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    train_rows, train_tag = spatial.pack_with_tags(
        grid,
        out=grid,
        training=True,
        rng=np.random.default_rng(9),
    )
    assert train_tag == (2, 0, 0)
    assert np.array_equal(
        train_rows[0].reshape(4, 4),  # SpatialAugmentation emits square packed grids.
        [[3, 3, 4, 4], [3, 3, 4, 4], [5, 5, 6, 6], [5, 5, 6, 6]],
    )
    eval_rows, eval_tag = spatial.pack_with_tags(
        grid,
        out=grid,
        training=False,
        rng=np.random.default_rng(9),
    )
    assert eval_tag == (1, 0, 0)
    assert np.array_equal(
        eval_rows[0].reshape(4, 4),  # SpatialAugmentation emits square packed grids.
        [[3, 4, 1, 0], [5, 6, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    )


def test_pack_with_tags_uses_rng_for_both_probability_gates() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=4))
    config.train_scale_weights = {2: 1.0}
    config.scale_prob = 0.5
    config.translation_prob = 0.5
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    _, tag = config.make().pack_with_tags(
        grid,
        out=grid,
        training=True,
        rng=np.random.default_rng(3),
    )
    assert tag == (2, 0, 0)


def test_pack_with_tags_uses_scaled_extents_for_translation_bounds() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=14))
    config.train_scale_weights = {2: 1.0}
    rng = _RecordingRng()
    inp = np.zeros((2, 6), dtype=np.uint8)
    out = np.zeros((4, 3), dtype=np.uint8)
    rows, tag = config.make().pack_with_tags(
        inp,
        out=out,
        training=True,
        rng=cast(np.random.Generator, rng),
    )
    assert tag == (2, 0, 0)
    assert rng.highs == [7, 3]
    assert rows[0].shape == rows[1].shape == (14**2,)


def test_pack_with_tags_rejects_oversized_input_before_rng_draws() -> None:
    config = SpatialAugmentation.Config(spec=ArcSpec(max_grid=3))
    config.scale_prob = 0.5
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="grid shape exceeds max_grid=3"):
        config.make().pack_with_tags(
            np.zeros((4, 2), dtype=np.uint8),
            out=np.zeros((2, 3), dtype=np.uint8),
            training=True,
            rng=rng,
        )
    assert rng.bit_generator.state == before
    with pytest.raises(
        ValueError,
        match=re.escape("grid shape exceeds max_grid=3: inp=(2, 3), out=(4, 2)."),
    ):
        config.make().pack_with_tags(
            np.zeros((2, 3), dtype=np.uint8),
            out=np.zeros((4, 2), dtype=np.uint8),
            training=True,
            rng=rng,
        )
    assert rng.bit_generator.state == before
    edge = np.zeros((3, 2), dtype=np.uint8)
    edge_rows, edge_tag = config.make().pack_with_tags(
        edge,
        out=np.zeros((2, 3), dtype=np.uint8),
        training=False,
        rng=rng,
    )
    assert edge_tag == (1, 0, 0)
    assert edge_rows[0].shape == (9,)


def test_inverse_transform_outputs_do_not_alias_inputs() -> None:
    grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    policy = ColorDihedral.Config(separator="|||", transforms=(1,)).make()
    name, forward = policy.sample("task", rng=np.random.default_rng(0))
    augmented = forward(grid)
    _, inverse = policy.inverse(name)
    restored = inverse(augmented)
    assert np.array_equal(restored, grid)
    assert not np.shares_memory(restored, augmented)
    _, inverse = inverse_aug(name, spec=ArcSpec())
    restored = inverse(augmented)
    assert np.array_equal(restored, grid)
    assert not np.shares_memory(restored, augmented)


def test_arc_grid_conversion_pins_each_validation_boundary() -> None:
    with pytest.raises(ValueError, match=r"Expected arr\.ndim == 2\."):
        arc_grid_to_np([], max_grid=4)
    with pytest.raises(ValueError, match=r"arr\.shape\[0\] <= max_grid=2"):
        arc_grid_to_np([[1], [2], [3]], max_grid=2)
    with pytest.raises(ValueError, match=r"arr\.shape\[1\] <= max_grid=2"):
        arc_grid_to_np([[1, 2, 3]], max_grid=2)
    for invalid in ([[-1, 0]], [[10]]):
        with pytest.raises(
            ValueError,
            match=re.escape("ARC grid colors must be in 0..9."),
        ):
            arc_grid_to_np(invalid, max_grid=2)
    assert np.array_equal(arc_grid_to_np([[9]], max_grid=1), [[9]])
    assert np.array_equal(arc_grid_to_np([[9.0]], max_grid=1), [[9]])


@pytest.mark.parametrize("cell", [9.5, -0.5, 9.9, math.nan, math.inf])
def test_arc_grid_conversion_rejects_non_integral_colors(cell: float) -> None:
    """A fractional color must not truncate into range before the check."""
    with pytest.raises(ValueError, match=re.escape("ARC grid colors must be in 0..9.")):
        arc_grid_to_np([[cell, 0]], max_grid=2)


def test_scale_weight_normalization_survives_huge_weights() -> None:
    assert normalize_scale_weights({1: 1e308, 2: 1e308}) == {1: 0.5, 2: 0.5}


def test_augmentation_finalize_accepts_any_makeable_transform() -> None:
    """``transform`` is a slot: a config of another type must not trip an assert."""
    config = ArcAugmentation.Config(spec=ArcSpec())
    config.transform = InlineConfig(_question_mark_policy)
    augmentation = config.finalize().make()
    assert augmentation.transform.config.separator == "?"


def _question_mark_policy() -> ColorDihedral:
    return ColorDihedral(ColorDihedral.Config(separator="?"))


def test_arc_grid_conversion_errors_match_exact_messages() -> None:
    with pytest.raises(ValueError, match=r".+") as error:
        arc_grid_to_np([[10]], max_grid=1)
    assert str(error.value) == "ARC grid colors must be in 0..9."
    with pytest.raises(ValueError, match=r".+") as error:
        arc_grid_to_np([], max_grid=1)
    assert str(error.value) == "Expected arr.ndim == 2."


def test_validation_errors_match_exact_messages() -> None:
    invalid_calls = (
        (
            lambda: grid_hash(np.arange(24, dtype=np.uint8).reshape(2, 3, 4)),
            "Expected grid.ndim == 2.",
        ),
        (
            lambda: grid_hash(np.array([[1]], dtype=np.int64)),
            "Expected grid.dtype == np.uint8.",
        ),
        (
            lambda: normalize_scale_weights({}),
            "train_scale_weights must contain at least one scale.",
        ),
        (
            lambda: ArcAugmentation.Config(spec=ArcSpec(), num_aug=-1).make(),
            "num_aug must be nonnegative and retries_factor positive.",
        ),
        (
            lambda: ArcAugmentation.Config().make(),
            "ARC augmentation spec must be filled by its dataset.",
        ),
        (
            lambda: SpatialAugmentation.Config().make(),
            "Spatial augmentation spec must be filled by its owner.",
        ),
        (
            lambda: SpatialAugmentation.Config(spec=ArcSpec(max_grid=0)).make(),
            "max_grid must be positive.",
        ),
    )
    for call, expected in invalid_calls:
        with pytest.raises(ValueError, match=r".+") as error:
            call()
        assert str(error.value) == expected


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
