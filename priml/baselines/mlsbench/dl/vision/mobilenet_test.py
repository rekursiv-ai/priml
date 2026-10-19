"""Tests for MobileNetV2."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from configgle import InlineConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.mobilenet import (
    InvertedResidual,
    MobileNetV2,
    relu6,
    sized_blocks,
)
from priml.baselines.mlsbench.dl.vision.model import Activation
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent

BATCH: Final = 2
CLASSES: Final = 7
SIDE: Final = 6


def tiny() -> MobileNetV2.Config:
    """Return a three-stage network: an unexpanded block, a residual one, a strided one."""
    config = MobileNetV2.Config()
    config.channels_in = 3
    config.channels_out = CLASSES
    config.image_size = SIDE
    config.channels_stem = 4
    config.channels_hidden = (5, 5, 8)
    config.blocks_per_stage = (1, 2, 1)
    config.expand_ratios = (1, 2, 3)
    config.strides = (1, 1, 2)
    config.channels_last = 9
    return config


def blocks(model: MobileNetV2) -> list[InvertedResidual]:
    """Return the blocks, typed."""
    return [cast(InvertedResidual, block) for block in model.blocks]


def test_forward_returns_one_score_per_class() -> None:
    model = tiny().make()
    assert model(torch.randn(BATCH, 3, SIDE, SIDE)).shape == (BATCH, CLASSES)


def test_blocks_are_sized_and_strided_per_stage() -> None:
    sized = sized_blocks(tiny())
    configs = [cast(InvertedResidual.Config, block) for block, _ in sized]
    assert [(c.channels_in, c.channels_out) for c in configs] == [
        (4, 5),
        (5, 5),
        (5, 5),
        (5, 8),
    ]
    assert [c.expand_ratio for c in configs] == [1, 2, 2, 3]
    assert [stride for _, stride in sized] == [1, 1, 1, 2]


def test_only_a_block_whose_shape_is_unchanged_adds_its_input() -> None:
    assert [block.use_res for block in blocks(tiny().make())] == [
        False,
        True,
        True,
        False,
    ]


def test_an_unexpanded_block_has_no_expanding_convolution() -> None:
    first, second, *_ = blocks(tiny().make())
    assert len(first.conv) == 5
    assert len(second.conv) == 8
    depthwise = cast(nn.Conv2d, second.conv[3])
    assert (depthwise.groups, depthwise.in_channels) == (10, 10)


def test_a_residual_block_adds_its_input_to_the_projection() -> None:
    block = InvertedResidual.Config(channels_in=4, channels_out=4, expand_ratio=2)
    model = block.make().eval()
    with torch.no_grad():
        cast(nn.Conv2d, model.conv[6]).weight.zero_()
    x = torch.randn(BATCH, 4, 3, 5)
    assert torch.equal(model(x), x)


def test_layers_are_registered_in_the_reference_order() -> None:
    model = tiny().make()
    assert list(dict(model.named_children())) == [
        "stem",
        "blocks",
        "last",
        "pool",
        "head",
    ]
    names = [name for name, _ in model.named_parameters()]
    assert names[:3] == ["stem.0.weight", "stem.1.weight", "stem.1.bias"]
    assert names[-5:] == [
        "last.0.weight",
        "last.1.weight",
        "last.1.bias",
        "head.weight",
        "head.bias",
    ]


def test_relu6_clips_at_zero_and_six() -> None:
    assert relu6(torch.tensor([-1.0, 3.0, 7.0])).tolist() == [0.0, 3.0, 6.0]


def test_the_activation_slot_reaches_every_activation() -> None:
    config = tiny()
    config.activation = torch.sigmoid
    acts = [m for m in config.make().modules() if isinstance(m, Activation)]
    # Stem, last, two per expanded block and one for the unexpanded one.
    assert len(acts) == 2 + 2 * 3 + 1
    assert all(act.fn is torch.sigmoid for act in acts)


def test_a_stateful_activation_is_built_once_per_use() -> None:
    config = tiny()
    config.activation = InlineConfig(nn.PReLU)
    acts = [m for m in config.make().modules() if isinstance(m, nn.PReLU)]
    assert len(acts) == len({id(act) for act in acts}) == 2 + 2 * 3 + 1


def test_the_norm_slot_reaches_every_normalization() -> None:
    config = tiny()
    config.norm = nn.InstanceNorm2d
    norms = [m for m in config.make().modules() if type(m) is nn.InstanceNorm2d]
    # Stem, last, three per expanded block, two for the unexpanded one.
    assert len(norms) == 2 + 3 * 3 + 2


def test_the_pool_slot_feeds_the_head() -> None:
    config = tiny()
    config.pool = InlineConfig(nn.Sequential, nn.AdaptiveMaxPool2d(1), nn.Flatten())
    model = config.make().eval()
    media = torch.randn(BATCH, 3, SIDE, SIDE)
    pooled = model.last(model.blocks(model.stem(media))).amax(dim=(-2, -1))
    assert torch.equal(model.features(media), pooled)
    assert torch.equal(model(media), model.head(pooled))


def test_initializer_slots_reach_the_layers_they_name() -> None:
    seen: list[Tensor] = []
    config = tiny()
    config.init_conv = nn.init.ones_
    config.init_head = nn.init.zeros_
    config.init_bias = nn.init.ones_
    config.init = lambda model: seen.append(cast(MobileNetV2, model).head.weight)
    model = config.make()
    assert all(
        torch.eq(m.weight, 1).all() for m in model.modules() if isinstance(m, nn.Conv2d)
    )
    assert torch.eq(model.head.weight, 0).all()
    assert torch.eq(cast(Tensor, model.head.bias), 1).all()
    assert seen == [model.head.weight]


def test_the_default_is_the_references_cifar_mobilenetv2() -> None:
    config = MobileNetV2.Config()
    assert sum(config.blocks_per_stage) == 17
    assert (config.channels_stem, config.channels_last) == (32, 1280)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("strides", (1, 1), "one entry per stage"),
        ("blocks_per_stage", (1, 0, 1), "Every stage needs a block"),
    ],
)
def test_malformed_stages_are_refused(field: str, value: object, match: str) -> None:
    config = tiny()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=match):
        config.make()


def test_a_block_infers_the_width_it_is_not_given() -> None:
    assert (
        InvertedResidual.Config(channels_in=4).copy_tree().finalize().channels_out == 4
    )
    assert (
        InvertedResidual.Config(channels_out=5).copy_tree().finalize().channels_in == 5
    )


def test_a_block_cannot_be_costed_on_a_grid_that_is_not_square() -> None:
    with pytest.raises(ValueError, match="square"):
        InvertedResidual.Config(channels_in=4).cost(
            seq_len=6,
            batch_size=BATCH,
            dtype=None,
        )


# A real image carries no gradient, so torch would skip the first layer's input
# gradient; the cost counts it, so the input requires grad, as ``conv_test.py``'s do.
def test_the_cost_counts_every_matmul_the_forward_issues() -> None:
    _ = assert_cost_matches_torch(
        tiny(),
        build_input=lambda: torch.randn(BATCH, 3, SIDE, SIDE, requires_grad=True),
        batch_size=BATCH,
        dtype=None,
    )


def test_mobilenetv2_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="mobilenetv2",
        build_module=tiny().make,
        build_input=lambda: torch.randn(BATCH, 3, SIDE, SIDE),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
