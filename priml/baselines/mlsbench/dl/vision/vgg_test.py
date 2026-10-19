"""Tests for VGG-16-BN."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from configgle import InlineConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.model import Activation
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent

BATCH: Final = 2
CLASSES: Final = 7
SIDE: Final = 8


def tiny() -> VGG16BN.Config:
    """Return a three-stage network over 8x8 images: 8 -> 4 -> 2 -> 1."""
    config = VGG16BN.Config()
    config.channels_in = 3
    config.channels_out = CLASSES
    config.image_size = SIDE
    config.channels_hidden = ((4, 5), (6,), (9,))
    config.channels_hidden_head = 10
    return config


def test_forward_returns_one_score_per_class() -> None:
    model = tiny().make()
    assert model(torch.randn(BATCH, 3, SIDE, SIDE)).shape == (BATCH, CLASSES)


def test_layers_are_registered_in_the_reference_order() -> None:
    model = tiny().make()
    kinds = [type(layer).__name__ for layer in model.layers]
    block = ["Conv2d", "BatchNorm", "Activation"]
    pool = "MaxPool2d"
    assert kinds == [*block, *block, pool, *block, pool, *block, pool]
    assert list(dict(model.named_children())) == ["layers", "pool", "head"]
    assert [type(layer).__name__ for layer in model.head] == [
        "Linear",
        "Activation",
        "Dropout",
        "Linear",
    ]


def test_every_linear_layer_takes_the_head_initializers_and_convs_keep_their_bias() -> (
    None
):
    config = tiny()
    config.init_conv = nn.init.ones_
    config.init_head = nn.init.zeros_
    config.init_bias = nn.init.ones_
    model = config.make()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2
    assert all(torch.eq(m.weight, 1).all() for m in convs)
    assert all(cast(Tensor, m.bias).abs().sum() > 0 for m in convs)
    assert all(torch.eq(m.weight, 0).all() for m in linears)
    assert all(torch.eq(cast(Tensor, m.bias), 1).all() for m in linears)


def test_the_activation_slot_reaches_every_stage_and_the_classifier() -> None:
    config = tiny()
    config.activation = torch.sigmoid
    model = config.make()
    acts = [m for m in model.modules() if isinstance(m, Activation)]
    assert len(acts) == 4 + 1
    assert all(act.fn is torch.sigmoid for act in acts)


def test_a_stateful_activation_is_built_once_per_use() -> None:
    config = tiny()
    config.activation = InlineConfig(nn.PReLU)
    acts = [m for m in config.make().modules() if isinstance(m, nn.PReLU)]
    assert len(acts) == len({id(act) for act in acts}) == 5


def test_the_norm_slot_reaches_every_convolution() -> None:
    config = tiny()
    config.norm = nn.InstanceNorm2d
    norms = [m for m in config.make().modules() if type(m) is nn.InstanceNorm2d]
    assert len(norms) == 4


def test_the_pool_slot_feeds_the_classifier() -> None:
    config = tiny()
    config.pool = InlineConfig(nn.Sequential, nn.AdaptiveMaxPool2d(1), nn.Flatten())
    model = config.make().eval()
    media = torch.randn(BATCH, 3, SIDE, SIDE)
    pooled = model.layers(media).amax(dim=(-2, -1))
    assert torch.equal(model.features(media), pooled)
    assert torch.equal(model(media), model.head(pooled))


def test_the_init_hook_runs_after_the_per_tensor_initializers() -> None:
    seen: list[Tensor] = []
    config = tiny()
    config.init_conv = nn.init.ones_
    config.init = lambda model: seen.append(
        cast(nn.Conv2d, cast(VGG16BN, model).layers[0]).weight.clone(),
    )
    _ = config.make()
    (first,) = seen
    assert torch.eq(first, 1).all()


def test_a_stage_without_a_convolution_is_refused() -> None:
    config = tiny()
    config.channels_hidden = ((4,), ())
    with pytest.raises(ValueError, match="Every stage needs a convolution"):
        config.make()


def test_the_default_is_vgg16() -> None:
    config = VGG16BN.Config()
    assert sum(map(len, config.channels_hidden)) == 13
    assert (config.channels_hidden_head, config.dropout) == (512, 0.5)


# A real image carries no gradient, so torch would skip the first layer's input
# gradient; the cost counts it, so the input requires grad, as ``conv_test.py``'s do.
def test_the_cost_counts_every_matmul_the_forward_issues() -> None:
    analytical = assert_cost_matches_torch(
        tiny(),
        build_input=lambda: torch.randn(BATCH, 3, SIDE, SIDE, requires_grad=True),
        batch_size=BATCH,
        dtype=None,
    )
    convs = [(3, 4, 8), (4, 5, 8), (5, 6, 4), (6, 9, 2)]
    products = sum(c_in * c_out * 9 * side**2 for c_in, c_out, side in convs)
    assert analytical["flops", "primal", "matmul"].sum() == 2 * BATCH * (
        products + 9 * 10 + 10 * CLASSES
    )


def test_vgg16bn_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="vgg16bn",
        build_module=lambda: tiny().make().eval(),
        build_input=lambda: torch.randn(BATCH, 3, SIDE, SIDE),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
