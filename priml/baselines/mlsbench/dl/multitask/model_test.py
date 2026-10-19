"""Tests for the two-head network: shapes, registration order and init order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import nn

import pytest
import torch

from priml.baselines.mlsbench.dl.multitask.model import (
    TrunkConfig,
    TwoHeadResNet,
    keep_default,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable

CLASSES: Final = 3
COARSE: Final = 2
SIDE: Final = 8


def resnet() -> ResNet.Config:
    """Return a tiny ResNet: widths 2, 4, 5, one block per stage."""
    trunk = ResNet.Config()
    trunk.channels_hidden = (2, 4, 5)
    trunk.blocks_per_stage = (1, 1, 1)
    trunk.image_size = SIDE
    return trunk


def vgg() -> VGG16BN.Config:
    """Return a tiny VGG: two stages, a 6-wide classifier."""
    trunk = VGG16BN.Config()
    trunk.channels_hidden = ((4,), (5,))
    trunk.channels_hidden_head = 6
    trunk.image_size = SIDE
    return trunk


def mobilenet() -> MobileNetV2.Config:
    """Return a tiny MobileNetV2: two stages, a 7-wide last convolution."""
    trunk = MobileNetV2.Config()
    trunk.channels_stem = 4
    trunk.channels_hidden = (5, 6)
    trunk.blocks_per_stage = (1, 1)
    trunk.expand_ratios = (1, 2)
    trunk.strides = (1, 2)
    trunk.channels_last = 7
    trunk.image_size = SIDE
    return trunk


TRUNKS: Final[list[Callable[[], TrunkConfig]]] = [resnet, vgg, mobilenet]

TRUNK_IDS: Final = ["resnet", "vgg", "mobilenet"]


def config(trunk: Callable[[], TrunkConfig] = resnet) -> TwoHeadResNet.Config:
    """Return a tiny two-head network over ``trunk``."""
    cfg = TwoHeadResNet.Config()
    cfg.channels_out = COARSE
    cfg.trunk = trunk()
    assert isinstance(cfg.trunk, ResNet.Config | VGG16BN.Config | MobileNetV2.Config)
    cfg.trunk.channels_in = 3
    cfg.trunk.channels_out = CLASSES
    return cfg


def media() -> torch.Tensor:
    """Return a fixed ``(2, 3, SIDE, SIDE)`` batch."""
    return (torch.arange(2 * 3 * SIDE * SIDE) % 7 - 3).reshape(2, 3, SIDE, SIDE) / 4


def fine_head(model: TwoHeadResNet) -> nn.Module:
    """Return the trunk's own head."""
    return model.trunk.get_submodule("head")


@pytest.mark.parametrize("trunk", TRUNKS, ids=TRUNK_IDS)
def test_both_heads_score_the_same_pooled_features(
    trunk: Callable[[], TrunkConfig],
) -> None:
    model = config(trunk).make().eval()
    fine, coarse = model.forward(media())
    assert fine.shape == (2, CLASSES)
    assert coarse.shape == (2, COARSE)
    features = model.trunk.features(media())
    assert torch.equal(cast(torch.Tensor, fine_head(model)(features)), fine)
    assert torch.equal(cast(torch.Tensor, model.head(features)), coarse)


@pytest.mark.parametrize("trunk", TRUNKS, ids=TRUNK_IDS)
def test_the_added_head_is_the_trunks_own_at_another_width(
    trunk: Callable[[], TrunkConfig],
) -> None:
    model = config(trunk).make()
    own = [(n, p.shape) for n, p in fine_head(model).named_parameters()]
    added = [(n, p.shape) for n, p in model.head.named_parameters()]
    assert [type(m) for m in fine_head(model).modules()] == [
        type(m) for m in model.head.modules()
    ]
    # The last layer's weight and bias are the only tensors whose width differs.
    assert own[:-2] == added[:-2]
    assert [shape[0] for _, shape in own[-2:]] == [CLASSES, CLASSES]
    assert [shape[0] for _, shape in added[-2:]] == [COARSE, COARSE]


def bias(layer: nn.Module) -> torch.Tensor:
    """Return a linear layer's bias, which these heads always have."""
    assert isinstance(layer, nn.Linear)
    assert layer.bias is not None
    return layer.bias


def test_the_added_head_is_registered_after_the_trunk() -> None:
    names = [name for name, _ in config().make().named_parameters()]
    assert names[-4:] == [
        "trunk.head.weight",
        "trunk.head.bias",
        "head.weight",
        "head.bias",
    ]


def test_both_heads_take_the_trunks_initializers() -> None:
    model = config().make()
    assert torch.equal(bias(model.head), torch.zeros(COARSE))
    assert torch.equal(bias(fine_head(model)), torch.zeros(CLASSES))


@pytest.mark.parametrize("trunk", TRUNKS, ids=TRUNK_IDS)
def test_every_module_is_built_before_any_initializer_draws(
    trunk: Callable[[], TrunkConfig],
) -> None:
    """Upstream builds both heads, then initializes; the draws land in that order."""
    torch.manual_seed(0)
    model = config(trunk).make()
    torch.manual_seed(0)
    cfg = config(trunk)
    cfg.trunk.init_conv = cfg.trunk.init_head = cfg.trunk.init_bias = keep_default
    built = cfg.make()
    for module in built.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(bias(module))
    for (name, want), got in zip(
        built.state_dict().items(),
        model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(want, got), name


def test_the_whole_model_hook_sees_the_trunk() -> None:
    seen: list[nn.Module] = []
    cfg = config()
    cfg.trunk.init = seen.append
    model = cfg.make()
    assert seen == [model.trunk]


def test_keep_default_leaves_a_tensor_alone() -> None:
    w = torch.arange(3.0)
    keep_default(w)
    assert torch.equal(w, torch.arange(3.0))


def test_the_trunk_config_is_not_mutated_by_the_build() -> None:
    cfg = config()
    before = cfg.trunk.init_conv
    _ = cfg.make()
    assert cfg.trunk.init_conv is before


def both_heads(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Join both heads' logits, so one backward reaches every parameter."""
    assert isinstance(model, TwoHeadResNet)
    return torch.cat(model(batch), dim=-1)


@pytest.mark.parametrize("trunk", TRUNKS, ids=TRUNK_IDS)
def test_the_cost_counts_both_heads_and_matches_torch(
    trunk: Callable[[], TrunkConfig],
) -> None:
    cfg = config(trunk)
    analytical = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(2, 3, SIDE, SIDE, requires_grad=True),
        run=both_heads,
        batch_size=2,
        dtype=None,
    )
    alone = cfg.trunk.cost(batch_size=2, dtype=None)
    head = cfg.trunk.head_cost(channels_out=COARSE, batch_size=2, dtype=None)
    assert analytical.params == alone.params + head.params


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
