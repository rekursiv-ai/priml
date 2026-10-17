"""Tests for the two-head ResNet: shapes, registration order and init order."""

from __future__ import annotations

from typing import Final

from torch import nn

import torch

from priml.baselines.mlsbench.multitask.model import TwoHeadResNet, keep_default
from priml.testing.cost import assert_cost_matches_torch


CLASSES: Final = 3
COARSE: Final = 2


def config() -> TwoHeadResNet.Config:
    """Return a tiny two-head network: widths 2, 4, 5, one block per stage."""
    cfg = TwoHeadResNet.Config()
    cfg.channels_out = COARSE
    cfg.trunk.channels_in = 3
    cfg.trunk.channels_out = CLASSES
    cfg.trunk.channels_hidden = (2, 4, 5)
    cfg.trunk.blocks_per_stage = (1, 1, 1)
    return cfg


def test_both_heads_score_the_same_pooled_features() -> None:
    model = config().make()
    fine, coarse = model(torch.randn(2, 3, 6, 7))
    assert fine.shape == (2, CLASSES)
    assert coarse.shape == (2, COARSE)
    media = torch.ones(2, 3, 6, 7)
    features = model.trunk.features(media)
    assert torch.equal(model.trunk.head(features), model(media)[0])
    assert torch.equal(model.head(features), model(media)[1])


def bias(layer: nn.Linear) -> torch.Tensor:
    """Return a layer's bias, which these heads always have."""
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
    assert torch.equal(bias(model.trunk.head), torch.zeros(CLASSES))


def test_every_module_is_built_before_any_initializer_draws() -> None:
    """Upstream builds both heads, then initializes; the draws land in that order."""
    torch.manual_seed(0)
    model = config().make()
    torch.manual_seed(0)
    cfg = config()
    cfg.trunk.init_conv = cfg.trunk.init_head = cfg.trunk.init_bias = keep_default
    built = cfg.make()
    for module in built.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    for head in (built.trunk.head, built.head):
        nn.init.kaiming_normal_(head.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(bias(head))
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


def both_heads(model: nn.Module, media: torch.Tensor) -> torch.Tensor:
    """Join both heads' logits, so one backward reaches every parameter."""
    assert isinstance(model, TwoHeadResNet)
    return torch.cat(model(media), dim=-1)


def test_the_cost_counts_both_heads_and_matches_torch() -> None:
    cfg = config()
    cfg.trunk.image_size = 8
    analytical = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(2, 3, 8, 8, requires_grad=True),
        run=both_heads,
        batch_size=2,
        dtype=None,
    )
    trunk = cfg.trunk.cost(batch_size=2, dtype=None)
    assert analytical.params == trunk.params + 5 * COARSE + COARSE


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
