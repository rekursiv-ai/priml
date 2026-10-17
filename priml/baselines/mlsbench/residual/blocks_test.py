"""Tests for the ``dl-residual-connection`` blocks, against their closed forms."""

from __future__ import annotations

from typing import Final, cast

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision.model import BasicBlock
from priml.baselines.mlsbench.residual.blocks import (
    GatedResidualBlock,
    PreActivationBlock,
    StochasticDepthBlock,
    stochastic_depth,
)
from priml.testing.cost import assert_cost_matches_torch


BATCH: Final = 2


def apply(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a tensor-to-tensor module, typed as one."""
    return cast(torch.Tensor, module(x))


def features(channels: int) -> torch.Tensor:
    """Return ``(BATCH, channels, 4, 4)`` signed features."""
    count = BATCH * channels * 4 * 4
    return ((torch.arange(count) * 7 % 23 - 11) / 8).reshape(BATCH, channels, 4, 4)


def test_pre_activation_scales_the_branch_and_projects_the_raw_input() -> None:
    config = PreActivationBlock.Config(channels_in=3, channels_out=5, stride=2)
    block = config.make()
    assert block.scale.item() == pytest.approx(0.1)
    x = features(3)
    branch = block.conv2(
        torch.relu(apply(block.norm2, block.conv1(torch.relu(apply(block.norm1, x))))),
    )
    torch.testing.assert_close(block(x), block.shortcut(x) + 0.1 * branch)
    # A module's own parameters come before its children's, upstream's ``alpha`` too.
    assert [name for name, _ in block.named_parameters()] == [
        "scale",
        "norm1.weight",
        "norm1.bias",
        "conv1.weight",
        "norm2.weight",
        "norm2.bias",
        "conv2.weight",
        "shortcut.0.weight",
        "shortcut.1.weight",
        "shortcut.1.bias",
    ]


def test_pre_activation_infers_a_missing_width() -> None:
    block = PreActivationBlock.Config(channels_out=4).make()
    assert isinstance(block.shortcut, torch.nn.Identity)
    assert block.conv1.in_channels == 4


def test_gated_block_starts_as_the_basic_block() -> None:
    config = GatedResidualBlock.Config(channels_in=3, channels_out=4)
    torch.manual_seed(0)
    gated = config.make()
    torch.manual_seed(0)
    basic = BasicBlock.Config(channels_in=3, channels_out=4).make()
    x = features(3)
    assert torch.equal(gated(x), basic(x))
    assert gated.gate.tolist() == [1.0]
    assert next(iter(dict(gated.named_parameters()))) == "gate"


def test_stochastic_depth_draws_once_per_training_forward_and_never_in_eval() -> None:
    config = StochasticDepthBlock.Config(channels_in=3, channels_out=3)
    config.depth, config.num_layers = 1, 2
    block = config.make()
    assert block.survival == 0.75
    x = features(3)
    block.eval()
    kept = apply(
        block.norm2,
        block.conv2(block.act(apply(block.norm1, block.conv1(x)))),
    )
    state = torch.get_rng_state()
    torch.manual_seed(1)
    draws = torch.rand(3)
    torch.manual_seed(1)
    for draw in draws:
        expected = block.act(kept + x) if draw.item() < 0.75 else block.act(x)
        # Training mode only for the draw; evaluation-mode norms keep it comparable.
        block.train()
        block.norm1.eval()
        block.norm2.eval()
        torch.testing.assert_close(block(x), expected)
    assert {d.item() < 0.75 for d in draws} == {True, False}
    position = torch.get_rng_state()
    block.eval()
    torch.testing.assert_close(block(x), block.act(0.75 * kept + x))
    assert torch.equal(torch.get_rng_state(), position)
    torch.set_rng_state(state)


def test_stochastic_depth_restarts_its_count_on_every_stem_width_block() -> None:
    blocks = stochastic_depth((16, 32, 64), (3, 2, 2), survival_last=0.25)
    assert all(isinstance(b, StochasticDepthBlock.Config) for b in blocks)
    depths = [b.depth for b in blocks if isinstance(b, StochasticDepthBlock.Config)]
    assert depths == [1, 1, 1, 2, 3, 4, 5]
    assert {
        (b.num_layers, b.survival_last)
        for b in blocks
        if isinstance(b, StochasticDepthBlock.Config)
    } == {(5, 0.25)}


def test_a_later_stage_at_the_stem_width_restarts_the_count_too() -> None:
    blocks = stochastic_depth((4, 4), (2, 2), survival_last=0.5)
    # The deepest position is the counter's last value, as upstream reads it,
    # even when an earlier block stands deeper.
    assert [
        (b.depth, b.num_layers)
        for b in blocks
        if isinstance(b, StochasticDepthBlock.Config)
    ] == [(1, 1), (1, 1), (2, 1), (1, 1)]


def test_pre_activation_cost_counts_every_matmul() -> None:
    analytical = assert_cost_matches_torch(
        PreActivationBlock.Config(channels_in=3, channels_out=5, stride=2),
        build_input=lambda: torch.randn(BATCH, 3, 6, 6, requires_grad=True),
        seq_len=6 * 6,
        batch_size=BATCH,
        dtype=None,
    )
    conv1, conv2, shortcut = 5 * 3 * 9, 5 * 5 * 9, 5 * 3
    assert analytical.params == conv1 + conv2 + shortcut + 2 * 3 + 2 * 2 * 5 + 1


def test_an_identity_pre_activation_block_costs_no_shortcut() -> None:
    analytical = assert_cost_matches_torch(
        PreActivationBlock.Config(channels_in=4, channels_out=4),
        build_input=lambda: torch.randn(BATCH, 4, 4, 4, requires_grad=True),
        seq_len=4 * 4,
        batch_size=BATCH,
        dtype=None,
    )
    assert analytical.params == 2 * (4 * 4 * 9) + 2 * (2 * 4) + 1


def test_gated_cost_adds_the_gate_to_the_basic_block() -> None:
    gated = assert_cost_matches_torch(
        GatedResidualBlock.Config(channels_in=3, channels_out=5, stride=2),
        build_input=lambda: torch.randn(BATCH, 3, 6, 6, requires_grad=True),
        seq_len=6 * 6,
        batch_size=BATCH,
        dtype=None,
    )
    basic = BasicBlock.Config(channels_in=3, channels_out=5, stride=2).cost(
        seq_len=6 * 6,
        batch_size=BATCH,
        dtype=None,
    )
    assert gated.params == basic.params + 1
    assert gated["flops", "primal", "elementwise"].sum() == (
        basic["flops", "primal", "elementwise"].sum() + 5 * BATCH * 3 * 3
    )


def test_a_block_cannot_be_costed_on_a_grid_that_is_not_square() -> None:
    with pytest.raises(ValueError, match="square"):
        PreActivationBlock.Config(channels_in=4, channels_out=4).cost(
            seq_len=6,
            batch_size=BATCH,
            dtype=None,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
