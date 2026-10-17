"""Tests for the ``dl-residual-connection`` blocks, against their closed forms."""

from __future__ import annotations

from typing import Final, cast

import pytest
import torch

from priml.baselines.mlsbench.dl.normalization.norms import GroupNorm
from priml.baselines.mlsbench.dl.residual.blocks import (
    GatedResidualBlock,
    PreActivationBlock,
    StochasticDepthBlock,
)
from priml.baselines.mlsbench.dl.vision.model import (
    BasicBlock,
    ResNet,
    sized_blocks,
)
from priml.baselines.mlsbench.dl.vision.reference import seeded
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
    with seeded(1):
        draws = torch.rand(3)
    with seeded(1):
        for draw in draws:
            expected = block.act(kept + x) if draw.item() < 0.75 else block.act(x)
            # Training mode only for the draw; evaluation-mode norms keep it comparable.
            block.train()
            block.norm1.eval()
            block.norm2.eval()
            torch.testing.assert_close(block(x), expected)
        position = torch.get_rng_state()
        block.eval()
        torch.testing.assert_close(block(x), block.act(0.75 * kept + x))
        assert torch.equal(torch.get_rng_state(), position)
    assert {d.item() < 0.75 for d in draws} == {True, False}


@pytest.mark.parametrize(
    ("depth", "num_layers", "match"),
    [(0, 3, "depth"), (2, 0, "num_layers")],
)
def test_stochastic_depth_rejects_a_position_before_the_first_or_an_empty_schedule(
    depth: int,
    num_layers: int,
    match: str,
) -> None:
    config = StochasticDepthBlock.Config(channels_in=3, channels_out=3)
    config.depth, config.num_layers = depth, num_layers
    with pytest.raises(ValueError, match=match):
        config.finalize()


def placed(
    channels_hidden: tuple[int, ...],
    blocks_per_stage: tuple[int, ...],
    *,
    reset_width: int = 16,
) -> list[tuple[int, int]]:
    """Size a stochastic-depth template in a ResNet; return each copy's position."""
    model = ResNet.Config()
    model.channels_hidden = channels_hidden
    model.blocks_per_stage = blocks_per_stage
    template = model.block = StochasticDepthBlock.Config()
    template.reset_width = reset_width
    copies = [block for stage in sized_blocks(model) for block, _ in stage]
    assert all(isinstance(b, StochasticDepthBlock.Config) for b in copies)
    return [
        (b.depth, b.num_layers)
        for b in copies
        if isinstance(b, StochasticDepthBlock.Config)
    ]


def test_stochastic_depth_restarts_its_count_on_every_reset_width_block() -> None:
    assert placed((16, 32, 64), (3, 2, 2)) == [(1, 5)] * 3 + [
        (d, 5) for d in range(2, 6)
    ]


def test_resnet56_positions_are_upstreams() -> None:
    assert placed((16, 32, 64), (9, 9, 9)) == [(1, 19)] * 9 + [
        (d, 19) for d in range(2, 20)
    ]


def test_a_later_stage_at_the_reset_width_restarts_the_count_too() -> None:
    # The deepest position is the counter's last value, as upstream reads it,
    # even when an earlier block stands deeper.
    assert placed((4, 4), (2, 2), reset_width=4) == [(1, 1), (1, 1), (2, 1), (1, 1)]


def test_a_network_without_a_reset_width_block_counts_every_block() -> None:
    assert placed((4, 5, 6), (1, 1, 1)) == [(1, 3), (2, 3), (3, 3)]


def test_placing_leaves_the_template_and_its_survival_alone() -> None:
    model = ResNet.Config()
    model.blocks_per_stage = (2, 1, 1)
    template = model.block = StochasticDepthBlock.Config()
    template.survival_last = 0.25
    copies = [block for stage in sized_blocks(model) for block, _ in stage]
    assert (template.depth, template.num_layers) == (1, 1)
    assert {
        b.survival_last for b in copies if isinstance(b, StochasticDepthBlock.Config)
    } == {0.25}


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


def test_pre_activation_costs_the_norm_its_slot_holds() -> None:
    config = PreActivationBlock.Config(channels_in=4, channels_out=6, stride=2)
    config.norm = GroupNorm
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(BATCH, 4, 6, 6, requires_grad=True),
        seq_len=6 * 6,
        batch_size=BATCH,
        dtype=None,
    )
    block = config.make()
    assert isinstance(block.norm1, GroupNorm)
    assert analytical.params == sum(p.numel() for p in block.parameters())


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
