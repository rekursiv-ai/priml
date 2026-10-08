"""Unit tests for the board encoder at tiny sizes on the CPU.

These pin the default geometry and its derivation, the zero-initialized
start, the previous action's path, the ConvNeXt trunk's geometry, init and
operation order, and the tiny encoders' bits. The tiny encoder's config is the
tiny board policy's, which ``model_test.py``'s golden pins. Their bits against
the recipe runs' encoders, at full size, are checked by the parity suite
(README, "Parity").
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
)

from torch import Tensor, nn
from torch.nn import functional

import pytest
import torch

from priml.baselines.craftax.policies.encoder import (
    MLP,
    ActionConditionedStatus,
    BoardCNN,
    BoardEncoder,
    ConvNextBlock,
    ConvNextTrunk,
    DepthwiseResidual,
    FilmBranch,
    MultiscaleMix,
)
from priml.baselines.craftax.testing import (
    fill_portable,
    packed_observations,
    tiny_board_policy,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from configgle import Makeable

    from priml.baselines.craftax.model import MinGRUPolicy


_CWD: Final = Path(__file__).resolve().parent


def test_the_default_encoder_is_the_reference_geometry() -> None:
    """844 floats in, 1,635 features out, through the reference's 76,480 weights."""
    config = BoardEncoder.Config()
    assert config.observation_size == 844
    assert config.channels_concat == 1635
    encoder = config.make()
    shapes = [
        (name, tuple(weight.shape)) for name, weight in encoder.named_parameters()
    ]
    assert shapes == [
        ("cells.weight", (154, 16)),
        ("status.embedding.weight", (44, 32)),
        ("status.blocks.0.proj_in.weight", (192, 83)),
        ("status.blocks.0.proj_out.weight", (51, 192)),
        ("status.blocks.1.proj_in.weight", (192, 83)),
        ("status.blocks.1.proj_out.weight", (51, 192)),
        ("board.blocks.0.depthwise.weight", (16, 1, 3, 3)),
        ("board.blocks.0.pointwise.weight", (16, 16, 1, 1)),
        ("board.blocks.1.depthwise.weight", (16, 1, 3, 3)),
        ("board.blocks.1.pointwise.weight", (16, 16, 1, 1)),
        ("board.spatial.proj_in.weight", (128, 16, 1, 1)),
        ("board.spatial.proj_condition.weight", (256, 51)),
        ("board.spatial.depthwise.weight", (128, 1, 3, 3)),
        ("board.spatial.proj_out.weight", (16, 128, 1, 1)),
        ("board.multiscale.proj_in.weight", (32, 48, 1, 1)),
        ("board.multiscale.proj_out.weight", (16, 32, 1, 1)),
    ]
    assert sum(weight.numel() for weight in encoder.parameters()) == 76_480
    assert all(layer.bias is None for layer in _layers(encoder))


def test_every_branch_output_starts_at_zero_and_nothing_else_does() -> None:
    encoder = BoardEncoder.Config().make()
    for name, weight in encoder.named_parameters():
        zero = name.endswith(("pointwise.weight", "proj_out.weight"))
        assert bool((weight == 0).all()) == zero, name


def test_the_encoder_starts_as_the_multi_hot_embedding() -> None:
    """With every branch at zero, the features are the cells' embedding, bit for bit."""
    config = tiny_board_policy()
    encoder = _tiny_encoder(config).make()
    observations = packed_observations(config, batch=3, time=2, seed=1)
    features = encoder(observations)
    assert features.shape == (3, 2, 99 * 2 + 51)
    assert torch.equal(features, encoder.cells(observations[..., :-1]))


def test_the_previous_action_reaches_the_features_through_its_table_row() -> None:
    config = tiny_board_policy()
    policy = config.make()
    fill_portable(policy, seed=2)
    encoder = policy.embedding
    assert isinstance(encoder, BoardEncoder)
    observations = packed_observations(config, batch=3, seed=3).bfloat16()
    observations[:, -1] = 5
    other = observations.clone()
    other[:, -1] = 43
    with torch.no_grad():
        assert not torch.equal(encoder(observations), encoder(other))
        encoder.status.embedding.weight[43] = encoder.status.embedding.weight[5]
        assert torch.equal(encoder(observations), encoder(other))


def test_a_board_that_does_not_hold_the_cells_is_refused() -> None:
    config = BoardEncoder.Config()
    config.board.grid = (9, 10)
    with pytest.raises(ValueError, match="does not hold 99 cells"):
        config.make()


def test_an_mlp_infers_its_widths_and_applies_its_activation_between() -> None:
    config = MLP.Config()
    config.channels_in = 3
    config.activation = torch.tanh
    mlp = config.make()
    assert tuple(mlp.proj_in.weight.shape) == (3, 3)
    assert tuple(mlp.proj_out.weight.shape) == (3, 3)
    inputs = torch.rand(2, 3)
    expected = mlp.proj_out(torch.tanh(mlp.proj_in(inputs)))
    assert torch.equal(mlp(inputs), expected)


def test_the_tiny_encoder_matches_its_bfb_golden() -> None:
    """The bf16 encoder's features and every weight's gradient, frozen on every host."""
    config = tiny_board_policy()
    encoder = _tiny_encoder(config)
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="board_encoder_tiny",
        build_module=lambda: encoder.make().to(torch.bfloat16),
        build_input=lambda: packed_observations(config, batch=2, time=3).bfloat16(),
        run=_encode_and_differentiate,
    )


def test_the_convnext_trunk_in_the_blocks_slot_is_the_reference_geometry() -> None:
    """One trunk of 64 channels and two blocks: 147,488 weights in the encoder."""
    config = BoardEncoder.Config()
    config.board.block = ConvNextTrunk.Config()
    config.board.num_layers = 1
    encoder = config.make()
    block = [
        ("layer_scale", (64,)),
        ("depthwise.weight", (64, 1, 5, 5)),
        ("norm.weight", (64,)),
        ("norm.bias", (64,)),
        ("mlp.proj_in.weight", (256, 64)),
        ("mlp.proj_in.bias", (256,)),
        ("mlp.proj_out.weight", (64, 256)),
        ("mlp.proj_out.bias", (64,)),
    ]
    assert [
        (name, tuple(weight.shape))
        for name, weight in encoder.board.blocks.named_parameters()
    ] == [
        ("0.stem.weight", (64, 16, 1, 1)),
        *(
            (f"0.blocks.{index}.{name}", shape)
            for index in (0, 1)
            for name, shape in block
        ),
        ("0.proj_out.weight", (16, 64, 1, 1)),
    ]
    assert sum(weight.numel() for weight in encoder.parameters()) == 147_488


def test_the_trunk_draws_torchs_defaults_and_starts_as_the_identity() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk(channels=2).make()
    for block in trunk.blocks:
        assert torch.equal(block.layer_scale, torch.full((6,), 1e-6))
        assert torch.equal(block.norm.weight, torch.ones(6))
        assert block.norm.bias is not None
        assert not block.norm.bias.any()
        for layer, fan_in in ((block.mlp.proj_in, 6), (block.mlp.proj_out, 24)):
            assert layer.bias is not None
            bound = fan_in**-0.5
            assert bound / 2 < float(layer.bias.detach().abs().max()) <= bound
    assert not trunk.proj_out.weight.any()
    board = torch.randn(3, 2, 4, 5)
    with torch.no_grad():
        assert torch.equal(trunk(board), board)


def test_a_convnext_block_runs_the_reference_operation_order() -> None:
    """Depthwise 5 x 5; per cell LayerNorm, linear, exact GELU, linear, scale; add."""
    config = ConvNextBlock.Config()
    config.channels_in = 3
    block = config.make().to(torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator) * 2 - 1)
    board = (torch.rand(2, 3, 4, 5, generator=generator) * 4 - 2).bfloat16()
    mixed = functional.conv2d(board, block.depthwise.weight, padding=2, groups=3)
    mixed = functional.layer_norm(
        mixed.permute(0, 2, 3, 1),
        (3,),
        block.norm.weight,
        block.norm.bias,
        eps=1e-6,
    )
    proj_in, proj_out = block.mlp.proj_in, block.mlp.proj_out
    mixed = functional.linear(mixed, proj_in.weight, proj_in.bias)
    mixed = functional.linear(functional.gelu(mixed), proj_out.weight, proj_out.bias)
    expected = board + (mixed * block.layer_scale).permute(0, 3, 1, 2)
    with torch.no_grad():
        assert torch.equal(block(board), expected)


def test_the_tiny_c2_encoder_matches_its_bfb_golden() -> None:
    """The tiny encoder with a ConvNeXt trunk: features and every weight's gradient.

    One row of three frames, so the golden stays under the testdata size cap.
    """
    config = tiny_board_policy()
    encoder = _tiny_encoder(config)
    encoder.board.block = _tiny_trunk()
    encoder.board.num_layers = 1
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="board_encoder_c2_tiny",
        build_module=lambda: encoder.make().to(torch.bfloat16),
        build_input=lambda: packed_observations(config, batch=1, time=3).bfloat16(),
        run=_encode_and_differentiate,
    )


@pytest.mark.parametrize(
    "config",
    [
        DepthwiseResidual.Config(channels_in=4),
        ConvNextBlock.Config(channels_in=4, expansion=2),
        ConvNextTrunk.Config(channels_in=4, channels_hidden=6),
        MultiscaleMix.Config(channels_in=4, channels_hidden=7),
    ],
    ids=["depthwise", "convnext_block", "convnext_trunk", "multiscale"],
)
def test_each_board_stage_costs_the_convolutions_torch_runs(
    config: Makeable[nn.Module],
) -> None:
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 4, 3, 5, requires_grad=True),
        input_grid=(3, 5),
        batch_size=2,
        dtype=None,
    )


def test_the_film_branch_costs_its_convolutions_and_the_conditions_projection() -> None:
    analytical = assert_cost_matches_torch(
        FilmBranch.Config(channels_in=4, channels_hidden=5, channels_condition=3),
        build_input=lambda: (
            torch.randn(2, 4, 3, 5, requires_grad=True),
            torch.randn(2, 3, requires_grad=True),
        ),
        input_grid=(3, 5),
        batch_size=2,
        dtype=None,
    )
    # FiLM: one add per board's wide channel, then a multiply and an add per
    # wide value; both activations are silu's five per value.
    wide = 2 * 5 * 3 * 5
    assert analytical["flops", "primal", "elementwise"].sum() == (
        2 * wide + 2 * 5 + 2 * 5 * wide + 2 * 4 * 3 * 5
    )


def test_an_mlp_costs_both_projections_and_its_activation() -> None:
    analytical = assert_cost_matches_torch(
        MLP.Config(3, 5, channels_hidden=4),
        build_input=lambda: torch.randn(2, 6, 3, requires_grad=True),
        seq_len=6,
        batch_size=2,
        dtype=None,
    )
    assert analytical["flops", "primal", "elementwise"].sum() == 5 * 2 * 6 * 4


def test_the_board_cnn_costs_one_board_per_row() -> None:
    config = BoardCNN.Config(channels_in=2, channels_condition=4)
    config.spatial.channels_hidden = 5
    config.multiscale.channels_hidden = 7
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 99 * 2, requires_grad=True),
            torch.randn(2, 3, 4, requires_grad=True),
        ),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )


def test_the_status_stage_costs_its_lookup_and_blocks() -> None:
    config = ActionConditionedStatus.Config(channels_in=5)
    config.embedding.channels_out = 3
    config.block.channels_hidden = 4
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 5, requires_grad=True),
            torch.randint(0, 44, (2, 3)).float(),
        ),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )


def test_the_tiny_encoder_costs_what_torch_runs() -> None:
    config = tiny_board_policy(dtype=torch.float32)
    assert_cost_matches_torch(
        _tiny_encoder(config),
        build_input=lambda: packed_observations(config, batch=2, time=3),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )


def _tiny_encoder(config: MinGRUPolicy.Config) -> BoardEncoder.Config:
    """Return the tiny board policy's encoder, narrowed from the slot's protocol."""
    encoder = config.embedding
    assert isinstance(encoder, BoardEncoder.Config)
    return encoder


def _tiny_trunk(*, channels: int = -1) -> ConvNextTrunk.Config:
    """Return a trunk 6 channels wide, distinct from the tiny encoder's other widths."""
    config = ConvNextTrunk.Config()
    config.channels_in = channels
    config.channels_hidden = 6
    return config


def _encode_and_differentiate(module: nn.Module, observations: Tensor) -> Tensor:
    """Return the features, then every weight's gradient of a fixed weighted sum, in fp32."""
    features = cast("object", module(observations))
    assert isinstance(features, Tensor)
    weights = torch.linspace(-1.0, 1.0, features.numel()).reshape(features.shape)
    (features.float() * weights).sum().backward()
    gradients: list[Tensor] = []
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        gradients.append(parameter.grad.float().flatten())
    return torch.cat((features.float().flatten(), *gradients))


def _layers(module: nn.Module) -> list[nn.Linear | nn.Conv2d]:
    """Return every projection and convolution in the module."""
    return [
        layer for layer in module.modules() if isinstance(layer, (nn.Linear, nn.Conv2d))
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
