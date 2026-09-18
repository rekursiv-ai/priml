"""Tests for Sequential and depth-based initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import torch

from priml.cost import Cost, cost
from priml.model.attention.self_attention import SelfAttention
from priml.model.init import mup_output
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_sequential_config_pprint() -> None:
    config = Sequential.Config(elements=Linear.Config(4, 4), repeat=2)
    assert_pprint_golden(
        test_file=__file__,
        name="sequential",
        config=config,
    )


def test_sequential_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="sequential",
        build_module=lambda: Sequential.Config(
            elements=Linear.Config(4, 4),
            repeat=2,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_single_layer():
    seq = Sequential.Config(elements=Linear.Config(64, 128)).make()
    assert len(seq) == 1
    layer = seq[0]
    assert isinstance(layer, Linear)
    assert layer.weight.shape == (128, 64)


def test_repeat():
    seq = Sequential.Config(elements=Linear.Config(128, 128), repeat=4).make()
    assert len(seq) == 4


def test_depth_index_propagation():
    """Each repeated layer gets a one-level global position."""
    seq = Sequential.Config(elements=Linear.Config(128, 128), repeat=4).make()
    for i, layer in enumerate(seq):
        assert isinstance(layer, Linear)
        assert layer.depth_index == ((i, 4),), (
            f"layer {i} depth_index={layer.depth_index}"
        )


def test_depth_propagation_nested():
    """Depth propagates through inner Sequential to Linear."""
    block = Sequential.Config(elements=Linear.Config(128, 128))
    seq = Sequential.Config(elements=block, repeat=4).make()
    for i, inner in enumerate(seq):
        assert isinstance(inner, Sequential)
        linear = inner[0]
        assert isinstance(linear, Linear)
        assert linear.depth_index == ((i, 4),), (
            f"block {i} linear depth_index={linear.depth_index}"
        )


def test_nested_depth_index_appends_local_position() -> None:
    sequence = Sequential.Config(
        elements=Linear.Config(128, 128),
        repeat=2,
        depth_index=((1, 3),),
    ).make()

    first = sequence[0]
    last = sequence[1]
    assert isinstance(first, Linear)
    assert isinstance(last, Linear)
    assert first.depth_index == ((1, 3), (0, 2))
    assert last.depth_index == ((1, 3), (1, 2))


def test_repeat_isolates_nested_config_trees() -> None:
    repeated = Sequential.Config(
        elements=TransformerBlock.Config(
            channels_in=64,
            attn=SelfAttention.Config(num_heads=4, channels_head=16),
        ),
        repeat=2,
    ).make()

    first = repeated[0]
    last = repeated[1]
    assert isinstance(first, TransformerBlock)
    assert isinstance(last, TransformerBlock)
    assert isinstance(first.attn, SelfAttention)
    assert isinstance(last.attn, SelfAttention)
    assert first.attn.depth_index == ((0, 2),)
    assert last.attn.depth_index == ((1, 2),)


def test_depth_based_init():
    """Later layers should have smaller weight std due to depth scaling."""
    torch.manual_seed(0)
    seq = Sequential.Config(elements=Linear.Config(128, 128), repeat=4).make()
    # Index 0 is unscaled; flattened index 3 divides by sqrt(4).
    first = seq[0]
    last = seq[3]
    assert isinstance(first, Linear)
    assert isinstance(last, Linear)
    std_first = first.weight.std().item()
    std_last = last.weight.std().item()
    assert std_first > std_last


def test_mup_output_with_depth():
    """MuP output init composes with depth index."""
    seq = Sequential.Config(
        elements=Linear.Config(128, 128, init_weight=mup_output),
        repeat=4,
    ).make()
    for i, layer in enumerate(seq):
        assert isinstance(layer, Linear)
        assert layer.depth_index == ((i, 4),)


def test_transformer_stack():
    m = Sequential.Config(
        elements=TransformerBlock.Config(
            channels_in=64,
            attn=SelfAttention.Config(num_heads=4, channels_head=16),
        ),
        repeat=3,
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_skip_with_ffn():
    m = Skip.Config(inner=SwiGLU.Config(channels_in=64)).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_sequential_of_norms_and_linear():
    m = Sequential.Config(
        elements=[
            RMSNorm.Config(64),
            Linear.Config(64, 128),
        ],
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 128)


def test_sequential_reset():
    m = Sequential.Config(
        elements=[
            Linear.Config(64, 64),
            Linear.Config(64, 64),
        ],
    ).make()
    m.reset_parameters()


def test_sequential_cost_sums_every_expanded_element() -> None:
    """``repeat`` is expanded by finalize, so each copy is priced once."""
    config = Sequential.Config(
        channels_in=4,
        elements=[RMSNorm.Config(elementwise_affine=True), Linear.Config(4, 4)],
        repeat=3,
    )
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.elements, list)
    assert len(finalized.elements) == 6
    expected = sum(
        (
            cost(element, seq_len=1, batch_size=1, dtype=None)
            for element in finalized.elements
        ),
        Cost(),
    )
    assert finalized.cost(seq_len=1, batch_size=1, dtype=None) == expected
    assert expected.params == 3 * (4 + 4 * 4)
    assert expected.params == sum(p.numel() for p in config.make().parameters())


def test_sequential_cost_matches_torch() -> None:
    """Every repeated projection is one matmul torch counts; the norms are free."""
    config = Sequential.Config(
        channels_in=4,
        elements=[RMSNorm.Config(elementwise_affine=True), Linear.Config(4, 4)],
        repeat=2,
    )
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 4, requires_grad=True),
        seq_len=3,
        batch_size=1,
        num_tokens=3,
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 2 * 4 * 4


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
