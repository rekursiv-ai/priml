"""Tests for Sequential and depth-based initialization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from priml.model.attention.self_attention import SelfAttention
from priml.model.init import mup_output
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    import pytest


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_sequential_config_pprint(request: pytest.FixtureRequest) -> None:
    config = Sequential.Config(Linear.Config(4, 4), repeat=2)
    assert_text_golden(
        request,
        test_file=__file__,
        name="sequential",
        rendered=config.pformat(hide_default_values=False),
    )


def test_sequential_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="sequential",
        build_module=lambda: Sequential.Config(
            Linear.Config(4, 4),
            repeat=2,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_single_layer():
    seq = Sequential.Config(Linear.Config(64, 128)).make()
    assert len(seq) == 1
    assert seq[0].weight.shape == (128, 64)


def test_repeat():
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    assert len(seq) == 4


def test_depth_index_propagation():
    """Each repeated layer gets a one-level global position."""
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    for i, layer in enumerate(seq):
        assert layer.depth_index == ((i, 4),), (
            f"layer {i} depth_index={layer.depth_index}"
        )


def test_depth_propagation_nested():
    """Depth propagates through inner Sequential to Linear."""
    block = Sequential.Config(Linear.Config(128, 128))
    seq = Sequential.Config(block, repeat=4).make()
    for i, inner in enumerate(seq):
        assert isinstance(inner, Sequential)
        linear = inner[0]
        assert linear.depth_index == ((i, 4),), (
            f"block {i} linear depth_index={linear.depth_index}"
        )


def test_nested_depth_index_appends_local_position() -> None:
    sequence = Sequential.Config(
        Linear.Config(128, 128),
        repeat=2,
        depth_index=((1, 3),),
    ).make()

    assert sequence[0].depth_index == ((1, 3), (0, 2))
    assert sequence[1].depth_index == ((1, 3), (1, 2))


def test_repeat_isolates_nested_config_trees() -> None:
    repeated = Sequential.Config(
        TransformerBlock.Config(
            channels_in=64,
            attn=SelfAttention.Config(num_heads=4, channels_head=16),
        ),
        repeat=2,
    ).make()

    assert repeated[0].attn.depth_index == ((0, 2),)
    assert repeated[1].attn.depth_index == ((1, 2),)


def test_depth_based_init():
    """Later layers should have smaller weight std due to depth scaling."""
    torch.manual_seed(0)
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    # Index 0 is unscaled; flattened index 3 divides by sqrt(4).
    std_first = seq[0].weight.std().item()
    std_last = seq[3].weight.std().item()
    assert std_first > std_last


def test_mup_output_with_depth():
    """MuP output init composes with depth index."""
    seq = Sequential.Config(
        Linear.Config(128, 128, init_weight=mup_output),
        repeat=4,
    ).make()
    for i, layer in enumerate(seq):
        assert layer.depth_index == ((i, 4),)


def test_transformer_stack():
    m = Sequential.Config(
        TransformerBlock.Config(
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
        [
            RMSNorm.Config(64),
            Linear.Config(64, 128),
        ],
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 128)


def test_sequential_reset():
    m = Sequential.Config(
        [
            Linear.Config(64, 64),
            Linear.Config(64, 64),
        ],
    ).make()
    m.reset_parameters()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
