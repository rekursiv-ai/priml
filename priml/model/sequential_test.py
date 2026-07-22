"""Tests for Sequential and depth-based initialization."""

from __future__ import annotations

import torch

from priml.model.attention import SelfAttention
from priml.model.init import mup_output
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_single_layer():
    seq = Sequential.Config(Linear.Config(64, 128)).make()
    assert len(seq) == 1
    assert seq[0].weight.shape == (128, 64)


def test_repeat():
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    assert len(seq) == 4


def test_depth_index_propagation():
    """Each repeated layer gets depth=i (0-indexed)."""
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    for i, layer in enumerate(seq):
        assert layer.depth == i, f"layer {i} depth={layer.depth}, expected {i}"


def test_depth_propagation_nested():
    """Depth propagates through inner Sequential to Linear."""
    block = Sequential.Config(Linear.Config(128, 128))
    seq = Sequential.Config(block, repeat=4).make()
    for i, inner in enumerate(seq):
        linear = inner[0]  # pyright: ignore[reportIndexIssue,reportUnknownVariableType]
        assert linear.depth == i, f"block {i} linear depth={linear.depth}, expected {i}"  # pyright: ignore[reportUnknownMemberType]


def test_depth_based_init():
    """Later layers should have smaller weight std due to depth scaling."""
    torch.manual_seed(0)
    seq = Sequential.Config(Linear.Config(128, 128), repeat=4).make()
    # depth=0 gets no scaling, depth=3 gets /sqrt(3)
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
        assert layer.depth == i


def test_transformer_stack():
    m = Sequential.Config(
        TransformerBlock.Config(
            channels_in=64,
            attn=SelfAttention.Config(heads=4, channels_head=16),
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
