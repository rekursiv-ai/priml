"""Tests for model channel-attribute Protocols and propagation helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import (
    ChannelsHead,
    ChannelsIn,
    ChannelsOut,
    NumHeads,
    flatten_depth_index,
    propagate_attr,
)
from priml.model.moe import Router
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_custom_types_public_contract(request: pytest.FixtureRequest) -> None:
    rendered = "\n".join(
        [
            "flatten_depth_index:",
            f"  unspecified: {flatten_depth_index(())}",
            f"  one level ((3, 12),): {flatten_depth_index(((3, 12),))}",
            f"  nested ((1, 4), (3, 12)): {flatten_depth_index(((1, 4), (3, 12)))}",
        ]
    )
    assert_text_golden(
        request,
        test_file=__file__,
        name="custom_types",
        rendered=rendered,
    )


def test_custom_types_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="custom_types",
        build_module=_FlattenDepthIndex,
        build_input=lambda: torch.tensor(
            [
                [[0, 2], [1, 3]],
                [[1, 2], [2, 3]],
            ]
        ),
        seed=0,
    )


def test_head_capabilities_are_direct_attributes() -> None:
    @dataclass(slots=True, kw_only=True)
    class Attention:
        num_heads: int
        channels_head: int

    attention = Attention(num_heads=4, channels_head=32)

    assert isinstance(attention, NumHeads)
    assert isinstance(attention, ChannelsHead)
    assert attention.num_heads * attention.channels_head == 128


def test_flatten_depth_index_uses_global_to_local_mixed_radix() -> None:
    assert flatten_depth_index(()) == -1
    assert flatten_depth_index(((3, 12),)) == 3
    assert flatten_depth_index(((1, 4), (3, 12))) == 15


@pytest.mark.parametrize("depth_index", [((-1, 4),), ((4, 4),), ((0, 0),)])
def test_flatten_depth_index_rejects_invalid_levels(
    depth_index: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError, match="depth_index"):
        flatten_depth_index(depth_index)


def test_propagate_settable_field():
    cfg = SwiGLU.Config()
    propagate_attr(cfg, "channels_in", 128, protocol=ChannelsIn)
    assert cfg.channels_in == 128


def test_propagate_width_preserving_field_is_mutable():
    cfg = SelfAttention.Config(channels_in=64)
    propagate_attr(cfg, "channels_out", 999)
    assert cfg.channels_out == 999
    with pytest.raises(ValueError, match="channels_in=64 must equal channels_out=999"):
        cfg.make()


def test_propagate_non_participant_skipped():
    """A child not implementing the gating Protocol opts out silently.

    ``Router`` is the stand-in because it genuinely emits no width: it returns
    ``(weights, indices, logits)``, so it declares no ``channels_out`` for a
    parent to push into. A norm was used here once and stopped being a
    non-participant the moment norms gained the derived property.
    """
    cfg = Router.Config(channels_in=32)
    assert not isinstance(cfg, ChannelsOut)
    propagate_attr(cfg, "channels_out", 999, protocol=ChannelsOut)
    propagate_attr(cfg, "depth_index", ((0, 1),))
    assert not hasattr(cfg, "channels_out")


def test_propagate_norm_width_is_mutable_but_checked_at_construction():
    cfg = RMSNorm.Config(channels_in=32)
    assert isinstance(cfg, ChannelsOut)
    propagate_attr(cfg, "channels_out", 999, protocol=ChannelsOut)
    assert cfg.channels_out == 999
    with pytest.raises(ValueError, match="channels_in=32 must equal channels_out=999"):
        cfg.make()


def test_propagate_missing_attr_raises():
    """A typo / missing attribute under a Protocol must raise."""

    @dataclass(slots=True, kw_only=True)
    class NoChannels:
        channels_in: int = -1

    with pytest.raises(AttributeError, match="channels_out"):
        propagate_attr(NoChannels(), "channels_out", 64, protocol=ChannelsIn)


class _FlattenDepthIndex(nn.Module):
    @override
    def forward(self, depth_indices: Tensor) -> Tensor:
        nested = cast(list[list[list[int]]], depth_indices.tolist())
        return torch.tensor(
            [
                flatten_depth_index(tuple((index, count) for index, count in levels))
                for levels in nested
            ]
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
