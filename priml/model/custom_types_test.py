"""Tests for model channel-attribute Protocols and propagation helper."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from priml.model.attention import SelfAttention
from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


def test_propagate_settable_field():
    cfg = SwiGLU.Config()
    propagate_attr(cfg, "channels_in", 128, protocol=ChannelsIn)
    assert cfg.channels_in == 128


def test_propagate_readonly_property_is_noop():
    """``channels_out`` is a derived read-only property; skip silently."""
    cfg = SelfAttention.Config(channels_in=64)
    propagate_attr(cfg, "channels_out", 999)
    assert cfg.channels_out == 64


def test_propagate_non_participant_skipped():
    """A child not implementing the gating Protocol opts out silently."""
    cfg = RMSNorm.Config(channels_in=32)
    # RMSNorm lacks channels_out and does not satisfy ChannelsOut -> skipped.
    assert not isinstance(cfg, ChannelsOut)
    propagate_attr(cfg, "channels_out", 999, protocol=ChannelsOut)
    propagate_attr(cfg, "depth", 7)  # no Protocol governs depth -> skipped
    assert not hasattr(cfg, "channels_out")


def test_propagate_missing_attr_raises():
    """A typo / missing attribute under a Protocol must raise."""

    @dataclass(slots=True, kw_only=True)
    class NoChannels:
        channels_in: int = -1

    with pytest.raises(AttributeError, match="channels_out"):
        propagate_attr(NoChannels(), "channels_out", 64, protocol=ChannelsIn)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
