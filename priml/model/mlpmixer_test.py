"""Tests for mlpmixer module."""

from __future__ import annotations

import torch

from priml.model.mlpmixer import MLPMixerBlock
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_mlp_mixer_block_prenorm():
    m = MLPMixerBlock.Config(channels_in=64, seq_len=8, prenorm=True).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_mlp_mixer_block_postnorm():
    m = MLPMixerBlock.Config(channels_in=64, seq_len=8, prenorm=False).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_mlp_mixer_block_reset():
    m = MLPMixerBlock.Config(channels_in=64, seq_len=8).make()
    m.reset_parameters()


def test_mlp_mixer_forward_drops_extra_args():
    m = MLPMixerBlock.Config(channels_in=64, seq_len=8).make()
    x = torch.randn(2, 8, 64)
    assert m(x, "extra", key="val").shape == (2, 8, 64)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
