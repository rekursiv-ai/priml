"""Tests for conv module."""

from __future__ import annotations

import torch

from priml.model.conv import Conv1d, Conv2d, Conv3d
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_conv1d():
    m = Conv1d.Config(3, 16).make()
    x = torch.randn(2, 3, 32)
    assert m(x).shape == (2, 16, 32)


def test_conv1d_channels_infer():
    cfg = Conv1d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv2d():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 16, 32, 32)


def test_conv2d_channels_infer():
    cfg = Conv2d.Config(channels_out=16).finalize()
    assert cfg.channels_in == 16


def test_conv3d():
    m = Conv3d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 8, 8, 8)
    assert m(x).shape == (2, 16, 8, 8, 8)


def test_conv3d_channels_infer():
    cfg = Conv3d.Config(channels_in=3).finalize()
    assert cfg.channels_out == 3


def test_conv_forward_drops_extra_args():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 32, 32)
    assert m(x, "extra").shape == (2, 16, 32, 32)


def test_conv_reset():
    for cls in (Conv1d, Conv2d, Conv3d):
        m = cls.Config(3, 16).make()
        m.reset_parameters()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
