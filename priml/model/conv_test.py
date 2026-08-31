"""Tests for conv module."""

from __future__ import annotations

from pathlib import Path

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.conv import Conv1d, Conv2d, Conv3d
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_conv1d_config_pprint() -> None:
    config = Conv1d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv1d",
        config=config,
    )


def test_conv1d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="conv1d",
        build_module=lambda: Conv1d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 4),
        seed=0,
    )


def test_conv2d_config_pprint() -> None:
    config = Conv2d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv2d",
        config=config,
    )


def test_conv2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="conv2d",
        build_module=lambda: Conv2d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 3, 3),
        seed=0,
    )


def test_conv3d_config_pprint() -> None:
    config = Conv3d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv3d",
        config=config,
    )


def test_conv3d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="conv3d",
        build_module=lambda: Conv3d.Config(2, 3).make(),
        build_input=lambda: torch.randn(1, 2, 3, 3, 3),
        seed=0,
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


def test_conv_forward_accepts_messages_and_rejects_positional_extras():
    m = Conv2d.Config(3, 16, kernel_size=3, padding=1).make()
    x = torch.randn(2, 3, 32, 32)
    assert m(x, key="val").shape == (2, 16, 32, 32)
    with pytest.raises(TypeError):
        m(x, "extra")


def test_conv_reset():
    for cls in (Conv1d, Conv2d, Conv3d):
        m = cls.Config(3, 16).make()
        m.reset_parameters()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
