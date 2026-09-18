"""Tests for conv module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.conv import Conv1d, Conv2d, Conv3d
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_conv1d_config_pprint() -> None:
    config = Conv1d.Config(2, 3)
    assert_pprint_golden(
        test_file=__file__,
        name="conv1d",
        config=config,
    )


def test_conv1d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
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
        golden_dir=_CWD / "testdata",
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
        golden_dir=_CWD / "testdata",
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


def test_conv2d_cost_is_a_matmul_over_the_receptive_field() -> None:
    """Per output position: ``channels_in * prod(kernel_size)`` -> ``channels_out``.

    ``padding="same"`` keeps every input position as an output position, so
    a ``4 x 6`` image is 24 tokens.
    """
    analytical = assert_cost_matches_torch(
        Conv2d.Config(2, 3, kernel_size=(3, 5), bias=True),
        build_input=lambda: torch.randn(1, 2, 4, 6, requires_grad=True),
        seq_len=4 * 6,
        batch_size=1,
        num_tokens=4 * 6,
        check_bytes=False,  # TODO(Issue#20739): conv/attention traffic convention.
        dtype=None,
    )
    weights = 3 * 2 * 15
    assert analytical["flops", "primal", "matmul"].sum() == 2 * weights
    assert analytical["flops", "adjoint", "matmul"].sum() == 4 * weights
    assert analytical.params == weights + 3
    assert analytical["bytes", "primal", "elementwise"].sum() == 4 * (2 * 3 + 3 / 24)


def test_conv1d_cost_divides_the_fan_in_by_groups() -> None:
    """Grouped: each output sees ``channels_in / groups`` inputs, forward and back.

    torch's ``convolution_backward`` formula ignores ``groups`` (it reads the
    full ``c_in`` off the weight shape), so it over-counts the backward by the
    group factor: measured 504 forward, 1512 backward for a 2-group conv whose
    true backward is 1008. Forward agrees; the total is held to 3/4 of torch's.
    """
    analytical = assert_cost_matches_torch(
        Conv1d.Config(4, 6, kernel_size=3, groups=2),
        build_input=lambda: torch.randn(1, 4, 7, requires_grad=True),
        seq_len=7,
        batch_size=1,
        num_tokens=7,
        check_bytes=False,  # TODO(Issue#20739): conv/attention traffic convention.
        dtype=None,
        expected_ratio=0.75,
    )
    weights = 6 * (4 // 2) * 3
    assert analytical["flops", "primal", "matmul"].sum() == 2 * weights
    assert analytical.params == weights
    assert analytical["bytes", "primal", "matmul"].sum() == 4 * (
        4 * 3 + 6 + weights / 7
    )


def test_conv3d_cost_cubes_a_scalar_kernel() -> None:
    analytical = assert_cost_matches_torch(
        Conv3d.Config(2, 3, kernel_size=3),
        build_input=lambda: torch.randn(1, 2, 3, 4, 5, requires_grad=True),
        seq_len=3 * 4 * 5,
        batch_size=1,
        num_tokens=3 * 4 * 5,
        check_bytes=False,  # TODO(Issue#20739): conv/attention traffic convention.
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() == 2 * 3 * 2 * 27


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
