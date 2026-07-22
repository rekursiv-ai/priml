"""Tests for linear module."""

from __future__ import annotations

import torch

from priml.model.linear import EnsembleLinear, Linear
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_linear():
    m = Linear.Config(32, 64).make()
    x = torch.randn(2, 8, 32)
    assert m(x).shape == (2, 8, 64)
    assert m.bias is None


def test_linear_bias():
    m = Linear.Config(32, 64, bias=True).make()
    assert m.bias is not None


def test_linear_channels_infer_from_out():
    m = Linear.Config(channels_out=64).make()
    assert m.in_features == 64


def test_linear_channels_infer_from_in():
    m = Linear.Config(channels_in=64).make()
    assert m.out_features == 64


def test_linear_forward_drops_extra_args():
    m = Linear.Config(32, 32).make()
    x = torch.randn(2, 8, 32)
    assert m(x, "extra", key="val").shape == (2, 8, 32)


def test_linear_reset_parameters():
    m = Linear.Config(32, 32).make()
    m.reset_parameters()


def test_linear_arbitrary_batch_dims():
    """Linear supports arbitrary leading batch dims."""
    m = Linear.Config(16, 32).make()
    x = torch.randn(2, 3, 4, 8, 16)
    assert m(x).shape == (2, 3, 4, 8, 32)


def test_ensemble_linear():
    m = EnsembleLinear.Config(channels_in=64, channels_out=16, num_ensemble=4).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 4, 16)


def test_ensemble_linear_with_bias():
    m = EnsembleLinear.Config(
        channels_in=64,
        channels_out=16,
        num_ensemble=4,
        bias=True,
    ).make()
    assert m.bias is not None
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 4, 16)


def test_ensemble_linear_reset():
    m = EnsembleLinear.Config(channels_in=64, channels_out=16, num_ensemble=4).make()
    m.reset_parameters()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
