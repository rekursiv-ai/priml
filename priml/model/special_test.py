"""Tests for special module."""

from __future__ import annotations

import pytest
import torch

from priml.model.linear import Linear
from priml.model.special import Identity, Skip
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_identity():
    m = Identity.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert m(x, "extra", key="val").shape == (2, 8, 64)


def test_identity_channels_infer():
    cfg = Identity.Config(channels_in=64).finalize()
    assert cfg.channels_out == 64

    cfg2 = Identity.Config(channels_out=32).finalize()
    assert cfg2.channels_in == 32


def test_identity_reset():
    m = Identity.Config(channels_in=64).make()
    m.reset_parameters()


def test_skip():
    m = Skip.Config(inner=Linear.Config(64, 64)).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_skip_reset():
    m = Skip.Config(inner=Linear.Config(64, 64)).make()
    m.reset_parameters()


def test_skip_requires_inner():
    with pytest.raises(ValueError, match="inner"):
        Skip.Config().make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
