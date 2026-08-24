"""Tests for patchify module."""

from __future__ import annotations

import pytest
import torch

from priml.model.patchify import Patchify, Unpatchify
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_patchify_2d():
    m = Patchify.Config(channels_in=3, patch_size=[4, 4]).make()
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2, 48, 8, 8)


def test_unpatchify_2d():
    m = Unpatchify.Config(channels_out=3, patch_size=[4, 4]).make()
    x = torch.randn(2, 48, 8, 8)
    out = m(x)
    assert out.shape == (2, 3, 32, 32)


def test_patchify_roundtrip():
    p = Patchify.Config(channels_in=3, patch_size=[4, 4]).make()
    u = Unpatchify.Config(channels_out=3, patch_size=[4, 4]).make()
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(u(p(x)), x)


def test_patchify_channels():
    cfg = Patchify.Config(channels_in=3, patch_size=[4, 4]).finalize()
    assert cfg.channels_out == 48

    ucfg = Unpatchify.Config(channels_out=3, patch_size=[4, 4]).finalize()
    assert ucfg.channels_in == 48


def test_patchify_forward_drops_extra_args():
    m = Patchify.Config(channels_in=3, patch_size=[2, 2]).make()
    x = torch.randn(2, 3, 8, 8)
    assert m(x, "extra", key="val").shape == (2, 12, 4, 4)


def test_patchify_rejects_degenerate_patch_size():
    """A non-positive or empty patch is rejected at construction, not at forward.

    ``patch_size=[0, 0]`` reached ``d // p`` and raised a bare
    ZeroDivisionError; ``[-2, -2]`` reached torch and complained about
    "invalid shape dimension -4", naming neither the config field nor the
    value the caller actually set.
    """
    for bad in ([0, 0], [-2, -2], [2, 0], []):
        with pytest.raises(ValueError, match="patch_size"):
            _ = Patchify.Config(channels_in=3, patch_size=bad).make()
        with pytest.raises(ValueError, match="patch_size"):
            _ = Unpatchify.Config(channels_out=3, patch_size=bad).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
