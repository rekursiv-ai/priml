"""Tests for patchify module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.patchify import Patchify, Unpatchify
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_patchify_config_pprint() -> None:
    config = Patchify.Config(channels_in=2, patch_size=[2, 2])
    assert_pprint_golden(
        test_file=__file__,
        name="patchify",
        config=config,
    )


def test_unpatchify_config_pprint() -> None:
    config = Unpatchify.Config(channels_out=2, patch_size=[2, 2])
    assert_pprint_golden(
        test_file=__file__,
        name="unpatchify",
        config=config,
    )


def test_patchify_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="patchify",
        build_module=lambda: Patchify.Config(
            channels_in=2,
            patch_size=[2, 2],
        ).make(),
        build_input=lambda: torch.randn(1, 2, 4, 4),
        seed=0,
    )


def test_unpatchify_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="unpatchify",
        build_module=lambda: Unpatchify.Config(
            channels_out=2,
            patch_size=[2, 2],
        ).make(),
        build_input=lambda: torch.randn(1, 8, 2, 2),
        seed=0,
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
    reverse = Patchify.Config(channels_out=48, patch_size=[4, 4]).finalize()
    assert reverse.channels_in == 3

    ucfg = Unpatchify.Config(channels_out=3, patch_size=[4, 4]).finalize()
    assert ucfg.channels_in == 48
    reverse_unpatch = Unpatchify.Config(
        channels_in=48,
        patch_size=[4, 4],
    ).finalize()
    assert reverse_unpatch.channels_out == 3


def test_patchify_rejects_inconsistent_channel_boundaries() -> None:
    with pytest.raises(ValueError, match="channels_out=47 must equal"):
        Patchify.Config(
            channels_in=3,
            channels_out=47,
            patch_size=[4, 4],
        ).finalize()
    with pytest.raises(ValueError, match="channels_in=47 must equal"):
        Unpatchify.Config(
            channels_in=47,
            channels_out=3,
            patch_size=[4, 4],
        ).finalize()


def test_patchify_forward_accepts_messages_and_rejects_positional_extras():
    m = Patchify.Config(channels_in=3, patch_size=[2, 2]).make()
    x = torch.randn(2, 3, 8, 8)
    assert m(x, key="val").shape == (2, 12, 4, 4)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


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
