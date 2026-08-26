"""Tests for mlpmixer module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from priml.model.mlpmixer import MLPMixerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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


def test_mlp_mixer_forward_accepts_messages_and_rejects_positional_extras():
    m = MLPMixerBlock.Config(channels_in=64, seq_len=8).make()
    x = torch.randn(2, 8, 64)
    assert m(x, key="val").shape == (2, 8, 64)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_mlp_mixer_block_config_pprint(request: pytest.FixtureRequest) -> None:
    config = MLPMixerBlock.Config(channels_in=4, seq_len=2)
    assert_text_golden(
        request,
        test_file=__file__,
        name="mlp_mixer_block",
        rendered=config.pformat(hide_default_values=False),
    )


def test_mlp_mixer_block_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="mlp_mixer_block",
        build_module=lambda: MLPMixerBlock.Config(channels_in=4, seq_len=2).make(),
        build_input=lambda: torch.randn(2, 2, 4),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
