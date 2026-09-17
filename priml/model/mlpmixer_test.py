"""Tests for mlpmixer module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.cost import cost
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


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


def test_mlp_mixer_block_config_pprint() -> None:
    config = MLPMixerBlock.Config(channels_in=4, seq_len=2)
    assert_pprint_golden(
        test_file=__file__,
        name="mlp_mixer_block",
        config=config,
    )


def test_mlp_mixer_block_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="mlp_mixer_block",
        build_module=lambda: MLPMixerBlock.Config(channels_in=4, seq_len=2).make(),
        build_input=lambda: torch.randn(2, 2, 4),
        seed=0,
    )


def test_mlp_mixer_block_cost_amortizes_the_token_mixer_over_tokens() -> None:
    """The token mixer maps ``seq_len`` rows, ``channels_in`` of them per image.

    A ``[seq_len=2, channels=4]`` input holds two tokens and four token-mixer
    rows, so per token the token mixer runs ``4 / 2 = 2`` times while the
    channel mixer runs once; torch's count agrees exactly.
    """
    config = MLPMixerBlock.Config(
        channels_in=4,
        seq_len=2,
        token_mixer=SwiGLU.Config(channels_hidden=3, round_to=1),
        channel_mixer=SwiGLU.Config(channels_hidden=5, round_to=1),
        norm_token=RMSNorm.Config(elementwise_affine=True),
    )
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(1, 2, 4, requires_grad=True),
        num_tokens=2,
    )
    finalized = config.copy_tree().finalize()
    over_tokens = cost(finalized.token_mixer, rows=2) + cost(
        finalized.norm_token,
        rows=2,
    )
    over_channels = cost(finalized.channel_mixer, rows=2) + cost(
        finalized.norm_channel,
        rows=2,
    )
    token = 2 * (2 * 3) + 3 * 2  # up_proj is twice the hidden width when gated.
    channel = 4 * (2 * 5) + 5 * 4
    assert model_cost.primal.flops.matmul == 2 * (2 * token + channel)
    assert model_cost.adjoint.flops.matmul == 4 * (2 * token + channel)
    assert model_cost.primal.flops.elementwise == (
        2 * over_tokens.primal.flops.elementwise
        + over_channels.primal.flops.elementwise
        + 2 * 4
    )
    assert model_cost.params == token + channel + 2


def test_mixer_cost_uses_transposed_row_sharing_for_traffic() -> None:
    config = MLPMixerBlock.Config()
    config.channels_in = 4
    config.seq_len = 2
    config.norm_token = RMSNorm.Config()
    config.norm_token.elementwise_affine = True
    finalized = config.finalize()
    result = finalized.cost(rows=6, itemsize=2)
    token = cost(finalized.token_mixer, rows=12, itemsize=2)
    channel = cost(finalized.channel_mixer, rows=6, itemsize=2)
    assert (
        result.primal.bytes.matmul
        == 2 * token.primal.bytes.matmul + channel.primal.bytes.matmul
    )
    assert (
        result.adjoint.bytes.matmul
        == 2 * token.adjoint.bytes.matmul + channel.adjoint.bytes.matmul
    )
    norm_token = cost(finalized.norm_token, rows=12, itemsize=2)
    norm_channel = cost(finalized.norm_channel, rows=6, itemsize=2)
    assert (
        result.adjoint.flops.reduction
        == 2 * norm_token.adjoint.flops.reduction + norm_channel.adjoint.flops.reduction
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
