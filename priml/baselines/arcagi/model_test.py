"""Tests for the convolutional feed-forward and the single-latent recurrence."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    PuzzleEmbedding,
    RotaryBlock,
    depthwise_conv,
    depthwise_shift,
)
from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import GridSolver
from priml.model.norm import RMSNorm
from priml.model.recurrence import SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.testing.cost import assert_cost_matches_torch


def test_conv_swiglu_reset_initializes_its_convolution() -> None:
    """A reset after meta materialization must cover the convolution too."""
    ffn = ConvSwiGLU.Config(channels_in=4, channels_hidden=4).make()
    assert ffn.conv.bias is not None
    with torch.no_grad():
        ffn.conv.weight.fill_(float("nan"))
        ffn.conv.bias.fill_(float("nan"))
    ffn.reset_parameters()
    assert torch.isfinite(ffn.conv.weight).all()
    assert torch.isfinite(ffn.conv.bias).all()


@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("short_conv", [depthwise_conv, depthwise_shift])
def test_conv_swiglu_cost_matches_torch(
    short_conv: object,
    kernel_size: int,
) -> None:
    """Shifted taps cost elementwise work; the convolution adds matmul work."""
    config = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        kernel_size=kernel_size,
        short_conv=depthwise_shift if short_conv is depthwise_shift else depthwise_conv,
    )
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 4, requires_grad=True),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )
    base = cost(
        SwiGLU.Config(channels_in=4, channels_hidden=4).finalize(),
        seq_len=3,
        batch_size=2,
        dtype=None,
    )
    extra = analytical["flops", "matmul"].sum() - base["flops", "matmul"].sum()
    assert (extra == 0) is (short_conv is depthwise_shift)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kernel_size", [2, 3, 5])
@pytest.mark.parametrize("length", [1, 4])
def test_shifted_taps_match_the_convolution(
    causal: bool,
    kernel_size: int,
    length: int,
) -> None:
    """Both windows keep the sequence length, even past a one-token input."""
    conv = nn.Conv1d(4, 4, kernel_size, groups=4, bias=True).double()
    x = torch.randn(2, length, 4, dtype=torch.float64)
    shifted = depthwise_shift(conv, x, causal=causal)
    reference = depthwise_conv(conv, x, causal=causal)
    assert shifted.shape == x.shape
    torch.testing.assert_close(shifted, reference)


@pytest.mark.parametrize("causal", [False, True])
def test_depthwise_conv_casts_bfloat_weights_and_restores_input_dtype(
    causal: bool,
) -> None:
    """Use weight precision and return the input precision."""
    conv = nn.Conv1d(4, out_channels=4, kernel_size=3, groups=4, bias=True).to(
        torch.bfloat16,
    )
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 7
    channel_first = x.transpose(1, 2)
    if causal:
        padded = nn.functional.pad(channel_first, (2, 0))
    else:
        padded = nn.functional.pad(channel_first, (1, 1))
    expected = (
        nn.functional.silu(
            conv(padded.to(conv.weight.dtype))[..., : x.shape[1]],
        )
        .transpose(1, 2)
        .to(x.dtype)
    )

    actual = depthwise_conv(conv, x, causal=causal)

    assert actual.dtype is torch.float32
    assert actual.shape == x.shape
    assert torch.equal(actual, expected)


def test_the_gate_norm_changes_the_output() -> None:
    """With a norm the gate is ``sigmoid(g) * norm(g * u)``, not ``silu(g) * u``."""
    plain = ConvSwiGLU.Config(channels_in=4, channels_hidden=4).make()
    normed = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        norm=RMSNorm.Config(),
    ).make()
    normed.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 3, 4)
    assert not torch.equal(plain(x), normed(x))


@pytest.mark.parametrize("inner_grad_loops", [0, 2])
def test_single_latent_recurrence_leaves_the_other_latent_untouched(
    inner_grad_loops: int,
) -> None:
    config = GridSolver.Config(channels_in=8, num_layers=1, vocab_size=5)
    config.embedding = GridEmbedding.Config(grid_shape=(4,))
    config.block = SwiGLU.Config(channels_hidden=8, round_to=1)
    config.recurrence = SingleLatentRecurrence.Config(
        slow_cycles=1,
        fast_cycles=3,
        inner_grad_loops=inner_grad_loops,
    )
    model = config.make()
    tokens = torch.randint(0, 5, (2, 4))
    z_slow, z_fast = model.init_latents(2)
    out = model(tokens, z_slow, z_fast)
    assert not torch.equal(out.z_slow, z_slow)
    assert torch.equal(out.z_fast, z_fast)
    (out.logits.sum() + out.halt.sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_inner_grad_loops_truncate_the_backward() -> None:
    """Passes before the last ``inner_grad_loops`` carry no gradient."""
    seen: list[bool] = []

    def mix(z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        del cos_sin
        seen.append(torch.is_grad_enabled())
        return z * 2.0

    recurrence = SingleLatentRecurrence.Config(fast_cycles=4, inner_grad_loops=1).make()
    x = torch.ones(2, 3, 4, requires_grad=True)
    recurrence.refine(mix, x, x, x, None)
    assert seen == [False, False, False, True]


def test_rotary_block_without_rope_uses_the_callers_cos_sin() -> None:
    config = RotaryBlock.Config(channels_in=4)
    assert isinstance(config.attn, Attention.Config)
    config.attn.num_heads = 2
    config.attn.channels_head = 2
    assert isinstance(config.ffn, SwiGLU.Config)
    config.ffn.round_to = 1
    assert config.rope is not None
    config.rope.channels_head = 2
    owned = config.make()
    bare_config = config.copy_tree()
    bare_config.rope = None
    bare = bare_config.make()
    bare.load_state_dict(owned.state_dict())
    x = torch.randn(2, 3, 4)
    cos_sin = owned.factors(3, device=x.device)
    assert cos_sin is not None
    assert bare.factors(3, device=x.device) is None
    with pytest.raises(ValueError, match="needs cos_sin"):
        bare(x)
    assert torch.equal(bare(x, cos_sin=cos_sin), owned(x))


def test_conv_swiglu_infers_channels_in_from_channels_out() -> None:
    config = ConvSwiGLU.Config(channels_out=4, channels_hidden=6).finalize()
    assert config.channels_in == 4
    assert config.channels_out == 4


def test_conv_swiglu_rounds_the_inferred_hidden_width_up_to_the_grid() -> None:
    config = ConvSwiGLU.Config(channels_in=4, round_to=5).finalize()
    # 4 * 8/3 = 10.67 rounds up to the next multiple of 5.
    assert config.channels_hidden == 15


@pytest.mark.parametrize("training", [True, False])
def test_puzzle_embedding_returns_float32_rows_rounded_through_bfloat16(
    training: bool,
) -> None:
    config = PuzzleEmbedding.Config(num_puzzles=5, num_tokens=2, batch_size=3)
    config.channels_out = 4
    config.init_std = 1.0
    torch.manual_seed(0)
    module = config.make()
    module.train(training)
    identifiers = torch.tensor([4, 1, 4], dtype=torch.int32)
    out = module(3, puzzle_identifiers=identifiers)
    rows = module.weights[identifiers.long()]
    expected = rows.to(torch.bfloat16).float()
    assert out.dtype is torch.float32
    assert not torch.equal(rows, expected)
    assert torch.equal(out[:, 0], expected * module.embed_scale)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
