"""Tests for the convolutional feed-forward and the single-latent recurrence."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi1.model import (
    ConvSwiGLU,
    UrmRecurrence,
    depthwise_conv,
    depthwise_shift,
)
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import SudokuNet
from priml.cost import cost
from priml.model.norm import RMSNorm
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
def test_urm_carries_one_latent_and_leaves_the_other_untouched(
    inner_grad_loops: int,
) -> None:
    config = SudokuNet.Config(channels_in=8, num_layers=1, vocab_size=5)
    config.embedding = GridEmbedding.Config(grid_shape=(4,))
    config.block = SwiGLU.Config(channels_hidden=8, round_to=1)
    config.recurrence = UrmRecurrence.Config(
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

    recurrence = UrmRecurrence.Config(fast_cycles=4, inner_grad_loops=1).make()
    x = torch.ones(2, 3, 4, requires_grad=True)
    recurrence.refine(mix, x, x, x, None)
    assert seen == [False, False, False, True]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
