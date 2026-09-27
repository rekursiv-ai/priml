"""Tests for the HPS solver and its convolutional feed-forward block."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi1.experiments import exp004
from priml.baselines.arcagi1.model import HPSURM, ConvSwiGLU
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import DeepRecurrence
from priml.cost import cost
from priml.model.attention.self_attention import SelfAttention
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.cost import assert_cost_matches_torch


@pytest.mark.parametrize(
    ("recurrence", "core_passes"),
    [(None, 1), (DeepRecurrence.Config(slow_cycles=2, fast_cycles=6), 12)],
)
def test_hps_cost_counts_the_actual_inner_passes(
    recurrence: DeepRecurrence.Config | None,
    core_passes: int,
) -> None:
    """The one-state core runs its stack exactly once per fast cycle."""
    config = HPSURM.Config(channels_in=4, num_layers=1, vocab_size=3)
    config.embedding = GridEmbedding.Config(grid_shape=(2,))
    config.block = SwiGLU.Config(channels_hidden=8, round_to=1)
    config.recurrence = recurrence
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 3, (1, 2)),
        run=_logits_and_halt,
        batch_size=1,
        dtype=None,
    )
    unstacked = config.copy_tree()
    unstacked.num_layers = 0
    block = cost(
        config.copy_tree().finalize().block,
        seq_len=2,
        batch_size=1,
        dtype=None,
    )
    base = cost(unstacked.finalize(), batch_size=1, dtype=None)
    assert (
        analytical["flops", "primal", "matmul"].sum()
        - base["flops", "primal", "matmul"].sum()
        == core_passes * block["flops", "primal", "matmul"].sum()
    )


def _logits_and_halt(module: nn.Module, tokens: Tensor) -> Tensor:
    """Pull gradients through both prediction heads."""
    assert isinstance(module, HPSURM)
    result = module(tokens)
    return result.logits.sum() + result.halt.sum()


def test_conv_swiglu_reset_initializes_its_convolution() -> None:
    """Post-materialization reset must cover the extra convolution parameters."""
    ffn = ConvSwiGLU.Config(channels_in=4, channels_hidden=4).make()
    assert ffn.conv.bias is not None
    with torch.no_grad():
        ffn.conv.weight.fill_(float("nan"))
        ffn.conv.bias.fill_(float("nan"))
    ffn.reset_parameters()
    assert torch.isfinite(ffn.conv.weight).all()
    assert torch.equal(ffn.conv.bias, torch.zeros_like(ffn.conv.bias))


def test_conv_swiglu_rejects_tensor_parallelism_explicitly() -> None:
    ffn = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        shard="colwise",
    ).make()
    assert ffn.shard == "colwise"
    with pytest.raises(NotImplementedError, match="ConvSwiGLU"):
        ffn.tensor_parallel_style()


@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("shift_conv", [False, True])
def test_conv_swiglu_cost_matches_its_selected_path(
    shift_conv: bool,
    kernel_size: int,
) -> None:
    """Shifted taps are elementwise; Conv1d computes the padded extra row."""
    config = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        kernel_size=kernel_size,
        shift_conv=shift_conv,
    )
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(1, 3, 4, requires_grad=True),
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    base = cost(
        SwiGLU.Config(channels_in=4, channels_hidden=4).finalize(),
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    actual_matmul = analytical["flops", "matmul"].sum()
    base_matmul = base["flops", "matmul"].sum()
    if shift_conv:
        assert actual_matmul == base_matmul
    else:
        assert actual_matmul > base_matmul
    elementwise = (
        analytical["flops", "primal", "elementwise"].sum()
        - base["flops", "primal", "elementwise"].sum()
    )
    conv_rows = 3 + 2 * (kernel_size // 2) - kernel_size + 1
    expected = (2 * kernel_size + 5) * 3 * 4 if shift_conv else (conv_rows + 5 * 3) * 4
    assert elementwise == expected


@pytest.mark.parametrize("kernel_size", [5, 7])
def test_shift_convolution_keeps_short_sequence_length(kernel_size: int) -> None:
    """Shifts beyond the input are all padding, even for a one-token input."""
    shifted = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        kernel_size=kernel_size,
        shift_conv=True,
    ).make()
    reference = ConvSwiGLU.Config(
        channels_in=4,
        channels_hidden=4,
        kernel_size=kernel_size,
        shift_conv=False,
    ).make()
    reference.load_state_dict(shifted.state_dict())
    x = torch.randn(2, 1, 4)
    actual = shifted(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, reference(x))


def test_hps_recipe_finalizes_and_tiny_model_runs_without_parallelism() -> None:
    """The transformer may auto-fill the FFN shard slot for a local model."""
    assert isinstance(exp004().copy_tree().finalize().step.model, HPSURM.Config)
    config = HPSURM.Config(channels_in=4, num_layers=1, vocab_size=3)
    config.embedding = GridEmbedding.Config(grid_shape=(2,))
    config.block = TransformerBlock.Config(
        attn=SelfAttention.Config(num_heads=2, channels_head=2),
        ffn=ConvSwiGLU.Config(channels_hidden=4),
        prenorm=False,
    )
    model = config.make()
    block = model.reasoning[0]
    assert isinstance(block, TransformerBlock)
    ffn = block.ffn
    assert isinstance(ffn, ConvSwiGLU)
    assert ffn.shard == "colwise"
    assert model(torch.randint(0, 3, (1, 2))).logits.shape == (1, 2, 3)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
