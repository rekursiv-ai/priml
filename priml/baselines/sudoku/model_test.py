"""Tests for the sudoku model and its recurrence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import (
    DeepRecurrence,
    ForwardOutput,
    SudokuNet,
)
from priml.baselines.sudoku.prefix import RegisterTokens
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden


_GOLDEN_DIR = Path(__file__).parent / "goldens"


def _config(*, recurrent: bool = False, mixer: bool = False) -> SudokuNet.Config:
    config = SudokuNet.Config(hidden_size=16, num_layers=1)
    config.embedding = GridEmbedding.Config()
    # The FFN width has to be shrunk EXPLICITLY. ``SwiGLU`` infers it as
    # ``channels_in * 8/3`` rounded up to ``round_to``, which defaults to 256 --
    # so a 16-channel model still builds a 512-wide gated hidden layer, and one
    # tensor ends up thirty times the size of everything else in the golden.
    config.block = TransformerBlock.Config(
        prenorm=False,
        ffn=SwiGLU.Config(round_to=16),
    )
    if mixer:
        config.block = MLPMixerBlock.Config(
            seq_len=81,
            prenorm=False,
            token_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
            channel_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
        )
    if recurrent:
        config.recurrence = DeepRecurrence.Config(slow_cycles=2, fast_cycles=2)
    return config


def _model(**kwargs: bool) -> SudokuNet:
    torch.manual_seed(0)
    return _config(**kwargs).make()


@pytest.mark.parametrize("mixer", [False, True])
@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.compute_large_fixture
def test_every_corner_of_the_lattice_runs(mixer: bool, recurrent: bool) -> None:
    """Architecture and recurrence vary independently, as config values."""
    model = _model(mixer=mixer, recurrent=recurrent)
    out = model(torch.randint(0, 11, (2, 81)))
    assert out.logits.shape == (2, 81, 11)
    assert out.halt.shape == (2,)


def test_recurrence_adds_no_parameters() -> None:
    """The recurrence schedules the core; it does not own weights.

    If it did, the plain-vs-recurrent comparison would confound depth with
    capacity and neither result would mean what it claims.
    """
    plain = sum(p.numel() for p in _model().parameters())
    recurrent = sum(p.numel() for p in _model(recurrent=True).parameters())
    assert plain == recurrent


@pytest.mark.compute_training
def test_backward_graph_is_flat_in_recurrence_depth() -> None:
    """Gradient cost must not grow with cycle count.

    That is the whole reason deep recurrence is affordable: all but the last
    cycle run under ``no_grad``, so a 32-cycle forward backpropagates through
    one cycle. A graph that grew with depth would mean the truncation broke.
    """
    assert _graph_size(1) == _graph_size(8) == _graph_size(32)


def test_prenorm_diverges_under_recurrence() -> None:
    """Post-norm is a requirement of feeding a block its own output.

    Pre-norm leaves the residual stream unnormalized, which is harmless in one
    pass and compounds when the output is fed back. This pins the reason the
    default is post-norm, so a future change to priml's default cannot silently
    reintroduce the divergence.
    """
    assert _carried_magnitude(prenorm=False) < _carried_magnitude(prenorm=True) / 10


def test_latents_carry_between_calls() -> None:
    """A second call from the first call's latents differs from a fresh one."""
    model = _model(recurrent=True)
    tokens = torch.randint(0, 11, (2, 81))
    first = model(tokens)
    carried = model(tokens, first.z_slow, first.z_fast)
    fresh = model(tokens)
    assert not torch.equal(carried.logits, fresh.logits)


def test_intermediates_are_one_per_cycle() -> None:
    model = _model(recurrent=True)
    out = model(torch.randint(0, 11, (2, 81)), collect_intermediates=True)
    assert len(out.all_logits) == 2  # slow_cycles


def test_sequence_length_counts_the_prefix_before_finalize() -> None:
    """A parent reading ``total_seq_len`` must not see the sentinel.

    A parent's ``finalize`` runs BEFORE this config's, so a naive
    ``num_prefix_tokens + grid_len`` returns one short by the whole prefix
    while the sentinel is still set. Anything sized from it -- an ACT pool's
    latent buffers -- is then built to the wrong shape and fails only later,
    deep in a matmul. Measured: 80 against the true 81.
    """
    config = _config()
    registers = RegisterTokens.Config(num_tokens=4)
    config.prefix = registers
    assert config.num_prefix_tokens == -1  # not yet finalized
    assert config.total_seq_len == 81 + 4
    # After finalize the count is materialized and agrees.
    final = config.copy_tree().finalize()
    assert final.num_prefix_tokens == 4
    assert final.total_seq_len == 81 + 4


def test_prefix_tokens_reach_the_sequence() -> None:
    """Prefix logits are stripped, so the output is still one row per cell."""
    config = _config()
    config.prefix = RegisterTokens.Config(num_tokens=3)
    torch.manual_seed(0)
    out = config.make()(torch.randint(0, 11, (2, 81)))
    assert out.logits.shape == (2, 81, 11)


def test_a_prefix_without_a_token_count_is_rejected() -> None:
    """Guessing 0 would silently shift every grid position."""
    config = _config()
    config.prefix = RMSNorm.Config(channels_in=16)  # not a prefix module
    with pytest.raises(ValueError, match="declares no"):
        config.copy_tree().finalize()


def test_cycle_counts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        DeepRecurrence.Config(slow_cycles=0).make()


def test_plain_forward_bfb() -> None:
    """Freeze exp000's architecture: same weights in, same logits out.

    Guards the whole forward path -- embedding composition, block stack,
    output head -- against a refactor that changes arithmetic while keeping
    every shape and name intact.
    """
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="plain_min_cpu",
        build_module=lambda: _config().make(),
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        seed=0,
        run=_logits,
    )


def test_recurrent_forward_bfb() -> None:
    """Freeze the recurrence: the cycle schedule is part of the arithmetic.

    Separate from the plain golden because the two differ only in a slot's
    value, so a change to the recurrence would leave the plain golden green.
    """
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="recurrent_min_cpu",
        build_module=lambda: _config(recurrent=True).make(),
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        seed=0,
        run=_logits,
    )


def _logits(module: nn.Module, tokens: Any) -> Tensor:
    """Run the model and return the logits the golden compares."""
    out = module(tokens)
    assert isinstance(out, ForwardOutput)
    return out.logits


def _graph_size(slow_cycles: int) -> int:
    """Count the autograd nodes reachable from one forward's loss."""
    config = _config(recurrent=True)
    assert isinstance(config.recurrence, DeepRecurrence.Config)
    config.recurrence.slow_cycles = slow_cycles
    torch.manual_seed(0)
    out = config.make()(torch.randint(0, 11, (2, 81)))
    loss = out.logits.square().mean()
    # Hold a reference to every node: ``next_functions`` yields fresh wrappers,
    # so a set of ids alone undercounts once one is collected.
    keep: list[Any] = []
    seen: set[Any] = set()
    stack: list[Any] = [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        keep.append(node)
        # Autograd nodes are untyped at the Python boundary.
        stack.extend(nxt for nxt, _ in node.next_functions)
    return len(keep)


def _carried_magnitude(*, prenorm: bool) -> float:
    """Largest carried-latent value after three recurrent steps."""
    config = _config(recurrent=True)
    assert isinstance(config.recurrence, DeepRecurrence.Config)
    config.recurrence.slow_cycles = 4
    config.block = TransformerBlock.Config(prenorm=prenorm)
    torch.manual_seed(0)
    model = config.make()
    tokens = torch.randint(2, 11, (2, 81))
    z_slow, z_fast = model.init_latents(2)
    for _ in range(3):
        out = model(tokens, z_slow, z_fast)
        z_slow, z_fast = out.z_slow, out.z_fast
    return float(z_slow.abs().max())


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
