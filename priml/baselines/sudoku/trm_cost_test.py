"""TRM and SudokuVerifier configs report the cost torch measures."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch

from priml.baselines.sudoku.eval import SudokuVerifier
from priml.baselines.sudoku.trm import TRM
from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.testing.cost import assert_cost_matches_torch


def test_unfilled_trm_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocabulary and grid shape"):
        TRM.Config().make()


def test_trm_cost_matches_torch() -> None:
    config = TRM.Config(
        vocab_size=11,
        puzzle_grid_shape=(81,),
        pos2d_grid_shape=(9, 9),
        pos2d_box_shape=(3, 3),
        channels_in=16,
        num_layers=1,
        num_heads=2,
        slow_cycles=2,
        fast_cycles=2,
        num_puzzle_identifiers=4,
        puzzle_emb_len=2,
        puzzle_emb_batch_size=2,
        compile=False,
        dtype=None,
    ).finalize()
    assert config.block is not None
    assert isinstance(config.block.attn, Attention.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()

    def run(module: nn.Module, tokens: Tensor) -> Tensor:
        assert isinstance(module, TRM)
        z_slow, z_fast = module.init_z(2)
        out = module(tokens, z_slow, z_fast, torch.tensor([1, 2]), feedback_ids=tokens)
        logits, q_halt = out["logits"], out["q_halt"]
        assert isinstance(logits, Tensor)
        assert isinstance(q_halt, Tensor)
        return logits.sum() + q_halt.sum()

    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        batch_size=2,
        dtype=None,
        run=run,
    )


def test_verifier_cost_matches_torch() -> None:
    """Torch counts attention only through the math kernel, so pin it.

    A fused SDPA kernel is invisible to ``FlopCounterMode``, and which kernel
    CPU SDPA picks varies by host and by what ran before in the process.
    """
    config = SudokuVerifier.Config(width=16, depth=1, heads=2).finalize()
    analytical = cost(config, batch_size=2, dtype=None)
    torch.manual_seed(0)
    module = config.make()
    module.train()
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        module(
            torch.randint(1, 11, (2, 81)),
            torch.randint(2, 11, (2, 81)),
        ).sum().backward()
    assert analytical.params == sum(p.numel() for p in module.parameters())
    assert analytical["flops", "matmul"].sum() == counter.get_total_flops()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
