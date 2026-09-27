"""TRM and SudokuVerifier configs report the cost torch measures."""

from __future__ import annotations

from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

import torch

from priml.baselines.sudoku.eval import SudokuVerifier
from priml.baselines.sudoku.trm import TRM
from priml.cost import cost
from priml.model.attention.kernel import SdpaNaive, attention_kernel_cost
from priml.model.attention.self_attention import SelfAttention
from priml.testing.cost import assert_cost_matches_torch


def _tiny_trm() -> TRM.Config:
    config = TRM.Config(
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
    )
    config = config.finalize()
    assert config.block is not None
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()
    return config


def _run_trm(module: nn.Module, tokens: Tensor) -> Tensor:
    assert isinstance(module, TRM)
    z_slow, z_fast = module.init_z(2)
    out = module(
        tokens,
        z_slow,
        z_fast,
        torch.tensor([1, 2]),
        feedback_ids=tokens,
    )
    logits, q_halt = out["logits"], out["q_halt"]
    assert isinstance(logits, Tensor)
    assert isinstance(q_halt, Tensor)
    return logits.sum() + q_halt.sum()


def test_trm_cost_matches_torch() -> None:
    assert_cost_matches_torch(
        _tiny_trm(),
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        batch_size=2,
        dtype=None,
        run=_run_trm,
    )


def test_verifier_cost_matches_torch_outside_attention() -> None:
    """Torch's counter skips the masked CPU SDPA call, so compare the rest."""
    config = SudokuVerifier.Config(width=16, depth=1, heads=2).finalize()
    analytical = cost(config, batch_size=2, dtype=None)
    kernel = attention_kernel_cost(
        seq_len=81,
        batch_size=2,
        dtype=None,
        num_heads=2,
        channels_head=8,
    )
    torch.manual_seed(0)
    module = config.make()
    module.train()
    with FlopCounterMode(display=False) as counter:
        module(
            torch.randint(1, 11, (2, 81)),
            torch.randint(2, 11, (2, 81)),
        ).sum().backward()
    assert analytical.params == sum(p.numel() for p in module.parameters())
    assert (
        analytical["flops", "matmul"].sum() - kernel["flops", "matmul"].sum()
        == counter.get_total_flops()
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
