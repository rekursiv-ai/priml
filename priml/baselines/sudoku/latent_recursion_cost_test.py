"""LatentRecursion configs report the cost torch measures."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.testing.cost import assert_cost_matches_torch


def test_unfilled_recursion_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocabulary and grid shape"):
        LatentRecursion.Config().make()


def test_latent_recursion_cost_matches_torch() -> None:
    config = LatentRecursion.Config(
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
        assert isinstance(module, LatentRecursion)
        z_slow, z_fast = module.init_z(2)
        out = module(tokens, z_slow, z_fast, torch.tensor([1, 2]), feedback_ids=tokens)
        logits, q_halt = out["logits"], out["q_halt"]
        return logits.sum() + q_halt.sum()

    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        batch_size=2,
        dtype=None,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
