from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch

from priml.baselines.sudoku.routing import SingleChain


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.sudoku.act import AtomicPool
    from priml.baselines.sudoku.latent_recursion import LatentRecursion


class _Model:
    def __init__(self, *, puzzle_emb: object) -> None:
        self.puzzle_emb = puzzle_emb
        self.puzzle_ids: Tensor | None = None

    def __call__(
        self,
        inputs: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        puzzle_ids: Tensor | None,
    ) -> dict[str, Tensor]:
        self.puzzle_ids = puzzle_ids
        return {
            "logits": inputs.float() + 1,
            "q_halt": inputs.float().sum(dim=-1),
            "z_slow": z_slow * 2,
            "z_fast": z_fast * 3,
        }


def _pool() -> AtomicPool:
    return cast(
        "AtomicPool",
        SimpleNamespace(
            inputs=torch.arange(6).reshape(2, 3),
            z_slow=torch.ones(2, 4, 5),
            z_fast=torch.full((2, 4, 5), 2.0),
            puzzle_ids=torch.tensor([7, 9]),
        ),
    )


def test_single_chain_trains_and_carries_the_pools_one_chain() -> None:
    model = _Model(puzzle_emb=None)
    pool = _pool()
    chain = SingleChain.Config().make()
    chain.seat(cast("LatentRecursion", model), pool, global_step=3)
    routed = chain(
        cast("LatentRecursion", model),
        pool,
        token_loss=torch.sub,
        ignore_index=-100,
    )
    assert model.puzzle_ids is None
    assert torch.equal(routed.logits, pool.inputs.float() + 1)
    assert torch.equal(routed.q_halt, torch.tensor([3.0, 12.0]))
    assert torch.equal(routed.z_slow, pool.z_slow * 2)
    assert torch.equal(routed.z_fast, pool.z_fast * 3)
    assert routed.metrics == {}


def test_single_chain_passes_puzzle_ids_only_to_a_model_with_a_table() -> None:
    model = _Model(puzzle_emb=object())
    pool = _pool()
    SingleChain.Config().make()(
        cast("LatentRecursion", model),
        pool,
        token_loss=torch.sub,
        ignore_index=-100,
    )
    assert model.puzzle_ids is not None
    assert model.puzzle_ids.tolist() == [7, 9]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
