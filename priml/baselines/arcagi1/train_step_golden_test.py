"""Replay the recurrent ARC rung on the shared sudoku step bit for bit.

``testdata/exp002.pt`` records ``exp002`` -- the shared step with deep
recurrence, the ACT pool, and the per-task table -- through :func:`record`,
shrunk by size only. It was recorded from the implementation that trained the
rung, so the pool settings that reproduce it (no fed-back grid, zero-seeded
slots) are pinned in the recipe rather than here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from torch import Tensor

import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.attention.attention import Attention
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    expect_golden_mismatch,
    joined,
    put_steps,
    rng_fingerprint,
    stored,
)
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from priml.baselines.arcagi1.experiments import ArcTrainLoop


_CWD: Final = Path(__file__).resolve().parent

GOLDEN: Final = _CWD / "testdata" / "exp002.pt"

SLOTS: Final = 3
VOCAB: Final = 12
"""ARC's vocabulary: pad, end-of-row, and ten colors."""

PUZZLES: Final = 5
TRAIN_STEPS: Final = 3


def shrunk() -> ArcTrainLoop:
    """Return ``exp002`` on a 3x3 grid, width 4, two heads, three slots."""
    cfg = experiments.exp002()
    cfg.dataset.spec.max_grid = 3
    step = cfg.step
    step.parallelism = NoParallel.Config(device="cpu")
    step.compile = None
    step.total_train_steps = TRAIN_STEPS
    model = step.model
    model.channels_in = 4
    model.num_layers = 1
    assert isinstance(model.embedding, GridEmbedding.Config)
    assert isinstance(model.block, TransformerBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    model.block.attn.num_heads = 2
    assert isinstance(model.block.ffn, SwiGLU.Config)
    model.block.ffn.round_to = 4
    assert isinstance(model.prefix, PrefixStack.Config)
    table = model.prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.num_puzzles = PUZZLES
    table.batch_size = SLOTS
    assert step.pool is not None
    step.pool.batch_size = SLOTS
    step.pool.max_steps = 3
    return cfg


def batches() -> list[dict[str, object]]:
    """Three train batches, the second short, then one evaluation batch."""
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "media": torch.randint(2, VOCAB, (SLOTS, 9), generator=generator),
            "label": torch.randint(2, VOCAB, (SLOTS, 9), generator=generator),
            "puzzle_identifiers": torch.randint(
                0,
                PUZZLES,
                (SLOTS,),
                generator=generator,
            ),
            "valid_count": SLOTS - 1 if index == 1 else SLOTS,
        }
        for index in range(TRAIN_STEPS + 1)
    ]


def record(*, perturb: bool = False) -> dict[str, Tensor]:
    """Build the shrunk rung from seed 0 and record its trajectory.

    Args:
      perturb: Move the first parameter's first element up one ULP after init.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    config = shrunk().copy_tree().finalize().step
    out: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = SudokuTrainStep(config)
        out["rng"] = rng_fingerprint()
        if perturb:
            with torch.no_grad():
                next(step.model.parameters()).view(-1).view(torch.int32)[0] += 1
        out["init"] = joined(step.model.parameters())
        data = batches()
        steps: list[dict[str, Tensor]] = []
        for batch in data[:TRAIN_STEPS]:
            result = step.train_step(**batch)
            pool = step.pool
            assert pool is not None
            steps.append(
                {
                    "loss": result["loss"].detach(),
                    "steps": pool.steps.clone(),
                    "halted": pool.halted.clone(),
                },
            )
        put_steps(out, "train", steps)
        pool = step.pool
        assert pool is not None
        out["pool/z_slow"] = stored(pool.z_slow)
        out["pool/z_fast"] = stored(pool.z_fast)
        out["post"] = joined(step.model.parameters())
        evaluation = step.eval_loss(**data[-1])
        out["eval/loss"] = evaluation["loss"].detach()
        out["eval/model"] = stored(evaluation["model"])
    return out


def test_exp002_replays_bit_for_bit() -> None:
    """The recurrent rung reproduces its recorded trajectory exactly."""
    assert_tensor_golden(GOLDEN, record())


def test_exp002_golden_bites() -> None:
    """A one-ULP change to one initial weight fails the golden."""
    with expect_golden_mismatch(match=r"\d+ mismatches:"):
        assert_tensor_golden(GOLDEN, record(perturb=True))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
