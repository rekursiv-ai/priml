"""Replay the maze rungs that run new code bit for bit, and test the boundary term.

``testdata/exp000.pt``, ``exp001.pt`` and ``exp002.pt`` record the baseline, the
first rung on the Universal Reasoning Model recurrence and the boundary loss, and
the one-hour best, each shrunk by size only. Before the first goldens were
minted, each recipe matched recordings from the implementation that trained the
recorded results tensor for tensor; the current ones are this port's own record
at a smaller size. ``exp003`` changes a count and the budget, not code, so it has
no golden of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from torch import Tensor

import pytest
import torch

from priml.baselines.arcagi.model import ConvSwiGLU, SingleLatentRecurrence
from priml.baselines.maze import experiments
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE, RoPEMixed
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    joined,
    mismatches,
    put_steps,
    read_tensors,
    rng_fingerprint,
    stored,
)
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.maze.experiments import MazeTrainLoop
    from priml.baselines.maze.train_step import BoundaryLossTrainStep


_CWD: Final = Path(__file__).resolve().parent

SIDE: Final = 4
SLOTS: Final = 3
WIDTH: Final = 8
TRAIN_STEPS: Final = 2

RUNGS: Final[dict[str, Callable[[], MazeTrainLoop]]] = {
    "exp000": experiments.exp000,
    "exp001": experiments.exp001,
    "exp002": experiments.exp002,
}


def shrunk(factory: Callable[[], MazeTrainLoop]) -> BoundaryLossTrainStep.Config:
    """Return the rung's finalized step on a 4x4 grid, width 8, three slots.

    The pool runs at most two ACT steps.

    Cycle counts drop to at most two slow by one fast, and two fast where the
    rung carries gradient through the last passes, so that split still has a pass
    on each side.
    """
    cfg = factory()
    cfg.dataset.spec.grid_shape = (SIDE, SIDE)
    step = cfg.step
    parallelism = step.parallelism
    assert isinstance(parallelism, NoParallel.Config)
    parallelism.device = "cpu"
    step.pool.batch_size = SLOTS
    step.pool.max_steps = 2
    model = step.model
    model.channels_in = WIDTH
    model.num_layers = 1
    model.compile_core = None
    prefix = model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    prefix.num_tokens = min(prefix.num_tokens, 2)
    if model.rope_grid_shape:
        model.rope_grid_shape = (SIDE * SIDE,)
    recurrence = model.recurrence
    assert isinstance(recurrence, DeepRecurrence.Config)
    recurrence.slow_cycles = min(recurrence.slow_cycles, 2)
    recurrence.fast_cycles = 1
    if (
        isinstance(recurrence, SingleLatentRecurrence.Config)
        and recurrence.inner_grad_loops
    ):
        recurrence.fast_cycles = 2
        recurrence.inner_grad_loops = 1
    rope = model.rope
    assert isinstance(rope, RoPE.Config)
    rope.channels_head = WIDTH // 2
    if isinstance(rope, RoPEMixed.Config):
        rope.num_heads = 2
    block = model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, Attention.Config)
    block.attn.num_heads = 2
    block.attn.channels_head = WIDTH // 2
    assert isinstance(block.ffn, SwiGLU.Config | ConvSwiGLU.Config)
    block.ffn.round_to = WIDTH
    return cfg.copy_tree().finalize().step


def batches() -> list[dict[str, object]]:
    """Two training batches, the second short, then one evaluation batch."""
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "media": torch.randint(1, 6, (SLOTS, SIDE * SIDE), generator=generator),
            "label": torch.randint(1, 6, (SLOTS, SIDE * SIDE), generator=generator),
            "valid_count": SLOTS - 1 if index == 1 else SLOTS,
            "puzzle_identifiers": torch.zeros(SLOTS, dtype=torch.int32),
        }
        for index in range(TRAIN_STEPS + 1)
    ]


def record(config: BoundaryLossTrainStep.Config) -> dict[str, Tensor]:
    """Build the step from seed 0, train it two steps, then evaluate once.

    Args:
      config: Finalized step to record.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    out: dict[str, Tensor] = {}
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = config.make()
        out["rng"] = rng_fingerprint()
        out["init"] = joined(step.model.parameters())
        data = batches()
        steps: list[dict[str, Tensor]] = []
        for batch in data[:TRAIN_STEPS]:
            result = step.train_step(**batch)
            # The stablemax loss sums in float64, whose last bit follows the
            # host's libm; rounding to float32 is what makes it portable.
            steps.append(
                {
                    "loss": result["loss"].detach().float(),
                    "steps": step.pool.steps.clone(),
                    "halted": step.pool.halted.clone(),
                },
            )
        put_steps(out, "train", steps)
        out["pool/z_slow"] = stored(step.pool.z_slow)
        out["post"] = joined(step.model.parameters())
        out["ema"] = joined((step.ema_shadow or {}).values())
        evaluation = step.eval_loss(**data[-1])
        out["eval/loss"] = evaluation["loss"].detach()
        out["eval/model"] = stored(evaluation["model"])
    return out


@pytest.mark.parametrize("name", RUNGS)
def test_the_rung_replays_bit_for_bit(name: str) -> None:
    assert_tensor_golden(_CWD / "testdata" / f"{name}.pt", record(shrunk(RUNGS[name])))


@pytest.mark.parametrize("name", RUNGS)
def test_the_golden_bites(name: str) -> None:
    """A one-ULP change to one initial weight fails the golden.

    Only the initial weights are recorded: the replay above already shows an
    unchanged rung reproduces every key.
    """
    expected = read_tensors(_CWD / "testdata" / f"{name}.pt")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        step = shrunk(RUNGS[name]).make()
        with torch.no_grad():
            next(step.model.parameters()).view(-1).view(torch.int32)[0] += 1
        init = joined(step.model.parameters())
    assert mismatches({"init": expected["init"]}, {"init": init})


def test_the_boundary_loss_joins_at_its_weight() -> None:
    """The step's loss is the reference loss plus the weighted boundary mean."""
    config = shrunk(experiments.exp001)
    assert config.boundary is not None
    config.boundary.weight = 0.25
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        result = config.make().train_step(**batches()[0])
    metrics = result.get("metrics", {})
    halting = config.pool.halting
    assert halting is not None
    expected = (
        torch.as_tensor(metrics["lm_loss"])
        + halting.weight * torch.as_tensor(metrics["q_halt_loss"])
        + 0.25 * torch.as_tensor(metrics["canonical_boundary_loss"])
    )
    assert torch.equal(result["loss"].reshape(()), expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
