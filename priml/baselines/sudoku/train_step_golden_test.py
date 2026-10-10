"""Replay the recurrent sudoku step bit for bit.

``testdata/train_step.pt`` records the shared step with deep recurrence and
the ACT pool through :func:`record`, shrunk by size only. It was recorded from
the implementation that trained the recipe, so the pool settings that
reproduce it (no fed-back grid, zero-seeded slots) are pinned in
:func:`shrunk`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor

import torch

from priml.baselines.sudoku.act import (
    AtomicPool,
    HaltTraining,
    SampledMinimum,
    ZeroStart,
)
from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridEmbedding,
    PredictionFeedback,
)
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.attention.attention import Attention
from priml.model.init import kaiming_uniform
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


_CWD: Final = Path(__file__).resolve().parent

GOLDEN: Final = _CWD / "testdata" / "train_step.pt"

SLOTS: Final = 2
VOCAB: Final = 6
TRAIN_STEPS: Final = 3


def shrunk() -> SudokuTrainStep.Config:
    """Return the recurrent step on a 4x4 grid, width 2, two heads, two slots.

    The recipe the golden recorded: a post-norm transformer over the grid with
    factored positions and a built-but-unfed feedback channel, deep
    recurrence, and an ACT pool seated from zero latents with sampled-minimum
    halt exploration.
    """
    step = SudokuTrainStep.Config()
    step.parallelism = NoParallel.Config(device="cpu")
    step.compile = None
    step.total_train_steps = TRAIN_STEPS
    model = step.model
    model.vocab_size = VOCAB
    model.channels_in = 2
    model.num_layers = 1
    model.recurrence = DeepRecurrence.Config()
    positions = FactoredPositions.Config()
    positions.grid_shape = (4, 4)
    positions.box_shape = (2, 2)
    embedding = model.embedding = GridEmbedding.Config()
    embedding.grid_shape = (16,)
    embedding.channels = [positions, PredictionFeedback.Config()]
    assert isinstance(model.block, TransformerBlock.Config)
    model.block.ffn = SwiGLU.Config(
        init_weight=kaiming_uniform,
        init_weight_out=kaiming_uniform,
        round_to=2,
    )
    assert isinstance(model.block.attn, Attention.Config)
    model.block.attn.num_heads = 2
    step.pool = AtomicPool.Config(
        batch_size=SLOTS,
        max_steps=2,
        halting=HaltTraining.Config(exploration=SampledMinimum.Config()),
        start=ZeroStart.Config(),
    )
    return step


def batches() -> list[dict[str, object]]:
    """Three train batches, the second short, then one evaluation batch."""
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "media": torch.randint(2, VOCAB, (SLOTS, 16), generator=generator),
            "label": torch.randint(2, VOCAB, (SLOTS, 16), generator=generator),
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
    config = shrunk().copy_tree().finalize()
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


def test_recurrent_step_replays_bit_for_bit() -> None:
    """The recurrent step reproduces its recorded trajectory exactly."""
    assert_tensor_golden(GOLDEN, record())


def test_recurrent_step_golden_bites() -> None:
    """A one-ULP change to one initial weight fails the golden."""
    with expect_golden_mismatch(match=r"\d+ mismatches:"):
        assert_tensor_golden(GOLDEN, record(perturb=True))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
