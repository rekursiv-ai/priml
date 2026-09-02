"""The ARC-AGI experiment ladder.

Each ARC task shows a few input/output grid pairs demonstrating a rule, then
asks for the rule applied to a held-out input. Nothing about the rule is
labelled, so the model must infer it from the examples -- which is why this is
a reasoning benchmark rather than a perception one.

The solver is the sudoku baseline's, with different values in its slots: a
30x30 grid instead of 9x9, twelve colors instead of ten digits, and a learned
per-task prefix sudoku has no use for. Two properties drive the rest:

* **Whole-grid scoring.** One wrong cell fails the puzzle.
* **Heavy augmentation.** Each task is stored many times under recolorings and
  dihedral transforms, so the answer is the consensus across those views
  (pass@K) rather than any single pass.

The ladder mirrors sudoku's: architecture and recurrence are independent slots,
so the same four-corner comparison holds on a harder benchmark.

Launch (8 GPUs)::

    uv --quiet run --frozen torchrun --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi1.experiments.exp000
"""

from __future__ import annotations

from dataclasses import field
from typing import Final

from configgle import Makes

from priml.baselines.arcagi1.data import ArcData
from priml.baselines.arcagi1.metric import PassK
from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.runtime import SingleProcess
from priml.train.checkpointing import Checkpointer
from priml.train.train_loop import TrainLoop


GRID_LEN: Final = 900
"""Cells in the 30x30 grid every ARC task is padded to."""

VOCAB_SIZE: Final = 12
"""Tokens: pad, a blank marker, and the ten ARC colors."""

NUM_PUZZLE_IDENTIFIERS: Final = 876_403
"""Distinct puzzle ids in ``arc1concept-aug-1000``: 876,402 puzzles plus blank.

A property of the prepared dataset, not a tunable. The build assigns ids once
across every split, so the largest is the puzzle count and the per-task table
needs a row for each -- a shorter table indexes off the end on the first batch
rather than training a smaller model. The dataset cannot supply this: a config
must build with no data on disk, so ``finalize`` may not read the tree."""


class ArcTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """A training loop with the ARC step and dataset already in place.

    Narrowing both slots here rather than at each call site lets a factory
    reach ``cfg.step.model`` directly, with no ``isinstance`` before a field it
    is about to set.
    """

    step: SudokuTrainStep.Config = field(default_factory=SudokuTrainStep.Config)
    """The shared puzzle step: solver, optimization, optional recurrence."""

    dataset: ArcData.Config = field(default_factory=ArcData.Config)
    """Augmented ARC tasks, grouped so a batch draws whole tasks."""


def exp000() -> ArcTrainLoop:
    """Post-norm transformer over the grid, one forward per puzzle.

    The baseline every other experiment forks, and the only one stating a
    recipe rather than a change. Frozen: improvements belong in a fork, so a
    result measured against it stays comparable.

    Hypothesis:
      A plain transformer with a learned per-task vector is the strongest
      recipe that uses nothing exotic -- the bar recurrence must clear on a
      benchmark whose tasks are genuinely novel at test time.

    References:
      https://arxiv.org/abs/1911.01547
        Chollet. On the Measure of Intelligence.

    Results:
      TBD.

    """
    cfg = ArcTrainLoop()
    cfg.study_name = "arcagi1"
    cfg.experiment_name = "exp000"
    cfg.seed = 0

    batch_size = 256
    model = cfg.step.model
    model.channels_in = 512
    model.num_layers = 2
    model.vocab_size = VOCAB_SIZE

    # Same embedding class as sudoku, a different grid: ARC pads every task to
    # 30x30, and its colors have no row/column/box structure to factor.
    embedding = GridEmbedding.Config()
    embedding.grid_shape = (GRID_LEN,)
    model.embedding = embedding

    # A per-task vector plus one register token. The task embedding comes
    # first, so it owns position 0 -- where the halt head reads.
    prefix = PrefixStack.Config()
    prefix.parts = [
        SparsePuzzleEmbedding.Config(
            num_puzzles=NUM_PUZZLE_IDENTIFIERS,
            num_tokens=16,
            batch_size=batch_size,
        ),
        RegisterTokens.Config(num_tokens=1),
    ]
    model.prefix = prefix

    cfg.step.total_train_steps = 388_670
    cfg.step.warmup_steps = 2_000
    cfg.step.ema_decay = 0.999
    cfg.step.ema_warmup_steps = 2_000

    cfg.dataset.batch_size = batch_size
    cfg.dataset.eval_batch_size = batch_size

    cfg.metrics["pass"] = PassK.Config()
    cfg.max_steps = cfg.step.total_train_steps
    cfg.num_steps_eval = 10_000
    cfg.num_steps_log = 100
    cfg.eval_warmup_batches = 1
    cfg.eval_every_epoch = False

    cfg.checkpointing = Checkpointer.Config()
    cfg.checkpointing.save_every = 4_000
    cfg.checkpointing.keep_last_n = 8
    cfg.checkpointing.keep_every = 40_000

    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> ArcTrainLoop:
    """exp000 with an MLP-mixer block instead of attention.

    Hypothesis:
      An ARC grid is a fixed 900 cells in a fixed arrangement, so the content
      addressing attention buys may be unnecessary: a learned mixing over
      positions can express the same spatial routing at lower cost.

    References:
      https://arxiv.org/abs/2105.01601
        Tolstikhin et al. MLP-Mixer: An all-MLP Architecture for Vision.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model.block = _mixer_block(cfg.step.model.total_seq_len)
    return cfg


def exp002() -> ArcTrainLoop:
    """exp000 plus deep recurrence with adaptive computation time.

    Hypothesis:
      An ARC rule is applied in steps -- find the shape, recolor it, place it
      -- so a fixed-depth network must learn in one pass what a recurrence can
      unroll. Letting each task choose its own depth should beat the same
      parameters spent in a single forward.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.step.model.recurrence = DeepRecurrence.Config(slow_cycles=3, fast_cycles=4)
    # The solver refines its own answer, so the previous step's decoded grid is
    # an input channel: without it each step re-reads the original task.
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    embedding.channels = [PredictionFeedback.Config()]
    cfg.step.act = ActPool.Config(
        batch_size=cfg.dataset.batch_size,
        max_steps=16,
        halt_weight=0.5,
    )
    # ARC's colors run to the end of the vocabulary, unlike sudoku's digits.
    cfg.step.act.given_high = VOCAB_SIZE - 1
    return cfg


def exp003() -> ArcTrainLoop:
    """exp002 with an MLP-mixer block: the fourth corner of the 2x2.

    Hypothesis:
      If recurrence is what makes the task work (exp002) and attention is not
      (exp001), the two gains are independent and the mixer keeps its recurrent
      gain. A drop here instead would mean the recurrence relies on attention
      specifically, not on depth.

    Results:
      TBD.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.step.model.block = _mixer_block(cfg.step.model.total_seq_len)
    return cfg


def exp_smoke() -> ArcTrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data staged and does the
    loop run -- so every axis that costs time without bearing on that answer is
    cut. Accuracy will be poor, which is expected.

    The per-task table keeps its full height. Capping tasks does not cap the
    ids they carry -- the build numbers puzzles once across every split -- so a
    table sized to ``num_tasks`` would index off the end. At this width it
    costs 112 MB and 35 ms, which does not bear on the question.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.model.channels_in = 32
    cfg.step.model.num_layers = 1
    cfg.dataset.batch_size = 8
    cfg.dataset.eval_batch_size = 8
    cfg.dataset.num_tasks = 4
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.num_steps_eval = 2
    cfg.checkpointing = None

    prefix = cfg.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.batch_size = cfg.dataset.batch_size
    return cfg


def _mixer_block(seq_len: int) -> MLPMixerBlock.Config:
    """An MLP-mixer block shaped for the padded grid plus its prefix.

    Post-norm, matching the transformer default: a recurrence feeds a block its
    own output, and an unnormalized residual stream compounds when it does.

    Args:
      seq_len: Prefix plus grid tokens. The mixer mixes ACROSS positions, so it
        must be built to the full sequence, not the grid alone.

    Returns:
      config: The block config.

    """
    return MLPMixerBlock.Config(
        seq_len=seq_len,
        prenorm=False,
        token_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
        channel_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
    )
