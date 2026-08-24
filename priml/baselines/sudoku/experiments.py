"""The sudoku experiment ladder.

Two mechanisms vary, independently, so the ladder is a 2x2 rather than a chain:

* which block mixes the grid tokens -- transformer or MLP-mixer,
* whether the solver runs a recurrence with adaptive computation time.

::

              transformer   MLP-mixer
    plain       exp000        exp001
    recurrent   exp002        exp003

Both axes are config VALUES -- a slot filled differently -- so the four share
one model class, one train step, and one dataset. That is what makes the
comparison meaningful: nothing differs except the thing named.

``exp000`` is the naive recipe and is never edited; improvements are forks, so
a number measured against it stays comparable. Run any of them::

    uv --quiet run --frozen python -m priml \
        priml.baselines.sudoku.experiments.exp000
"""

from __future__ import annotations

from dataclasses import field
from typing import Final

from configgle import Makes

from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.data import SudokuData
from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridEmbedding,
    PredictionFeedback,
)
from priml.baselines.sudoku.metric import GridAccuracy
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


GRID_LEN: Final = 81
"""Cells in a sudoku grid, fixed by the puzzle."""


class SudokuTrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """A training loop with the sudoku step and dataset already in place.

    Narrowing the two slots here rather than at each call site is what lets a
    factory reach ``cfg.step.model`` directly, with no ``isinstance`` narrow
    before a field it is about to set.
    """

    step: SudokuTrainStep.Config = field(default_factory=SudokuTrainStep.Config)
    """Model, optimization, and the optional recurrence."""

    dataset: SudokuData.Config = field(default_factory=SudokuData.Config)
    """Prepared sudoku puzzles, served from device memory."""


def exp000() -> SudokuTrainLoop:
    """Post-norm transformer over the grid, one forward per puzzle.

    The baseline every other experiment forks, and the only one stating a
    recipe rather than a change. Frozen: improvements belong in a fork, so a
    result measured against it stays comparable.

    Hypothesis:
      A plain transformer with learned row/column/box positions, AdamW on the
      lookup tables and Muon on the reasoning matrices, is the strongest recipe
      that uses nothing exotic -- the bar recurrence must clear to earn its
      cost.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    Results:
      TBD.

    """
    cfg = SudokuTrainLoop()
    cfg.study_name = "sudoku"
    cfg.experiment_name = "exp000"

    embedding = GridEmbedding.Config()
    # Sudoku's constraints are row, column, and box, so positions are described
    # by which of each a cell belongs to rather than by its index in a flattened
    # sequence.
    embedding.channels = [FactoredPositions.Config()]
    cfg.step.model.embedding = embedding
    cfg.step.model.channels_in = 512
    cfg.step.model.num_layers = 2

    cfg.dataset.batch_size = 384
    cfg.dataset.seed = 0
    # Mid-training evaluation reads a fixed prefix of the test split as a proxy;
    # the reported number is measured on the whole thing, and the two
    # populations are not comparable.
    cfg.dataset.num_eval_puzzles = 2_000

    cfg.metrics["accuracy"] = GridAccuracy.Config()
    cfg.max_steps = cfg.step.total_train_steps = 19_500
    cfg.num_steps_eval = 1_000
    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> SudokuTrainLoop:
    """exp000 with an MLP-mixer block instead of attention.

    Hypothesis:
      A sudoku grid is a fixed 81 cells in a fixed arrangement, so the content
      addressing attention buys may be unnecessary: a learned mixing over
      positions can express the same row/column/box routing at lower cost. If
      so, the mixer matches the transformer, and attention is not what makes
      this task work.

    References:
      https://arxiv.org/abs/2105.01601
        Tolstikhin et al. MLP-Mixer: An all-MLP Architecture for Vision.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model.block = _mixer_block()
    return cfg


def exp002() -> SudokuTrainLoop:
    """exp000 plus deep recurrence with adaptive computation time.

    Hypothesis:
      Constraint propagation is iterative -- filling one cell licenses filling
      the next -- so a fixed-depth network must learn in one pass what a
      recurrence can unroll. Re-applying a small stack over a carried latent,
      and letting each puzzle choose its own depth, should beat the same
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
    cfg.step.model.recurrence = DeepRecurrence.Config()
    # The solver refines its own answer, so the previous step's decoded grid is
    # an input channel: without it each step re-reads the original puzzle and
    # the recurrence carries belief only in the latent.
    embedding = cfg.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    embedding.channels = [FactoredPositions.Config(), PredictionFeedback.Config()]
    cfg.step.act = ActPool.Config(batch_size=cfg.dataset.batch_size)
    return cfg


def exp003() -> SudokuTrainLoop:
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
    cfg.step.model.block = _mixer_block()
    return cfg


def exp_smoke() -> SudokuTrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data prepared and does the
    loop run -- so every axis that costs time without bearing on that answer is
    cut: a few steps, and a network narrow enough to finish in seconds.
    Accuracy will be poor, which is expected.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.step.model.channels_in = 32
    cfg.step.model.num_layers = 1
    cfg.dataset.batch_size = 8
    cfg.dataset.num_train_puzzles = 4
    cfg.dataset.num_eval_puzzles = 4
    cfg.max_steps = cfg.step.total_train_steps = 4
    cfg.num_steps_eval = 2
    return cfg


def _mixer_block() -> MLPMixerBlock.Config:
    """An MLP-mixer block shaped for the sudoku grid.

    Post-norm, matching the transformer default: a recurrence feeds a block its
    own output, and an unnormalized residual stream compounds when it does.
    """
    return MLPMixerBlock.Config(
        seq_len=GRID_LEN,
        prenorm=False,
        token_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
        channel_mixer=SwiGLU.Config(norm=RMSNorm.Config()),
    )
