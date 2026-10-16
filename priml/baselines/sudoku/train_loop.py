"""The training loop with the sudoku step and puzzle dataset in place.

Its own module rather than ``experiments.py``, which holds the experiment
factories: the reproduction pipelines train a generator of this type, and
``experiments.py`` imports them.
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

import math

from configgle import Makes

from priml.baselines.sudoku.act import SlotScramble
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.routed_act_step import RoutedActTrainStep
from priml.train.train_loop import TrainLoop


class SudokuTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[RoutedActTrainStep.Config, PuzzleDataset.Config],
):
    """A training loop whose step and dataset are the sudoku recipe's.

    Narrowing both slots here lets a factory reach ``cfg.step.model`` with no
    ``isinstance`` before a field it is about to set.
    """

    step: RoutedActTrainStep.Config = field(default_factory=RoutedActTrainStep.Config)
    """The pooled ACT step with its routing slot."""

    dataset: PuzzleDataset.Config = field(default_factory=PuzzleDataset.Config)
    """Prepared puzzle grids, held on the device."""

    @override
    def finalize(self) -> Self:
        # The dataset owns the geometry and vocabulary: the model, the pool's
        # clue range and corruption tokens, and the cardinality groups are
        # laid out on it.
        spec = self.dataset.spec
        model = self.step.model
        model.puzzle_grid_shape = (math.prod(spec.grid_shape),)
        model.vocab_size = spec.vocab_size
        if model.pos2d_grid_shape is not None:
            model.pos2d_grid_shape = spec.grid_shape
            model.pos2d_box_shape = spec.box_shape
        # One batch size, three buffers: a pool seats a whole loader batch, and
        # the sparse table's gradient buffer holds one row per pool slot.
        pool = self.step.pool
        pool.batch_size = self.dataset.batch_size
        model.puzzle_emb_batch_size = self.dataset.batch_size
        feedback = pool.feedback
        if feedback is not None:
            feedback.givens = (2, spec.vocab_size - 1)
            if isinstance(feedback.corruption, SlotScramble.Config):
                feedback.corruption.high = spec.vocab_size
        if self.step.cardinality is not None:
            self.step.cardinality.spec = spec
        return super().finalize()
